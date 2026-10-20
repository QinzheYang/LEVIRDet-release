'use strict';
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { EventEmitter } = require('node:events');
const { randomBytes } = require('node:crypto');
const transfers = require('./transfers.cjs');

const CODE_FILES = ['demo/image_demo.py', 'mmdet/__init__.py', 'mmdet/apis/levir_image_demo.py',
  'configs/_base_/levirdetnet.py', 'configs/levirdetnet/levirdetnet-30class.py',
  'configs/levirdetnet/levirdetnet-159class.py', 'fast_demo/server.py',
  'fast_demo/assets.json', 'fast_demo/desktop/main.cjs', 'fast_demo/desktop/preload.cjs',
  'fast_demo/desktop/package.json', 'fast_demo/static/index.html', 'fast_demo/static/app.js', 'fast_demo/static/style.css'];
const names = { docker: 'Docker environment', weights30: '30-class model', weights159: '159-class model' };
async function exists(file) { try { return (await fsp.stat(file)).isFile(); } catch { return false; } }
function inside(root, relative) {
  const result = path.resolve(root, relative);
  const rel = path.relative(root, result);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Invalid resource path.');
  return result;
}
async function rejectLinkedDestination(root, target) {
  const relative = path.relative(root, target);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Destination is outside the selected directory.');
  let current = root;
  for (const component of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current, component);
    const stat = await fsp.lstat(current).catch(error => { if (error.code === 'ENOENT') return null; throw error; });
    if (stat?.isSymbolicLink()) throw new Error('A destination directory contains a symbolic link or junction: ' + current);
  }
}
async function readJSON(file, fallback) { try { return JSON.parse((await fsp.readFile(file, 'utf8')).replace(/^\uFEFF/, '')); } catch { return fallback; } }
async function writeJSON(file, value) {
  await fsp.mkdir(path.dirname(file), { recursive: true });
  const temp = file + '.' + randomBytes(4).toString('hex') + '.tmp';
  await fsp.writeFile(temp, JSON.stringify(value, null, 2));
  await fsp.rename(temp, file);
}
async function availableSpace(root) {
  let nearest = root;
  while (!fs.existsSync(nearest) && path.dirname(nearest) !== nearest) nearest = path.dirname(nearest);
  try { const stat = await fsp.statfs(nearest); return stat.bavail * stat.bsize; } catch { return null; }
}
function processRun(executable, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { windowsHide: true, shell: false, ...options });
    let stdout = '', stderr = '';
    child.stdout?.on('data', data => { stdout = (stdout + data.toString()).slice(-200000); options.onOutput?.(data.toString(), false); });
    child.stderr?.on('data', data => { stderr = (stderr + data.toString()).slice(-200000); options.onOutput?.(data.toString(), true); });
    child.on('error', reject);
    child.on('close', code => {
      if (code === 0 || options.allowFailure) resolve({ code, stdout, stderr });
      else reject(new Error((stderr || stdout || `${path.basename(executable)} exited with code ${code}`).trim().slice(-5000)));
    });
  });
}
function waitForDesktop({ child, readyFile, nonce, timeoutMs = 45000, pollMs = 150 }) {
  return new Promise((resolve, reject) => {
    let timer, finished = false;
    const deadline = Date.now() + timeoutMs;
    const finish = (error, value) => {
      if (finished) return;
      finished = true; clearTimeout(timer);
      child.removeListener('error', failed); child.removeListener('exit', exited);
      error ? reject(error) : resolve(value);
    };
    const failed = error => finish(error);
    const exited = (code, signal) => finish(new Error(`The inference window exited before it was ready (${signal || code}).`));
    child.once('error', failed); child.once('exit', exited);
    const check = async () => {
      if (finished) return;
      if (child.exitCode !== null && child.exitCode !== undefined) return exited(child.exitCode);
      const status = await readJSON(readyFile, null);
      if (finished) return;
      if (status?.nonce === nonce && status.pid === child.pid) {
        if (status.error) return finish(new Error(status.error));
        if (status.visible === true && status.loaded === true) return finish(null, status);
      }
      if (Date.now() >= deadline) return finish(new Error('The inference window did not confirm that it was visible and loaded.'));
      timer = setTimeout(check, pollMs);
    };
    void check();
  });
}

class LauncherController extends EventEmitter {
  constructor(options) {
    super();
    this.appRoot = options.appRoot;
    this.launcherRoot = options.launcherRoot;
    this.launcherFile = options.launcherFile;
    this.settingsFile = options.settingsFile;
    this.defaultRoot = options.defaultRoot;
    this.electronPath = options.electronPath;
    this.locale = options.locale;
    this.manifest = options.manifest;
    this.runner = options.runner || processRun;
    this.requestTransport = options.requestTransport;
    this.io = options.transfers || transfers;
    this.session = null;
    this.abort = null;
    this.docker = null;
    this.state = { phase: 'idle', busy: false, canPause: false, installRoot: '', projectRoot: '',
      language: this.locale?.startsWith('zh') ? 'zh' : 'en', assets: [], progress: null,
      message: '', error: null, logs: [], freeBytes: null, requiredBytes: 0, ready: false, codeReady: false };
  }
  text(zh, en) { return this.state.language === 'zh' ? zh : en; }
  snapshot() { return JSON.parse(JSON.stringify(this.state)); }
  update(patch) { Object.assign(this.state, patch); this.emit('state', this.snapshot()); }
  log(text) {
    const clean = String(text).replace(/\x1b\[[0-9;]*m/g, '').trim();
    if (!clean) return;
    this.state.logs = [...this.state.logs, clean.slice(-2000)].slice(-25);
    this.emit('state', this.snapshot());
  }
  async init() {
    const settings = await readJSON(this.settingsFile, {});
    this.state.language = ['zh', 'en'].includes(settings.language) ? settings.language : this.state.language;
    this.state.installRoot = path.resolve(settings.installRoot || this.defaultRoot);
    this.state.projectRoot = this.state.installRoot;
    await this.refresh();
    return this.snapshot();
  }
  async saveSettings() {
    await writeJSON(this.settingsFile, { version: 1, installRoot: this.state.installRoot, language: this.state.language });
  }
  async setLanguage(language) {
    if (!['zh', 'en'].includes(language)) throw new Error('Unsupported language.');
    this.update({ language }); await this.saveSettings();
  }
  async setPath(root) {
    if (this.state.busy || this.session) throw new Error(this.text('请先暂停当前任务。', 'Pause the current operation first.'));
    root = path.resolve(root);
    if (root.includes(',') || /[\r\n]/.test(root)) throw new Error(this.text('路径不能包含逗号或换行。', 'Choose a path without commas or line breaks.'));
    this.update({ installRoot: root, projectRoot: root, error: null, phase: 'idle', progress: null });
    await this.saveSettings(); await this.refresh();
  }
  async cached(asset) {
    const target = inside(this.state.installRoot, asset.path);
    const stat = await fsp.stat(target, { bigint: true }).catch(() => null);
    if (!stat || stat.size !== BigInt(asset.size)) return false;
    const own = await readJSON(path.join(this.state.installRoot, '.fast_demo_cache', 'launcher-verified.json'), {});
    const record = own[asset.id];
    if (record?.invalid) return false;
    if (record?.sha256 === asset.sha256 && record.path === target && record.mtimeNs === stat.mtimeNs.toString()) return true;
    // Reuse the previous launcher's verified cache without rereading 13 GiB.
    if (process.platform === 'win32') {
      const prior = await readJSON(path.join(this.state.installRoot, '.fast_demo_cache', asset.id + '.json'), {});
      const ticks = (stat.mtimeNs / 100n + 621355968000000000n).toString();
      if (prior.sha256 === asset.sha256 && prior.size === asset.size && String(prior.mtime) === ticks &&
          String(prior.path).toLowerCase() === target.toLowerCase()) return true;
    }
    return false;
  }
  async remember(asset) {
    const target = inside(this.state.installRoot, asset.path);
    const stat = await fsp.stat(target, { bigint: true });
    const file = path.join(this.state.installRoot, '.fast_demo_cache', 'launcher-verified.json');
    const records = await readJSON(file, {});
    records[asset.id] = { path: target, size: asset.size, sha256: asset.sha256, mtimeNs: stat.mtimeNs.toString() };
    await writeJSON(file, records);
  }
  async invalidate(asset) {
    const file = path.join(this.state.installRoot, '.fast_demo_cache', 'launcher-verified.json');
    const records = await readJSON(file, {});
    records[asset.id] = { invalid: true };
    await writeJSON(file, records);
    this.update({ ready: false });
  }
  async codePresent(root = this.state.installRoot) {
    return (await Promise.all(CODE_FILES.map(file => exists(path.join(root, file))))).every(Boolean);
  }
  async refresh() {
    const root = this.state.installRoot;
    const assets = [];
    for (const asset of this.manifest.assets) {
      const target = inside(root, asset.path);
      const stat = await fsp.stat(target).catch(() => null);
      const partial = await fsp.stat(target + '.part').catch(() => null);
      const verified = await this.cached(asset);
      assets.push({ id: asset.id, name: names[asset.id], path: target, size: asset.size,
        downloaded: stat?.size || partial?.size || 0,
        status: verified ? 'verified' : stat ? (stat.size === asset.size ? 'available' : 'invalid') : partial ? 'partial' : 'missing' });
    }
    const codeReady = await this.codePresent();
    assets.push({ id: 'code', name: 'LEVIRDetNet code', path: root, size: null, downloaded: null, status: codeReady ? 'available' : 'missing' });
    this.update({ assets, codeReady, ready: codeReady && assets.slice(0, 3).every(a => a.status === 'verified'),
      freeBytes: await availableSpace(root),
      requiredBytes: assets.slice(0, 3).reduce((sum, a) => sum + (a.status === 'missing' || a.status === 'partial' ? Math.max(0, a.size - a.downloaded) : 0), 0) + (codeReady ? 0 : 512 * 1024 ** 2) });
  }
  progress(item, phase) {
    return event => {
      const currentPhase = event.phase === 'verify' ? 'verifying' : phase;
      const label = event.source ? `${item} · ${event.source}` : item;
      const message = event.phase === 'verify' ? this.text('正在校验文件大小和 SHA-256…', 'Verifying file size and SHA-256…') : event.phase === 'source' ? (event.reason ? this.text(`当前下载源速度较慢或不可用，正在切换到 ${event.source}…`, `The previous source is slow or unavailable. Switching to ${event.source}…`) : this.text(`正在连接 ${event.source}…`, `Connecting to ${event.source}…`)) :
        event.source && event.phase === 'download' ? this.text(`正在从 ${event.source} 下载`, `Downloading from ${event.source}`) : undefined;
      if (event.phase === 'source' && event.reason) this.log(`${event.source}: ${event.reason}`);
      this.update({ phase: currentPhase, canPause: ['downloading', 'verifying'].includes(currentPhase), progress: { ...event, item: label }, ...(message ? { message } : {}) });
    };
  }
  async prerequisites(mode = 'Ensure') {
    this.currentResource = { url: process.platform === 'win32' ? 'https://docs.docker.com/desktop/setup/install/windows-install/' :
      'https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html',
      path: path.join(this.state.installRoot, '.fast_demo_setup') };
    this.update({ phase: 'prerequisites', canPause: false, progress: null,
      message: this.text('检查 Docker 和 GPU 前置条件；缺失时会询问是否安装。', 'Checking Docker and GPU prerequisites; missing components require your consent.') });
    const windows = process.platform === 'win32';
    const executable = windows ? path.join(process.env.WINDIR || 'C:\\Windows', 'System32/WindowsPowerShell/v1.0/powershell.exe') : '/bin/bash';
    const args = windows ? ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(this.appRoot, 'prerequisites.ps1'), '-Mode', mode, '-SetupRoot', this.state.installRoot] :
      [path.join(this.appRoot, 'prerequisites.sh'), '--mode', mode, '--setup-root', this.state.installRoot];
    let buffer = '';
    const result = await this.runner(executable, args, { onOutput: chunk => {
      buffer += chunk;
      const lines = buffer.split(/\r?\n/); buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('LEVIR_EVENT ')) {
          try { const event = JSON.parse(line.slice(12)); this.update({ message: event.message || this.state.message,
            progress: event.total || event.received ? { ...event, item: event.item || 'Docker Desktop' } : null }); } catch { this.log(line); }
        } else this.log(line);
      }
    }, allowFailure: true });
    let info;
    for (const line of result.stdout.split(/\r?\n/).reverse()) { try { const parsed = JSON.parse(line); if ('ok' in parsed) { info = parsed; break; } } catch { } }
    if (!info?.ok) throw new Error(Array.isArray(info?.details) ? info.details.join('\n') : info?.details || result.stderr || this.text('前置环境尚未就绪，请查看安装提示。', 'Prerequisites are not ready. Review the setup instructions.'));
    this.docker = info.dockerPath || 'docker';
  }
  async dockerRun(args, options = {}) {
    return this.runner(this.docker || 'docker', args, { ...options, onOutput: text => { this.log(text); options.onOutput?.(text); } });
  }
  async ensureImage() {
    const inspect = await this.dockerRun(['image', 'inspect', this.manifest.docker_image, '--format', '{{.Id}}'], { allowFailure: true });
    if (inspect.code !== 0) {
      this.update({ phase: 'loading', canPause: false, progress: null, message: this.text('正在导入 Docker 环境镜像…', 'Importing the Docker environment…') });
      const asset = this.manifest.assets.find(item => item.id === 'docker');
      await this.dockerRun(['load', '--input', inside(this.state.installRoot, asset.path)]);
    }
    await this.dockerRun(['run', '--rm', '--pull', 'never', '--gpus', 'all', '--entrypoint', 'nvidia-smi', this.manifest.docker_image, '-L']);
  }
  async resource(asset, force = false) {
    const target = inside(this.state.installRoot, asset.path);
    this.currentResource = { url: asset.url, path: target };
    if (!force && await this.cached(asset)) return;
    const signal = this.abort.signal;
    if (await exists(target)) {
      this.update({ phase: 'verifying', message: this.text('正在校验已有资源', 'Verifying existing resources'), canPause: true });
      const result = await this.io.verify({ path: target, size: asset.size, sha256: asset.sha256, signal,
        onProgress: this.progress(names[asset.id], 'verifying') });
      if (!result.ok) {
        await this.invalidate(asset);
        throw new Error(this.text('文件校验失败，请将原文件移开后重新下载：', 'File verification failed. Move the incorrect file aside and retry: ') + target);
      }
    } else {
      if (force) throw new Error(this.text('缺少资源：', 'Missing resource: ') + target);
      const free = await availableSpace(this.state.installRoot);
      const partial = await fsp.stat(target + '.part').catch(() => null);
      if (free !== null && free < Math.max(0, asset.size - (partial?.size || 0)) + 128 * 1024 ** 2) throw new Error(this.text('下载目录空间不足，请更改路径。', 'Not enough disk space. Choose another download folder.'));
      this.update({ phase: 'downloading', message: this.text('正在下载资源', 'Downloading resources'), canPause: true });
      const showProgress = this.progress(names[asset.id], 'downloading');
      showProgress({ phase: 'source', source: asset.drive_id ? 'Google Drive' : 'Primary', received: partial?.size || 0,
        total: asset.size, percent: asset.size ? Math.min(100, (partial?.size || 0) / asset.size * 100) : null, speed: 0, eta: null });
      await this.io.downloadWithMirrors({ url: asset.url, destination: target, driveId: asset.drive_id, size: asset.size,
        mirrors: asset.mirrors || [], sha256: asset.sha256, signal, requestTransport: this.requestTransport, onProgress: event => {
          if (event.source) this.currentResource = { url: asset.mirrors?.find(mirror => mirror.name === event.source)?.url || asset.url, path: target };
          showProgress(event);
        } });
    }
    await this.remember(asset); await this.refresh();
  }
  async installCode() {
    if (await this.codePresent()) return;
    const root = this.state.installRoot;
    const cache = path.join(root, '.fast_demo_cache');
    const archive = path.join(cache, 'LEVIRDet-release-main.zip');
    this.currentResource = { url: this.manifest.repository.url, path: root };
    this.update({ phase: 'downloading', canPause: true, message: this.text('正在下载项目代码', 'Downloading project code') });
    if (!await exists(archive)) {
      await this.io.download({ url: this.manifest.repository.archive_url, destination: archive,
        signal: this.abort.signal, requestTransport: this.requestTransport, onProgress: this.progress('LEVIRDetNet code', 'downloading') });
    }
    this.update({ phase: 'extracting', canPause: false, progress: null, message: this.text('正在整理项目代码…', 'Extracting project code…') });
    const extract = path.join(cache, 'code-' + randomBytes(6).toString('hex'));
    await this.io.extractZip({ archive, destination: extract, signal: this.abort.signal });
    const children = await fsp.readdir(extract, { withFileTypes: true });
    const folders = children.filter(entry => entry.isDirectory());
    if (folders.length !== 1) throw new Error('The repository ZIP does not have one project directory.');
    const source = path.join(extract, folders[0].name);
    if (!await this.codePresent(source)) throw new Error(this.text('GitHub 代码尚未包含完整的推理入口，请更新仓库后重试。', 'The GitHub repository does not contain the required inference files. Publish the complete project and retry.'));
    // Enumerate conflicts before copying anything; never replace a user's code.
    const files = [];
    const walk = async (directory, relative = '') => {
      for (const entry of await fsp.readdir(directory, { withFileTypes: true })) {
        const rel = path.join(relative, entry.name);
        if (entry.isSymbolicLink()) throw new Error('Symbolic links are not accepted in the project archive.');
        if (entry.isDirectory()) await walk(path.join(directory, entry.name), rel);
        else if (entry.isFile()) files.push(rel);
      }
    };
    await walk(source);
    const missing = [];
    for (const relative of files) {
      const from = inside(source, relative), to = inside(root, relative);
      await rejectLinkedDestination(root, to);
      if (await exists(to)) {
        const [sourceStat, targetStat] = await Promise.all([fsp.stat(from), fsp.stat(to)]);
        let identical = sourceStat.size === targetStat.size;
        if (identical) {
          const a = await this.io.verify({ path: from });
          const b = await this.io.verify({ path: to, sha256: a.sha256 });
          identical = a.ok && b.ok;
        }
        if (!identical) throw new Error(this.text('已有代码与下载版本不同，请使用空目录或手动整理：', 'An existing file differs. Choose a fresh folder or reconcile it manually: ') + to);
      } else missing.push(relative);
    }
    for (const relative of missing) {
      const to = inside(root, relative);
      await rejectLinkedDestination(root, to);
      await fsp.mkdir(path.dirname(to), { recursive: true });
      await fsp.copyFile(inside(source, relative), to, fs.constants.COPYFILE_EXCL);
    }
  }
  async reportError(error) {
    const helpPath = path.join(this.state.installRoot, 'LEVIRDetNet-setup-help.txt');
    const lines = ['LEVIRDetNet Launcher', String(error.message || error), '', ...this.manifest.assets.map(asset =>
      `${names[asset.id]}\nGoogle Drive: ${asset.url}\n${(asset.mirrors || []).map(mirror => `${mirror.name}: ${mirror.url}`).join('\n')}\nSave as: ${inside(this.state.installRoot, asset.path)}\nSize: ${asset.size}\nSHA256: ${asset.sha256}\n`),
      `Project code: ${this.manifest.repository.url}\nExtract project contents into: ${this.state.installRoot}`,
      'Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/',
      'Linux GPU Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html'];
    await fsp.mkdir(this.state.installRoot, { recursive: true }).then(() => fsp.writeFile(helpPath, lines.join('\n'))).catch(() => {});
    this.update({ phase: 'error', busy: false, canPause: false, ready: false, message: '',
      error: { message: String(error.message || error), url: this.currentResource?.url || this.manifest.repository.url,
        path: this.currentResource?.path || this.state.installRoot, helpPath } });
  }
  async operation(work) {
    if (this.state.busy) return;
    this.abort = new AbortController(); this.currentResource = null;
    this.update({ busy: true, error: null, phase: 'checking', progress: null });
    try { await work(); }
    catch (error) {
      if (this.abort.signal.aborted) this.update({ phase: 'paused', busy: false, canPause: false, message: this.text('下载已暂停，已下载部分会保留。', 'Paused. Partial downloads are preserved.') });
      else await this.reportError(error);
    } finally { this.update({ busy: false, canPause: false }); }
  }
  async start() {
    if (this.state.ready) return this.launch();
    await this.operation(async () => {
      await fsp.mkdir(this.state.installRoot, { recursive: true });
      await this.saveSettings(); await this.prerequisites();
      for (const id of ['docker', 'weights30', 'weights159']) {
        await this.resource(this.manifest.assets.find(asset => asset.id === id));
        if (id === 'docker') await this.ensureImage();
      }
      await this.installCode(); await this.refresh();
      this.update({ phase: 'ready', ready: true, progress: null, message: this.text('资源已就绪，点击启动进入推理界面。', 'Resources are ready. Click Launch to open inference.') });
    });
  }
  pause() { if (this.state.canPause && this.abort) this.abort.abort(); }
  async integrity() {
    await this.operation(async () => {
      for (const asset of this.manifest.assets) await this.resource(asset, true);
      if (!await this.codePresent()) throw new Error(this.text('项目代码不完整，请点击下载补齐。', 'Project code is incomplete. Use Download to complete setup.'));
      await this.refresh();
      this.update({ phase: 'ready', progress: null, message: this.text('镜像和两个权重的 SHA-256 校验通过，代码入口完整。', 'Docker and both checkpoints passed SHA-256 verification; code entrypoints are present.') });
    });
  }
  async launch() {
    if (this.session) return;
    await this.operation(async () => {
      await this.prerequisites(); await this.ensureImage();
      this.update({ phase: 'launching', progress: null, message: this.text('正在启动推理程序…', 'Starting inference…') });
      const output = path.join(this.state.installRoot, 'fast_demo_outputs');
      await fsp.mkdir(path.join(output, '.runtime-home'), { recursive: true });
      const name = 'levir-launcher-' + randomBytes(6).toString('hex');
      const token = randomBytes(32).toString('hex');
      const args = ['run', '--detach', '--name', name, '--label', 'levir-launcher=true', '--init', '--pull', 'never', '--gpus', 'all', '--shm-size=4g',
        '--publish', '127.0.0.1::8765', '--workdir', '/workspace/levirdetnet',
        '--env', 'NO_ALBUMENTATIONS_UPDATE=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'LEVIR_DEMO_TOKEN=' + token, '--env', 'LEVIR_DEMO_HOST_OUTPUT=' + output,
        '--mount', `type=bind,source=${this.state.installRoot},target=/workspace/levirdetnet,readonly`,
        '--mount', `type=bind,source=${output},target=/fast_demo_outputs`];
      if (process.platform !== 'win32') args.push('--user', `${process.getuid()}:${process.getgid()}`, '--env', 'HOME=/fast_demo_outputs/.runtime-home');
      args.push(this.manifest.docker_image, 'python', '-u', 'fast_demo/server.py', '--host', '0.0.0.0', '--port', '8765', '--output-root', '/fast_demo_outputs');
      this.session = { name, child: null };
      try {
        await this.dockerRun(args);
        const address = (await this.dockerRun(['port', name, '8765/tcp'])).stdout.trim();
        if (!/^127\.0\.0\.1:\d+$/.test(address)) throw new Error('Docker did not publish a local-only port.');
        const url = 'http://' + address;
        let ready = false;
        for (let i = 0; i < 45; i++) {
          try { const response = await fetch(url + '/health', { signal: AbortSignal.timeout(1800) }); if (response.ok) { ready = true; break; } } catch { }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        if (!ready) throw new Error('The local inference service did not start.');
        const readyFile = path.join(output, '.runtime-home', name + '.ready.json');
        const logFile = path.join(output, '.runtime-home', name + '.desktop.log');
        const nonce = randomBytes(24).toString('hex');
        this.currentResource = { url: this.manifest.repository.url, path: logFile };
        this.update({ message: this.text('正在打开桌面窗口，等待界面加载…', 'Opening the desktop window and waiting for its interface…') });
        const logStream = fs.createWriteStream(logFile, { flags: 'a' });
        logStream.on('error', error => this.log('Desktop log: ' + error.message));
        const child = spawn(this.electronPath, [path.join(this.state.installRoot, 'fast_demo/desktop')], {
          cwd: this.state.installRoot, windowsHide: false,
          env: { ...process.env, LEVIR_DEMO_URL: `${url}/#token=${token}`, LEVIR_DEMO_OUTPUT: output,
            LEVIR_DEMO_READY_FILE: readyFile, LEVIR_DEMO_READY_NONCE: nonce }, stdio: ['ignore', 'pipe', 'pipe'] });
        for (const stream of [child.stdout, child.stderr]) stream.on('data', data => logStream.write(data.toString().split(token).join('[redacted]')));
        child.once('close', () => logStream.end());
        this.session.child = child;
        this.session.readyFile = readyFile;
        const session = this.session;
        child.once('error', error => {
          if (session.started && this.session === session) this.stopSession().finally(() => this.reportError(error));
        });
        child.once('exit', code => {
          if (session.started && this.session === session) this.stopSession().then(() => {
            if (code) return this.reportError(new Error(this.text('推理窗口异常退出，日志：', 'The inference window exited unexpectedly. Log: ') + logFile));
            this.update({ phase: 'ready', message: this.text('推理窗口已关闭。', 'Inference window closed.') });
          }).catch(error => this.reportError(error));
        });
        try { await waitForDesktop({ child, readyFile, nonce }); }
        catch (error) { throw new Error(error.message + this.text('\n窗口启动日志：', '\nDesktop startup log: ') + logFile); }
        if (this.session !== session || child.exitCode !== null) throw new Error('The inference window closed during startup.');
        session.started = true;
        this.update({ phase: 'running', message: this.text('推理窗口已打开。', 'Inference window is open.') });
      } catch (error) { await this.stopSession(); throw error; }
    });
  }
  async stopSession() {
    const session = this.session; if (!session) return;
    this.session = null;
    session.child?.kill();
    await this.dockerRun(['stop', '--timeout', '10', session.name], { allowFailure: true }).catch(() => {});
    await this.dockerRun(['rm', session.name], { allowFailure: true }).catch(() => {});
    if (session.readyFile) await fsp.unlink(session.readyFile).catch(() => {});
  }
}
module.exports = { LauncherController, processRun, CODE_FILES, availableSpace, waitForDesktop };
