'use strict';
const { app, BrowserWindow, ipcMain, dialog, shell, Menu, screen, net } = require('electron');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { LauncherController } = require('./controller.cjs');

const launcherRoot = path.resolve(process.env.LEVIR_LAUNCHER_ROOT || path.join(__dirname, '../..'));
const launcherFile = process.env.LEVIR_LAUNCHER_FILE || path.join(launcherRoot, process.platform === 'win32' ? 'launch_fast_demo.cmd' : 'launch_fast_demo.sh');
const settingsRoot = path.join(launcherRoot, '.fast_demo_setup');
app.setName('LEVIRDetNet Launcher');
app.setPath('userData', path.join(settingsRoot, 'launcher-profile'));
let window, controller, closing = false, launchPromise;
const pageURL = pathToFileURL(path.join(__dirname, 'index.html')).href;
function trusted(event) {
  if (!window || event.sender !== window.webContents || event.senderFrame?.url.split('#')[0] !== pageURL) throw new Error('Untrusted launcher request.');
}
async function shortcut() {
  const desktop = app.getPath('desktop');
  if (!await fsp.stat(launcherFile).catch(() => null)) throw new Error('The original launcher file is unavailable.');
  if (process.platform === 'win32') {
    const destination = path.join(desktop, 'LEVIRDetNet.lnk');
    if (fs.existsSync(destination)) {
      const { response } = await dialog.showMessageBox(window, { type: 'question', buttons: ['Cancel', 'Replace'], defaultId: 0, cancelId: 0, message: 'Replace the existing LEVIRDetNet desktop shortcut?' });
      if (response !== 1) return;
    }
    if (!shell.writeShortcutLink(destination, 'create', { target: launcherFile, cwd: path.dirname(launcherFile), description: 'LEVIRDetNet Launcher' })) throw new Error('Could not create the shortcut.');
  } else {
    const destination = path.join(desktop, 'LEVIRDetNet.desktop');
    if (/[\r\n]/.test(launcherFile)) throw new Error('Unsupported launcher filename.');
    if (fs.existsSync(destination)) {
      const { response } = await dialog.showMessageBox(window, { type: 'question', buttons: ['Cancel', 'Replace'], defaultId: 0, cancelId: 0, message: 'Replace the existing LEVIRDetNet desktop shortcut?' });
      if (response !== 1) return;
    }
    const quoted = launcherFile.replace(/([\\"`$])/g, '\\$1').replace(/%/g, '%%');
    await fsp.writeFile(destination, `[Desktop Entry]\nType=Application\nName=LEVIRDetNet\nComment=Remote sensing object detection\nExec=bash "${quoted}"\nTerminal=false\nIcon=applications-science\n`, { mode: 0o755 });
    await fsp.chmod(destination, 0o755);
  }
  controller.update({ message: controller.text('桌面快捷方式已创建。', 'Desktop shortcut created.') });
}
async function handleAction(name, payload) {
  switch (name) {
    case 'start': case 'resume':
      if (!controller.state.busy && !launchPromise) {
        launchPromise = controller.start().finally(() => { launchPromise = null; });
      }
      break;
    case 'pause': controller.pause(); break;
    case 'integrity':
      if (!controller.state.busy && !launchPromise) launchPromise = controller.integrity().finally(() => { launchPromise = null; });
      break;
    case 'choosePath': {
      if (controller.state.busy || controller.session) throw new Error(controller.text('请先暂停当前任务。', 'Pause the current operation first.'));
      const result = await dialog.showOpenDialog(window, { title: controller.text('选择资源和代码保存目录', 'Choose the resource and code directory'),
        defaultPath: controller.state.installRoot, properties: ['openDirectory', 'createDirectory'] });
      if (!result.canceled && result.filePaths[0]) await controller.setPath(await fsp.realpath(result.filePaths[0]));
      break;
    }
    case 'language': await controller.setLanguage(payload?.language); break;
    case 'shortcut': await shortcut(); break;
    case 'github': await shell.openExternal(controller.manifest.repository.url); break;
    case 'openDownload': {
      const allowed = new Set([controller.manifest.repository.url, ...controller.manifest.assets.flatMap(item => [item.url, ...(item.mirrors || []).map(mirror => mirror.url)]),
        'https://docs.docker.com/desktop/setup/install/windows-install/',
        'https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html']);
      if (!allowed.has(payload?.url)) throw new Error('Unknown download URL.');
      await shell.openExternal(payload.url); break;
    }
    case 'openFolder': {
      await fsp.mkdir(controller.state.installRoot, { recursive: true });
      const error = await shell.openPath(controller.state.installRoot); if (error) throw new Error(error); break;
    }
    default: throw new Error('Unknown launcher action.');
  }
  return controller.snapshot();
}
app.whenReady().then(async () => {
  const manifest = JSON.parse((await fsp.readFile(path.join(__dirname, '../assets.json'), 'utf8')).replace(/^\uFEFF/, ''));
  const candidate = path.resolve(__dirname, '../..');
  const existing = fs.existsSync(path.join(candidate, 'demo/image_demo.py')) ? candidate :
    fs.existsSync(path.join(launcherRoot, 'demo/image_demo.py')) ? launcherRoot : path.join(launcherRoot, 'levirdetnet-release');
  controller = new LauncherController({ appRoot: __dirname, launcherRoot, launcherFile,
    settingsFile: path.join(settingsRoot, 'launcher-settings.json'), defaultRoot: process.env.LEVIR_INSTALL_ROOT || existing,
    electronPath: process.execPath, locale: app.getLocale(), manifest, requestTransport: net.request.bind(net) });
  await controller.init();
  Menu.setApplicationMenu(null);
  // Electron sizes use logical pixels; a fixed 1380x800 overflowed 150%-scaled screens.
  const workArea = screen.getDisplayNearestPoint(screen.getCursorScreenPoint()).workArea;
  const ratio = 1639 / 907;
  const width = Math.floor(Math.min(1100, workArea.width * 0.9, workArea.height * 0.9 * ratio));
  const height = Math.round(width / ratio);
  window = new BrowserWindow({ title: 'LEVIRDetNet Launcher', width, height,
    x: workArea.x + Math.round((workArea.width - width) / 2), y: workArea.y + Math.round((workArea.height - height) / 2),
    resizable: false, maximizable: false, fullscreenable: false,
    frame: false, backgroundColor: '#141414', show: false, icon: path.join(__dirname, 'assets/logo.png'),
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false,
      sandbox: true, webSecurity: true, devTools: false } });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, target) => { if (target !== pageURL) event.preventDefault(); });
  window.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  controller.on('state', state => { if (!window.isDestroyed()) window.webContents.send('launcher:state', state); });
  ipcMain.handle('launcher:state', event => { trusted(event); return controller.snapshot(); });
  ipcMain.handle('launcher:action', async (event, name, payload) => { trusted(event); return handleAction(name, payload); });
  ipcMain.handle('launcher:minimize', event => { trusted(event); window.minimize(); });
  ipcMain.handle('launcher:close', event => { trusted(event); window.close(); });
  window.on('close', async event => {
    if (closing) return;
    event.preventDefault();
    if (controller.state.busy && !controller.state.canPause) {
      await dialog.showMessageBox(window, { type: 'info', message: controller.text('正在安装、校验或启动，请等待当前步骤完成后关闭。', 'An installation or startup step is in progress. Wait for it to finish before closing.') });
      return;
    }
    if (controller.state.busy || controller.session) {
      const result = await dialog.showMessageBox(window, { type: 'question', buttons: [controller.text('取消', 'Cancel'), controller.text('关闭', 'Close')], defaultId: 0, cancelId: 0,
        message: controller.text('关闭会暂停下载并结束由此启动器打开的推理程序。是否继续？', 'Closing pauses downloads and stops the inference session opened by this launcher. Continue?') });
      if (result.response !== 1) return;
    }
    controller.pause();
    await launchPromise?.catch(() => {});
    await controller.stopSession();
    closing = true; window.close();
  });
  window.once('ready-to-show', () => window.show());
  await window.loadFile(path.join(__dirname, 'index.html'));
}).catch(error => { dialog.showErrorBox('LEVIRDetNet Launcher', error.message); app.quit(); });
app.on('window-all-closed', () => app.quit());
