'use strict';

(() => {
  const api = window.launcher;
  const $ = id => document.getElementById(id);
  const translations = {
    en: {
      settings: 'Launcher settings', minimize: 'Minimize', close: 'Close',
      caption: 'A closer look at our world.', smallMoment: 'A SMALL MOMENT',
      visitGithub: 'Meet us on GitHub', shortcut: 'Add desktop shortcut',
      integrity: 'Check the resource integrity', changePath: 'Change the download path',
      changeLanguage: 'Change language', setupLabel: 'YOUR WORKSPACE', location: 'Download location',
      systemPrompt: 'System prompt', requiredShort: 'Required space:', changePathShort: 'Change path', ok: 'OK', resourceDetails: 'Show folder and resource details',
      browse: 'Browse', pathNote: 'Resources and inference results are stored in this folder. Existing files stay in their current location when you change it.',
      required: 'Estimated additional space', free: 'Available disk space',
      dockerSpace: 'Docker also needs space for its imported image in its own storage location.',
      lowSpace: 'This drive may not have enough free space. Choose a larger drive before starting. Docker also needs space for its imported image.',
      resources: 'Resources', language: 'Language', done: 'Done', letsFix: 'LET’S GET YOU STARTED',
      manualHelp: 'Download help', showHelp: 'View download instructions', downloadFrom: 'Download from',
      saveTo: 'Save the file at', retryNote: 'After placing the file here or resolving the issue, close this window and click Retry. Your completed downloads will be reused.',
      openFolder: 'Open download folder', openDownload: 'Open download page', download: 'Download', pause: 'Pause', resume: 'Resume',
      launch: 'Launch', running: 'Running', retry: 'Retry', preparing: 'Preparing', working: 'Working',
      idleHint: 'Set up once. Explore any time.', readyHint: 'Everything is ready. Open the inference demo.',
      runningHint: 'Your inference demo is open.', pausedHint: 'Your download is saved. Resume when you’re ready.',
      errorHint: 'Setup needs your attention.', busyHint: 'Completed downloads are kept for next time.',
      idle: 'Ready to set up', checking: 'Checking your computer', prerequisites: 'Preparing prerequisites',
      downloading: 'Downloading', verifying: 'Checking resource integrity', extracting: 'Extracting project files',
      loading: 'Importing Docker environment', paused: 'Download paused', ready: 'Ready to launch', launching: 'Starting the demo',
      error: 'Could not finish setup', unknown: 'Not yet known', eta: 'ETA', remaining: 'remaining',
      missing: 'Not downloaded', pending: 'Pending', present: 'Found', complete: 'Complete', verified: 'Verified',
      installed: 'Installed', invalid: 'Needs repair', partial: 'Partial download', skipped: 'Reused', available: 'Available',
      changed: 'Updated', shortcutDone: 'Desktop shortcut created.', integrityDone: 'Resource check complete.',
      morning: 'Good morning.', afternoon: 'Good afternoon.', evening: 'Good evening.', night: 'Hello, night explorer.',
      wishMorning: 'May a fresh perspective bring something wonderful into view today.',
      wishAfternoon: 'Take a breath. There’s always something new to discover in the details.',
      wishEvening: 'May your work find a good ending, and your evening a little peace.',
      wishNight: 'The world can wait a little. Remember to rest as you explore.',
      bridgeError: 'The desktop bridge is unavailable. Please open this app using launch_fast_demo.cmd or launch_fast_demo.sh.',
      downloadSettings: 'Download settings', resourceSingular: 'resource', resourcePlural: 'resources',
      folderUnknown: 'Choose a download location', closeDialog: 'Close dialog',
      docker: 'Docker environment', weights30: '30-class checkpoint', weights159: '159-class checkpoint', code: 'Project source code', project: 'Project source code'
    },
    zh: {
      settings: '启动器设置', minimize: '最小化', close: '关闭', caption: '从细节，看见更大的世界。',
      smallMoment: '此刻，与你相遇', visitGithub: '在 GitHub 与我们见面', shortcut: '添加桌面快捷方式',
      integrity: '检查资源完整性', changePath: '更改下载路径', changeLanguage: '更改语言',
      systemPrompt: '系统提示', requiredShort: '所需空间：', changePathShort: '更改路径', ok: '确定', resourceDetails: '查看路径与资源详情',
      setupLabel: '你的工作空间', location: '下载路径', browse: '更改',
      pathNote: '资源与推理结果保存在此文件夹。更改路径后，已下载的文件仍保留在原位置。',
      required: '预计新增占用空间', free: '磁盘可用空间', dockerSpace: 'Docker 导入镜像时，还需要在它自己的存储位置占用空间。',
      lowSpace: '此磁盘的可用空间可能不足，请先选择空间更大的磁盘。Docker 导入镜像还需额外空间。',
      resources: '资源清单', language: '语言', done: '确定', letsFix: '一起完成最后一步', manualHelp: '下载与安装帮助',
      showHelp: '查看下载与处理说明', downloadFrom: '请从此处下载', saveTo: '将文件放到此路径',
      retryNote: '放好文件或解决问题后，关闭此窗口并点击“重试”。已完成的下载会继续复用。',
      openFolder: '打开下载文件夹', openDownload: '打开下载页面', download: '启动下载', pause: '暂停', resume: '继续下载',
      launch: '启动演示', running: '运行中', retry: '重试', preparing: '准备中', working: '处理中',
      idleHint: '完成一次配置，随时开始探索。', readyHint: '资源已就绪，可以打开推理演示。',
      runningHint: '推理演示窗口已打开。', pausedHint: '下载进度已保留，随时可以继续。',
      errorHint: '有一项设置需要你处理。', busyHint: '已经下载的文件会保留，下次可继续使用。',
      idle: '准备下载', checking: '检查运行环境', prerequisites: '准备运行依赖', downloading: '正在下载',
      verifying: '校验资源完整性', extracting: '解压项目代码', loading: '导入 Docker 环境', paused: '下载已暂停',
      ready: '资源已就绪', launching: '启动演示中', error: '暂时无法完成配置', unknown: '暂时未知', eta: '预计剩余', remaining: '剩余',
      missing: '尚未下载', pending: '等待中', present: '已找到', complete: '已完成', verified: '校验通过',
      installed: '已安装', invalid: '需要修复', partial: '部分下载', skipped: '已复用', available: '可用',
      changed: '已更新', shortcutDone: '桌面快捷方式已创建。', integrityDone: '资源检查完成。',
      morning: '早上好。', afternoon: '下午好。', evening: '晚上好。', night: '你好，夜行的探索者。',
      wishMorning: '愿今天的新视角，为你带来一点惊喜。', wishAfternoon: '稍作休息，世界的细节里，总有新的发现。',
      wishEvening: '愿今天的努力有所收获，愿此刻的你轻松而平静。', wishNight: '探索的脚步可以慢一点，也记得好好休息。',
      bridgeError: '桌面接口暂不可用。请通过 launch_fast_demo.cmd 或 launch_fast_demo.sh 打开程序。',
      downloadSettings: '下载设置', resourceSingular: '项资源', resourcePlural: '项资源',
      folderUnknown: '请选择下载路径', closeDialog: '关闭对话框',
      docker: 'Docker 环境镜像', weights30: '30 类模型权重', weights159: '159 类模型权重', code: '完整项目代码', project: '完整项目代码'
    }
  };
  let state = {phase: 'idle', busy: false, canPause: false, language: navigator.language.startsWith('zh') ? 'zh' : 'en', assets: [], progress: {}, ready: false};
  const pendingActions = new Set();
  let previousError = '';
  let toastTimer;
  let languageRendered;
  function scaleCanvas() {
    const scale = Math.min(window.innerWidth / 1639, window.innerHeight / 907);
    document.documentElement.style.setProperty('--ui-scale', String(scale));
    document.documentElement.style.setProperty('--canvas-left', `${(window.innerWidth - 1639 * scale) / 2}px`);
    document.documentElement.style.setProperty('--canvas-top', `${(window.innerHeight - 907 * scale) / 2}px`);
  }
  scaleCanvas();
  window.addEventListener('resize', scaleCanvas);
  const t = key => translations[state.language === 'zh' ? 'zh' : 'en'][key] || key;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const bytes = value => {
    if (!finite(value) || value < 0) return '—';
    if (value === 0) return '0 B';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    const power = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
    return `${(value / 1024 ** power).toFixed(power === 0 ? 0 : (value / 1024 ** power >= 100 ? 0 : 1))} ${units[power]}`;
  };
  const duration = value => {
    if (!finite(value) || value < 0) return '—';
    const seconds = Math.ceil(value);
    if (state.language === 'zh') {
      if (seconds < 60) return `${seconds} 秒`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
      return `${Math.floor(seconds / 3600)} 小时 ${Math.ceil((seconds % 3600) / 60)} 分`;
    }
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.ceil((seconds % 3600) / 60)}m`;
  };
  function notify(text) {
    if (!text) return;
    $('toast').textContent = String(text);
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6000);
  }
  function updateGreeting() {
    const now = new Date();
    $('greetingTime').dateTime = now.toISOString();
    $('greetingTime').textContent = now.toLocaleTimeString(state.language === 'zh' ? 'zh-CN' : 'en-GB', {hour: '2-digit', minute: '2-digit', hour12: false});
    const hour = now.getHours();
    const period = hour >= 5 && hour < 12 ? 'Morning' : hour < 18 && hour >= 12 ? 'Afternoon' : hour >= 18 && hour < 23 ? 'Evening' : 'Night';
    $('greetingTitle').textContent = t(period.toLowerCase());
    $('greetingWish').textContent = t(`wish${period}`);
  }
  function translate() {
    document.documentElement.lang = state.language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-text]').forEach(el => { el.textContent = t(el.dataset.text); });
    document.querySelectorAll('[data-title]').forEach(el => { el.title = t(el.dataset.title); el.setAttribute('aria-label', t(el.dataset.title)); });
    $('quickSettings').setAttribute('aria-label', t('downloadSettings'));
    $('quickMenu').setAttribute('aria-label', t('downloadSettings'));
    ['closeSettings', 'closeError', 'dismissGreeting'].forEach(id => $(id).setAttribute('aria-label', t('closeDialog')));
    $('languageSelect').value = state.language === 'zh' ? 'zh' : 'en';
    $('resourceDetailsButton').setAttribute('aria-label', t('resourceDetails'));
    updateGreeting();
  }
  function menu(open) {
    $('quickMenu').hidden = !open;
    $('quickSettings').setAttribute('aria-expanded', String(open));
    if (open) $('shortcutButton').focus();
  }
  function showSettings() {
    menu(false);
    if (!$('settingsDialog').open) {
      $('settingsDetails').hidden = true;
      $('settingsDialog').classList.remove('expanded');
      $('resourceDetailsButton').setAttribute('aria-expanded', 'false');
      $('settingsDialog').showModal();
    }
  }
  function showError() {
    const error = state.error;
    if (!error) return;
    $('errorMessage').textContent = error.message || state.message || t('error');
    $('errorUrl').textContent = error.url || '';
    $('errorPath').textContent = error.path || '';
    $('errorLinkSection').hidden = !error.url;
    $('errorPathSection').hidden = !error.path;
    $('openErrorLink').hidden = !error.url;
    if (!$('errorDialog').open) $('errorDialog').showModal();
  }
  function resourceRow(asset) {
    const li = document.createElement('li');
    const name = document.createElement('span'); name.className = 'resource-name'; name.textContent = ['docker', 'weights30', 'weights159', 'code', 'project'].includes(asset.id) ? t(asset.id) : asset.name || asset.id || '';
    const size = document.createElement('span'); size.className = 'resource-size'; size.textContent = bytes(asset.size);
    const path = document.createElement('span'); path.className = 'resource-path selectable'; path.textContent = asset.path || '';
    const status = document.createElement('span'); status.className = 'resource-status'; status.dataset.status = asset.status || 'pending'; status.textContent = t(asset.status || 'pending');
    li.append(name, size, path, status);
    return li;
  }
  function render() {
    if (languageRendered !== state.language) { translate(); languageRendered = state.language; }
    const phase = state.phase || 'idle';
    const progress = state.progress || {};
    const actualTotal = finite(progress.total) && progress.total > 0 ? progress.total : null;
    const actualReceived = finite(progress.received) && progress.received >= 0 ? progress.received : null;
    let percent = finite(progress.percent) ? progress.percent : actualTotal && actualReceived !== null ? actualReceived / actualTotal * 100 : null;
    if (percent !== null) percent = Math.min(100, Math.max(0, percent));
    const showProgress = !['idle', 'ready', 'running'].includes(phase);
    $('progressCard').hidden = !showProgress;
    $('progressCard').classList.toggle('is-error', phase === 'error');
    $('progressCard').classList.toggle('is-paused', phase === 'paused');
    $('progressItem').textContent = progress.item || t(phase);
    $('phaseMessage').textContent = state.message || '';
    $('phaseMessage').hidden = !state.message;
    $('progressPercent').textContent = percent === null ? '—' : `${Math.floor(percent)}%`;
    $('progressBytes').textContent = `${bytes(actualReceived)} / ${bytes(actualTotal)}`;
    $('progressFill').style.width = `${percent || 0}%`;
    const indeterminate = percent === null && !['paused', 'error'].includes(phase);
    $('progressTrack').classList.toggle('indeterminate', indeterminate);
    $('progressTrack').setAttribute('aria-label', progress.item || t(phase));
    if (percent === null) $('progressTrack').removeAttribute('aria-valuenow');
    else $('progressTrack').setAttribute('aria-valuenow', String(Math.floor(percent)));
    const transferring = phase === 'downloading' || (phase === 'prerequisites' && (progress.phase === 'download' || actualTotal !== null));
    $('progressSpeed').textContent = finite(progress.speed) && progress.speed >= 0 && transferring ? `${bytes(progress.speed)}/s` : '—';
    $('progressEta').textContent = `${t('eta')} ${transferring ? duration(progress.eta) : '—'}`;
    $('errorDetails').hidden = phase !== 'error' || !state.error;
    let action = 'download', icon = 'download', hint = 'idleHint';
    let disabled = false;
    if (phase === 'paused') { action = 'resume'; icon = 'play'; hint = 'pausedHint'; }
    else if (phase === 'error') { action = 'retry'; hint = 'errorHint'; }
    else if (phase === 'running') { action = 'running'; icon = 'check'; hint = 'runningHint'; disabled = true; }
    else if (state.busy) { action = state.canPause ? 'pause' : phase === 'launching' ? 'preparing' : 'working'; icon = state.canPause ? 'pause' : 'download'; hint = 'busyHint'; disabled = !state.canPause; }
    else if (state.ready || phase === 'ready') { action = 'launch'; icon = 'play'; hint = 'readyHint'; }
    if (pendingActions.size && !state.canPause) disabled = true;
    $('primaryAction').disabled = disabled;
    $('primaryLabel').textContent = t(action === 'pause' ? 'download' : action);
    $('primaryAction').setAttribute('aria-label', t(action));
    $('primaryAction').title = t(action);
    $('primaryIcon').setAttribute('href', `#i-${icon}`);
    $('actionHint').textContent = t(hint);
    $('choosePathButton').disabled = !!state.busy || !!pendingActions.size;
    $('pathButton').disabled = !!state.busy || !!pendingActions.size;
    $('integrityButton').disabled = !!state.busy || !!pendingActions.size;
    $('installPath').textContent = state.installRoot || t('folderUnknown');
    $('resourceDetailsButton').title = `${state.installRoot || t('folderUnknown')} — ${t('resourceDetails')}`;
    $('requiredSpace').textContent = bytes(state.requiredBytes);
    $('freeSpace').textContent = bytes(state.freeBytes);
    const lowSpace = finite(state.freeBytes) && finite(state.requiredBytes) && state.freeBytes < state.requiredBytes;
    $('spaceNote').textContent = t(lowSpace ? 'lowSpace' : 'dockerSpace');
    $('spaceNote').classList.toggle('warning', lowSpace);
    const assets = Array.isArray(state.assets) ? state.assets : [];
    const knownTotal = assets.reduce((sum, asset) => sum + (finite(asset.size) ? asset.size : 0), 0);
    const knownReceived = assets.reduce((sum, asset) => sum + (['verified', 'complete', 'installed'].includes(asset.status) && finite(asset.size) ? asset.size : finite(asset.downloaded) ? asset.downloaded : 0), 0);
    const settingsTotal = actualTotal || knownTotal;
    const settingsReceived = actualTotal ? actualReceived : knownReceived;
    $('settingsReceived').textContent = bytes(settingsReceived);
    $('settingsTotal').textContent = bytes(settingsTotal || null);
    $('settingsProgressFill').style.width = `${settingsTotal && finite(settingsReceived) ? Math.min(100, Math.max(0, settingsReceived / settingsTotal * 100)) : 0}%`;
    $('resourceCount').textContent = `${assets.length} ${t(assets.length === 1 ? 'resourceSingular' : 'resourcePlural')}`;
    $('resourceList').replaceChildren(...assets.map(resourceRow));
    const errorSignature = state.error ? `${state.error.message}|${state.error.url}|${state.error.path}` : '';
    if (errorSignature && errorSignature !== previousError && phase === 'error') { previousError = errorSignature; showError(); }
    if (!state.error) previousError = '';
  }
  function accept(next) {
    if (!next || typeof next !== 'object') return;
    state = {...state, ...next};
    if (!['zh', 'en'].includes(state.language)) state.language = navigator.language.startsWith('zh') ? 'zh' : 'en';
    render();
  }
  async function action(name, payload) {
    if (!api) { notify(t('bridgeError')); return; }
    menu(false);
    const pending = Symbol(name);
    pendingActions.add(pending);
    render();
    try {
      const result = await api.action(name, payload);
      if (result && result.phase) accept(result);
      else if (result && result.state) accept(result.state);
      if (result && result.message) notify(result.message);
      else if (name === 'shortcut' && result !== false && !(result && result.cancelled)) notify(t('shortcutDone'));
      if (!result || (!result.phase && !result.state)) accept(await api.getState());
    } catch (error) {
      notify(error && error.message ? error.message : String(error));
    } finally {
      pendingActions.delete(pending);
      render();
    }
  }
  $('primaryAction').addEventListener('click', () => {
    if (state.phase === 'paused') void action('resume');
    else if (state.busy && state.canPause) void action('pause');
    else void action('start');
  });
  $('settingsButton').addEventListener('click', showSettings);
  $('quickSettings').addEventListener('click', () => menu($('quickMenu').hidden));
  $('closeSettings').addEventListener('click', () => $('settingsDialog').close());
  $('settingsDone').addEventListener('click', () => $('settingsDialog').close());
  $('resourceDetailsButton').addEventListener('click', () => {
    const expanded = $('settingsDetails').hidden;
    $('settingsDetails').hidden = !expanded;
    $('settingsDialog').classList.toggle('expanded', expanded);
    $('resourceDetailsButton').setAttribute('aria-expanded', String(expanded));
  });
  $('closeError').addEventListener('click', () => $('errorDialog').close());
  $('choosePathButton').addEventListener('click', () => void action('choosePath'));
  $('pathButton').addEventListener('click', () => { showSettings(); void action('choosePath'); });
  $('shortcutButton').addEventListener('click', () => void action('shortcut'));
  $('integrityButton').addEventListener('click', () => void action('integrity'));
  $('languageButton').addEventListener('click', () => void action('language', {language: state.language === 'zh' ? 'en' : 'zh'}));
  $('languageSelect').addEventListener('change', event => void action('language', {language: event.target.value}));
  ['railHome', 'greetingGithub'].forEach(id => $(id).addEventListener('click', () => void action('github')));
  $('dismissGreeting').addEventListener('click', () => { $('greeting').hidden = true; });
  $('errorDetails').addEventListener('click', showError);
  $('openErrorLink').addEventListener('click', () => void action('openDownload', {url: state.error && state.error.url}));
  $('openErrorFolder').addEventListener('click', () => void action('openFolder'));
  $('minimizeButton').addEventListener('click', () => api && api.minimize());
  $('closeButton').addEventListener('click', () => api && api.close());
  document.addEventListener('pointerdown', event => { if (!event.target.closest('.quick-menu, #quickSettings')) menu(false); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !$('quickMenu').hidden) { menu(false); $('quickSettings').focus(); }
    if (!$('quickMenu').hidden && ['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      const items = [...$('quickMenu').querySelectorAll('button:not(:disabled)')];
      const current = items.indexOf(document.activeElement);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      if (items[index]) items[index].focus();
      event.preventDefault();
    }
  });
  render();
  if (api) {
    const unsubscribe = api.onState(accept);
    window.addEventListener('unload', () => { if (typeof unsubscribe === 'function') unsubscribe(); });
    api.getState().then(accept).catch(error => notify(error.message || String(error)));
    const random = new Uint32Array(1); crypto.getRandomValues(random);
    if (random[0] / 4294967296 < 0.2) $('greeting').hidden = false;
    setInterval(updateGreeting, 30000);
  } else {
    $('primaryAction').disabled = true;
    $('actionHint').textContent = t('bridgeError');
  }
})();
