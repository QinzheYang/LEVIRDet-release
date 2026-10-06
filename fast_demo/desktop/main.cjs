'use strict';
const { app, BrowserWindow, dialog, ipcMain, nativeImage, shell, Menu, clipboard } = require('electron');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');

const suffixes = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm']);
const selected = new Map();
let win;
let demoURL;
let token;
const outputRoot = process.env.LEVIR_DEMO_OUTPUT || process.env.LEVIR_DEMO_HOST_OUTPUT;

function trusted(event) {
  if (!win || event.sender !== win.webContents || !event.senderFrame ||
      new URL(event.senderFrame.url).origin !== demoURL.origin) {
    throw new Error('Untrusted desktop request.');
  }
}

async function collect(paths, folder = null) {
  const rows = [];
  async function visit(filename) {
    const stat = await fsp.lstat(filename);
    if (stat.isSymbolicLink()) return;
    if (stat.isDirectory()) {
      for (const name of (await fsp.readdir(filename)).sort()) await visit(path.join(filename, name));
    } else if (stat.isFile() && suffixes.has(path.extname(filename).toLowerCase())) {
      if (rows.length >= 10000) throw new Error('Please select at most 10,000 images for one demo run.');
      if (stat.size <= 0 || stat.size > 512 * 1024 * 1024) throw new Error(`${path.basename(filename)} must be nonempty and at most 512 MiB.`);
      const absolute = await fsp.realpath(filename);
      selected.set(absolute, { size: stat.size, mtimeMs: stat.mtimeMs });
      rows.push({ path: absolute, name: path.basename(filename),
        relative_path: folder ? path.relative(folder, filename).split(path.sep).join('/') : path.basename(filename), size: stat.size });
    }
  }
  for (const filename of paths) await visit(filename);
  rows.sort((a, b) => a.relative_path.localeCompare(b.relative_path, undefined, { numeric: true }));
  return rows;
}

async function checkedSelection(filename) {
  if (typeof filename !== 'string' || !selected.has(filename)) throw new Error('Select this image again using the file picker.');
  const stat = await fsp.stat(filename);
  const previous = selected.get(filename);
  if (stat.size !== previous.size || stat.mtimeMs !== previous.mtimeMs || !stat.isFile()) {
    throw new Error('A selected image changed on disk. Please select the images again.');
  }
  return stat;
}

function registerIPC() {
  ipcMain.handle('levir:choose-files', async event => {
    trusted(event);
    const result = await dialog.showOpenDialog(win, {
      title: 'Choose input images', properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Images', extensions: [...suffixes].map(s => s.slice(1)) }]
    });
    return result.canceled ? [] : collect(result.filePaths);
  });
  ipcMain.handle('levir:choose-folder', async event => {
    trusted(event);
    const result = await dialog.showOpenDialog(win, { title: 'Choose input folder', properties: ['openDirectory'] });
    return result.canceled ? [] : collect(result.filePaths, result.filePaths[0]);
  });
  ipcMain.handle('levir:preview', async (event, filename) => {
    trusted(event);
    await checkedSelection(filename);
    try {
      const image = nativeImage.createFromPath(filename);
      if (image.isEmpty()) return null;
      const size = image.getSize();
      const scale = Math.min(1, 1600 / Math.max(size.width, size.height));
      return image.resize({ width: Math.max(1, Math.round(size.width * scale)),
        height: Math.max(1, Math.round(size.height * scale)) }).toDataURL();
    } catch { return null; }
  });
  ipcMain.handle('levir:upload', async (event, filename, address, providedToken) => {
    trusted(event);
    const stat = await checkedSelection(filename);
    const destination = new URL(address, demoURL.origin);
    if (destination.origin !== demoURL.origin ||
        !/^\/api\/jobs\/[a-f0-9]{32}\/files\/[0-9]{6}$/.test(destination.pathname) ||
        destination.search || destination.hash || providedToken !== token) {
      throw new Error('Invalid upload destination.');
    }
    return new Promise((resolve, reject) => {
      const request = http.request(destination, { method: 'PUT', headers: {
        'X-Levir-Token': token, 'Content-Type': 'application/octet-stream', 'Content-Length': stat.size
      } }, response => {
        let body = '';
        response.setEncoding('utf8');
        response.on('data', chunk => { body += chunk; if (body.length > 1024 * 1024) request.destroy(new Error('Unexpected upload response.')); });
        response.on('end', () => {
          try {
            const data = JSON.parse(body);
            if (response.statusCode < 200 || response.statusCode >= 300) reject(new Error(data.error || 'Upload failed.'));
            else resolve(data);
          } catch (error) { reject(error); }
        });
        response.on('error', reject);
      });
      request.setTimeout(120000, () => request.destroy(new Error('Upload timed out. Try again.')));
      request.on('error', reject);
      const source = fs.createReadStream(filename);
      source.on('error', error => request.destroy(error));
      request.on('close', () => source.destroy());
      source.pipe(request);
    });
  });
  ipcMain.handle('levir:save-json', async (event, name, text) => {
    trusted(event);
    if (typeof name !== 'string' || typeof text !== 'string' || text.length > 10 * 1024 * 1024) throw new Error('Invalid prediction JSON.');
    JSON.parse(text);
    const filename = path.basename(name).replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/\.json$/i, '') + '.json';
    const result = await dialog.showSaveDialog(win, {
      title: 'Save prediction JSON', defaultPath: path.join(outputRoot || app.getPath('documents'), filename),
      filters: [{ name: 'JSON predictions', extensions: ['json'] }]
    });
    if (result.canceled || !result.filePath) return false;
    await fsp.writeFile(result.filePath, text, 'utf8');
    return true;
  });
  ipcMain.handle('levir:copy-text', (event, text) => {
    trusted(event);
    if (typeof text !== 'string' || text.length > 8192) throw new Error('Invalid clipboard text.');
    clipboard.writeText(text);
    return true;
  });
  ipcMain.handle('levir:open-output', async (event, directory) => {
    trusted(event);
    if (!outputRoot || typeof directory !== 'string') throw new Error('The output directory is unavailable.');
    const root = await fsp.realpath(outputRoot);
    const target = await fsp.realpath(directory);
    const relative = path.relative(root, target);
    if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Only demo output folders can be opened.');
    const error = await shell.openPath(target);
    if (error) throw new Error(error);
    return true;
  });
}

app.setName('LEVIRDetNet');
// Keep preferences separate from other Electron apps; never use a shared root profile.
if (outputRoot) app.setPath('userData', path.join(outputRoot, '.desktop-profile'));
app.whenReady().then(() => {
  try {
    demoURL = new URL(process.env.LEVIR_DEMO_URL || '');
    token = new URLSearchParams(demoURL.hash.slice(1)).get('token');
    if (demoURL.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(demoURL.hostname) || !token || token.length < 24) {
      throw new Error('Use launch_fast_demo.cmd or launch_fast_demo.sh to start LEVIRDetNet.');
    }
  } catch (error) {
    dialog.showErrorBox('LEVIRDetNet', error.message);
    app.quit();
    return;
  }
  Menu.setApplicationMenu(null);
  win = new BrowserWindow({ title: 'LEVIRDetNet', width: 1440, height: 940,
    minWidth: 960, minHeight: 640, backgroundColor: '#f3f4f2', show: false,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true,
      nodeIntegration: false, sandbox: true, webSecurity: true, devTools: false }
  });
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', (event, address) => {
    if (new URL(address).origin !== demoURL.origin) event.preventDefault();
  });
  win.webContents.session.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  win.webContents.on('page-title-updated', event => event.preventDefault());
  win.webContents.on('did-fail-load', (_event, code, description) => {
    if (code !== -3) dialog.showErrorBox('LEVIRDetNet', `The local inference service could not be reached. ${description}`);
  });
  registerIPC();
  win.once('ready-to-show', () => win.show());
  win.loadURL(demoURL.href);
});
app.on('window-all-closed', () => app.quit());

// Export pure selection helpers for the repository's tests; no external modules needed.
module.exports = { collect, checkedSelection };
