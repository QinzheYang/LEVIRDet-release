'use strict';
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('levirDesktop', Object.freeze({
  chooseFiles: () => ipcRenderer.invoke('levir:choose-files'),
  chooseFolder: () => ipcRenderer.invoke('levir:choose-folder'),
  preview: path => ipcRenderer.invoke('levir:preview', path),
  uploadFile: (path, url, token) => ipcRenderer.invoke('levir:upload', path, url, token),
  openOutput: path => ipcRenderer.invoke('levir:open-output', path),
  copyText: text => ipcRenderer.invoke('levir:copy-text', text),
  saveJSON: (name, text) => ipcRenderer.invoke('levir:save-json', name, text)
}));
