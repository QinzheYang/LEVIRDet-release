'use strict';
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('launcher', Object.freeze({
  getState: () => ipcRenderer.invoke('launcher:state'),
  action: (name, payload) => ipcRenderer.invoke('launcher:action', name, payload),
  minimize: () => ipcRenderer.invoke('launcher:minimize'),
  close: () => ipcRenderer.invoke('launcher:close'),
  onState: callback => {
    const listener = (_event, state) => callback(state);
    ipcRenderer.on('launcher:state', listener);
    return () => ipcRenderer.removeListener('launcher:state', listener);
  }
}));
