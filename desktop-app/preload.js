const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  copyToClipboard: (imageData) => ipcRenderer.invoke('copy-to-clipboard', imageData),
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
  platform: process.platform
});
