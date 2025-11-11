const { app, BrowserWindow, ipcMain, clipboard, nativeImage } = require('electron');
const path = require('path');
const { spawn, execSync } = require('child_process');

let mainWindow;
let backendProcess;

// Kill backend process properly on Windows
function killBackend() {
  if (backendProcess) {
    console.log('Killing backend process...');
    try {
      if (process.platform === 'win32') {
        // On Windows, use taskkill to kill the process tree
        execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' });
      } else {
        backendProcess.kill('SIGTERM');
      }
      backendProcess = null;
      console.log('Backend killed successfully');
    } catch (error) {
      console.error('Error killing backend:', error.message);
    }
  }
}

// Kill any existing Python process on port 5000
function killExistingBackend() {
  console.log('Checking for existing processes on port 5000...');
  try {
    if (process.platform === 'win32') {
      // Kill all python.exe processes as a simple approach
      execSync('taskkill /F /IM python.exe /T', { stdio: 'ignore' });
      console.log('Cleared any existing Python processes');
      
      // Wait a bit for port to be released
      setTimeout(() => {
        console.log('Port should be free now');
      }, 1500);
    }
  } catch (error) {
    // Ignore errors - might mean no processes to kill
    console.log('No existing Python processes to kill');
  }
}

// Start backend server
function startBackend() {
  console.log('=== BACKEND STARTUP INITIATED ===');
  console.log('isPackaged:', app.isPackaged);
  console.log('resourcesPath:', process.resourcesPath);
  
  // No installation needed - backend is a standalone executable
  actuallyStartBackend();
}

function actuallyStartBackend() {
  console.log('=== ACTUALLY STARTING BACKEND ===');
  
  // Check if backend is already running (started by run.bat)
  const http = require('http');
  
  function checkBackend(attempt = 0) {
    console.log(`Checking for existing backend (attempt ${attempt + 1}/5)...`);
    
    const options = {
      host: '127.0.0.1',
      port: 5000,
      path: '/health',
      timeout: 1000
    };
    
    const req = http.get(options, (res) => {
      console.log('✓ Backend already running externally - ready to use');
      backendProcess = null; // Don't manage the external process
    });
    
    req.on('error', (err) => {
      console.log(`Backend check failed: ${err.message}`);
      
      if (attempt < 5) {
        // Backend might still be starting, wait and retry
        console.log(`Waiting for backend... (attempt ${attempt + 1}/5)`);
        setTimeout(() => checkBackend(attempt + 1), 1000);
      } else {
        // After 5 attempts, start our own backend
        console.log('Backend not detected, starting our own...');
        
        // Determine backend path
        const fs = require('fs');
        if (app.isPackaged) {
          // Check for bundled backend.exe (portable build) OR Python source (light installer)
          const backendExe = path.join(process.resourcesPath, 'backend', 'backend.exe');
          const backendScript = path.join(process.resourcesPath, 'backend', 'app.py');
          
          if (fs.existsSync(backendExe)) {
            // Portable build - use bundled executable
            console.log('Using bundled backend executable:', backendExe);
            backendProcess = spawn(backendExe, [], {
              cwd: path.dirname(backendExe),
              stdio: ['ignore', 'pipe', 'pipe'],
              detached: false
            });
          } else if (fs.existsSync(backendScript)) {
            // Light installer - run Python source
            console.log('Using Python source (light installer):', backendScript);
            const pythonPath = 'python';
            backendProcess = spawn(pythonPath, [backendScript], {
              cwd: path.dirname(backendScript),
              stdio: ['ignore', 'pipe', 'pipe'],
              detached: false
            });
          } else {
            console.error('ERROR: Backend not found at:', backendExe, 'or', backendScript);
            const { dialog } = require('electron');
            dialog.showErrorBox(
              'Backend Missing',
              'Backend files not found.\n\nPlease reinstall the application.'
            );
            return;
          }
        } else {
          // Development: Use Python to run app.py
          const pythonPath = 'python';
          const backendScript = path.join(__dirname, '..', 'local-ai-backend', 'app.py');
          console.log('Using system Python (development mode)');
          console.log('Starting backend from:', backendScript);
          
          backendProcess = spawn(pythonPath, [backendScript], {
            cwd: path.dirname(backendScript),
            stdio: ['ignore', 'pipe', 'pipe'],
            windowsHide: false
          });
        }
        
        // Set up backend process logging
        backendProcess.stdout.on('data', (data) => {
          console.log(`Backend: ${data.toString().trim()}`);
        });
        
        backendProcess.stderr.on('data', (data) => {
          console.error(`Backend Error: ${data.toString().trim()}`);
        });
        
        backendProcess.on('error', (error) => {
          console.error(`Backend spawn error: ${error.message}`);
        });
        
        backendProcess.on('close', (code) => {
          console.log(`Backend process exited with code ${code}`);
          if (code !== 0 && code !== null) {
            console.error('Backend crashed - check logs for details');
          }
        });
      }
    });
    
    req.setTimeout(1000);
    req.end();
  }
  
  checkBackend();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false // Allow camera access
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    title: 'Whiteboard Scanner',
    backgroundColor: '#ffffff'
  });

  // Load the React app
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // DevTools disabled for production
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    // Only kill backend if we started it (not if run.bat started it)
    if (backendProcess) {
      killBackend();
    }
  });
}

// IPC Handlers
ipcMain.handle('copy-to-clipboard', async (event, imageData) => {
  try {
    const image = nativeImage.createFromDataURL(imageData);
    clipboard.writeImage(image);
    return { success: true, message: 'Image copied to clipboard! You can paste it into OneNote.' };
  } catch (error) {
    return { success: false, message: `Failed to copy: ${error.message}` };
  }
});

ipcMain.handle('get-backend-url', async () => {
  return 'http://127.0.0.1:5000';
});

// App lifecycle
app.on('ready', () => {
  // Start backend in background
  startBackend();
  
  // Create window immediately - frontend will show loading overlay
  createWindow();
});

app.on('window-all-closed', () => {
  // Only kill backend if we started it
  if (backendProcess) {
    killBackend();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', () => {
  // Only kill backend if we started it
  if (backendProcess) {
    killBackend();
  }
});

app.on('will-quit', () => {
  // Only kill backend if we started it
  if (backendProcess) {
    killBackend();
  }
});
