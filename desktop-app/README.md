# OneNote Whiteboard Scanner - Desktop App# OneNote Whiteboard Scanner - Desktop App# Whiteboard Scanner - Desktop App



A standalone Windows desktop application for scanning and digitizing whiteboards with AI processing and direct OneNote integration.



## FeaturesA standalone Windows desktop application for scanning and digitizing whiteboards with AI processing and direct OneNote integration.A standalone Windows desktop application for scanning and digitizing whiteboards with AI processing.



- ✅ **Full Camera Access** - No browser restrictions

- ✅ **Local AI Processing** - Fast, private, offline-capable

- ✅ **OneNote Integration** - Browse and upload directly to OneNote## Features## Features

- ✅ **Clipboard Support** - Copy results to paste anywhere

- ✅ **Native Windows App** - Professional desktop experience



## Development- ✅ **Full Camera Access** - No browser restrictions- ✅ **Full Camera Access** - No browser restrictions



Start the app in development mode:- ✅ **Local AI Processing** - Fast, private, offline-capable- ✅ **Local AI Processing** - Fast, private, offline-capable

```cmd

npm start- ✅ **OneNote Integration** - Browse and upload directly to OneNote- ✅ **Clipboard Integration** - Copy results directly to paste in OneNote

```

- ✅ **Clipboard Support** - Copy results to paste anywhere- ✅ **No Internet Required** - Works completely offline

This will:

- Start the Python backend automatically- ✅ **Native Windows App** - Professional desktop experience- ✅ **Native Windows App** - Professional desktop experience

- Launch the Electron app

- Connect to `http://127.0.0.1:5000`



## Production Build## Development## Quick Start



To build a production installer, simply run:



```cmdStart the app in development mode:### 1. Install Dependencies

build.bat

``````cmd



This **single script** does everything:npm start```powershell



### Step 1: Build Backend Executable```cd desktop-app

- Creates standalone `backend.exe` with PyInstaller

- Includes all Python dependencies (PyTorch, OpenCV, FastAPI, etc.)npm install

- No Python required on target machines

- **Time: 5-15 minutes (first build), 2-5 minutes (subsequent)**This will:```

- The first build is slower because it downloads ~500MB of PyTorch and dependencies

- Start the Python backend automatically

### Step 2: Build Frontend

- Bundles React app for production with webpack- Launch the Electron app### 2. Build the Frontend

- **Time: 30-60 seconds**

- Connect to `http://127.0.0.1:5000`

### Step 3: Create Installer

- Packages everything into NSIS installer with electron-builder```powershell

- Creates `dist/OneNote Whiteboard Scanner Setup X.X.X.exe`

- **Time: 2-5 minutes**## Production Buildnpm install --save-dev webpack webpack-cli babel-loader @babel/core @babel/preset-react style-loader css-loader



### Total Build Timenpm install react react-dom

- **First build: ~10-20 minutes** (downloads PyTorch, OpenCV, etc.)

- **Subsequent builds: ~5-10 minutes** (uses cached dependencies)To build a production installer, simply run:npx webpack



### Build Features```

The build script includes:

- ✅ **Progress indicators** showing current step (1/3, 2/3, 3/3)```cmd

- ✅ **Real-time logging** to timestamped log file in %TEMP%

- ✅ **Time estimates** for each stepbuild.bat### 3. Run the App

- ✅ **Error detection** with automatic log tail on failure

- ✅ **File size reporting** for backend.exe and installer```

- ✅ **Status messages** so you know if it's frozen or just slow

```powershell

If a step appears frozen, check the log file path shown at the start of the build.

This **single script** does everything:npm start

### Requirements (Build Machine Only)

- Python 3.10+```

- Node.js 16+

- npm1. **Builds backend.exe** with PyInstaller (~5-10 minutes first time)

- PyInstaller (auto-installed if missing)

   - Creates standalone executable with all Python dependenciesThe app will:

### Output

```   - No Python required on target machines1. Automatically start the Python backend

desktop-app/dist/OneNote Whiteboard Scanner Setup 1.0.0.exe

```   2. Open the desktop window



### Distribution2. **Builds frontend** with webpack3. Ready to scan whiteboards!

The installer is fully self-contained:

- ✅ Includes standalone backend.exe (~500-800 MB)   - Bundles React app for production

- ✅ All dependencies bundled

- ✅ No Python needed on target machines## How to Use

- ✅ Works on clean Windows 10/11 installations

3. **Creates installer** with electron-builder

**Target machines only need:**

- Windows 10/11 (64-bit)   - NSIS installer at `dist/OneNote Whiteboard Scanner Setup X.X.X.exe`1. Click **"Start Camera"** and allow camera permissions

- ~2 GB disk space

- Internet connection (for OneNote API)2. Position your whiteboard in the frame



**Target machines DO NOT need:**### Requirements (Build Machine Only)3. Click **"Capture Photo"**

- ❌ Python installation

- ❌ pip or package managers- Python 3.10+4. Wait for AI processing (2-5 seconds)

- ❌ Visual C++ redistributables (bundled)

- Node.js 16+5. Click **"Copy to Clipboard"**

## Troubleshooting

- npm6. Open OneNote and press **Ctrl+V** to paste

### Build appears frozen

Check the log file shown at the start of the build. The PyInstaller step is normally slow:

- "Analyzing dependencies" - 1-2 minutes

- "Collecting packages" - 5-10 minutes (downloading PyTorch)### Output## Building for Distribution

- "Building executable" - 2-3 minutes

```

If it's truly frozen (no log updates for 10+ minutes), Ctrl+C and try again.

desktop-app/dist/OneNote Whiteboard Scanner Setup 1.0.0.exeCreate an installer:

### Backend build fails

```cmd```

cd ..\local-ai-backend

pip install -r requirements.txt```powershell

```

### Distributionnpm run build:win

Then check the build log file for specific errors.

The installer is fully self-contained:```

### Installer build fails

Make sure backend.exe was created:- ✅ Includes standalone backend.exe

```cmd

dir ..\local-ai-backend\dist\backend\backend.exe- ✅ All dependencies bundledThis creates:

```

- ✅ No Python needed on target machines- `dist/Whiteboard Scanner Setup.exe` - Installer

### Clean build from scratch

```cmd- ✅ Works on clean Windows 10/11 installations- `dist/Whiteboard Scanner.exe` - Portable version

cd ..\local-ai-backend

rmdir /s /q dist build



cd ..\desktop-app**Target machines only need:**## Requirements

rmdir /s /q dist

build.bat- Windows 10/11 (64-bit)

```

- ~2 GB disk space- Windows 10/11

## Project Structure

- Internet connection (for OneNote API)- Python 3.9+ (with dependencies from `../local-ai-backend/requirements.txt`)

```

desktop-app/- Webcam

├── build.bat              ← Single build script (production)

├── package.json           ← Electron + dependencies**Target machines DO NOT need:**

├── electron-main.js       ← Main process (backend startup)

├── preload.js             ← IPC bridge- ❌ Python installation## Architecture

└── renderer/

    ├── App.jsx            ← Main React component- ❌ pip or package managers

    ├── index.html         ← HTML entry point

    └── styles.css         ← Styles- ❌ Visual C++ redistributables (bundled)```



local-ai-backend/desktop-app/

├── app.py                 ← FastAPI backend

├── backend.spec           ← PyInstaller config## Troubleshooting├── electron-main.js       # Main Electron process (starts backend, manages windows)

├── requirements.txt       ← Python dependencies

└── onenote_simple.py      ← OneNote API wrapper├── preload.js            # Secure bridge between renderer and main process

```

### Backend build fails├── renderer/

## Build Process Details

```cmd│   ├── index.html        # HTML shell

### Why is the backend so large?

The backend.exe is 500-800 MB because it includes:cd ..\local-ai-backend│   ├── index.jsx         # React entry point

- Complete Python runtime (~50 MB)

- PyTorch (~500 MB) - for AI model processingpip install -r requirements.txt│   ├── App.jsx           # Main React app (camera, processing, clipboard)

- OpenCV (~100 MB) - for image processing

- Transformers, Ultralytics, and other ML libraries (~100-200 MB)```│   └── styles.css        # Styling

- All other dependencies

└── package.json          # Dependencies and build config

This is normal for bundling a complete ML/AI environment. The benefit is zero configuration on target machines.

### Installer build fails```

### Why does PyInstaller take so long?

PyInstaller must:Make sure backend.exe was created:

1. Analyze all Python imports and dependencies

2. Download/copy all package files (PyTorch alone is ~500 MB)```cmd## Differences from Office Add-in

3. Bundle everything into a single executable

4. Compress the resultdir ..\local-ai-backend\dist\backend\backend.exe



First builds are slower because it downloads everything. Subsequent builds use cached files.```| Feature | Office Add-in | Desktop App |



## Notes|---------|--------------|-------------|



- First build takes 10-20 minutes (downloads packages)### Clean build from scratch| Camera Access | ❌ Blocked | ✅ Full access |

- Subsequent builds are faster (5-10 minutes) using cached dependencies

- Backend exe is ~500-800 MB (includes PyTorch, AI models, etc.)```cmd| Installation | Upload manifest | One-click install |

- Total installer size: ~600-900 MB

- This is normal for bundling a complete Python ML environmentcd ..\local-ai-backend| Internet Required | ✅ Yes (ngrok) | ❌ No |

- The build script shows detailed progress so you can monitor each step

rmdir /s /q dist build| Works Offline | ❌ No | ✅ Yes |

## License

| OneNote Integration | Direct insert | Copy/paste |

MIT

cd ..\desktop-app| Platform | Web only | Windows native |

rmdir /s /q dist

build.bat## Troubleshooting

```

### Backend won't start

## Project StructureMake sure Python and all dependencies are installed:

```powershell

```cd ../local-ai-backend

desktop-app/pip install -r requirements.txt

├── build.bat              ← Single build script (production)```

├── package.json           ← Electron + dependencies

├── electron-main.js       ← Main process (backend startup)### Camera not working

├── preload.js             ← IPC bridge- Check camera permissions in Windows Settings

└── renderer/- Make sure no other app is using the camera

    ├── App.jsx            ← Main React component- Try restarting the app

    ├── index.html         ← HTML entry point

    └── styles.css         ← Styles### "Cannot find module" errors

```powershell

local-ai-backend/rm -r node_modules

├── app.py                 ← FastAPI backendnpm install

├── backend.spec           ← PyInstaller confignpx webpack

├── requirements.txt       ← Python dependenciesnpm start

└── onenote_simple.py      ← OneNote API wrapper```

```

## License

## Notes

MIT

- First build takes 5-10 minutes (PyInstaller compiles everything)
- Subsequent builds are faster (cached dependencies)
- Backend exe is ~500-800 MB (includes PyTorch, AI models, etc.)
- Total installer size: ~600-900 MB
- This is normal for bundling a complete Python environment

## License

MIT
