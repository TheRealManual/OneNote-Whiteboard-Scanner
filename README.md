# OneNote Whiteboard Scanner# OneNote AI Whiteboard Scanner - Desktop App



Desktop application for scanning and digitizing whiteboards with AI processing and direct OneNote integration.[![Version](https://img.shields.io/badge/version-1.0-blue.svg)](https://github.com/TheRealManual/OneNote-Whiteboard-Scanner)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start[![Platform](https://img.shields.io/badge/platform-Windows-blue.svg)](https://github.com/TheRealManual/OneNote-Whiteboard-Scanner)



### Development Mode## 📋 Overview

Double-click **`start.bat`** to run the app in development mode.

- Starts Python backend automaticallyA standalone Windows desktop application that transforms physical whiteboard sketches into editable digital ink! Uses local AI processing to capture, clean, and vectorize whiteboard content from your laptop's camera, then sends directly to your personal OneNote notebooks.

- Opens Electron desktop app

- Hot reload enabled### Key Features



### Test Backend- 🎯 **Full Camera Access**: Direct camera capture in native Windows app

Double-click **`test.bat`** to verify backend is working.- 🎨 **Color Preservation**: Accurately detects and preserves marker colors

- Checks dependencies- ✏️ **Editable Output**: Generates vector SVG for crisp scaling

- Starts backend- � **OneNote Integration**: Send scans directly to YOUR notebooks and sections

- Tests health endpoint- 👤 **Multi-User Support**: Each user signs in with their own Microsoft account

- �🔒 **Privacy-First**: 100% local processing, credentials stay on your machine

### Build Portable Package- ⚡ **Fast Processing**: 1-3 second average processing time (Intel GPU accelerated)

Double-click **`build-portable.bat`** to create a distributable package.- 📋 **Clipboard Integration**: One-click copy to paste anywhere

- Builds standalone backend.exe (no Python needed)- 🌐 **Works Offline**: No internet connection required for processing

- Packages frontend- 🚀 **Multi-Backend**: Auto-detects best AI backend (OpenVINO/DirectML/CPU)

- Creates ZIP file ready to distribute

## 👥 Per-User Design

**Output**: `OneNote-Whiteboard-Scanner-Portable.zip`

This application is designed for **individual users**, not multi-tenant cloud services:

Users extract the ZIP and run `Run OneNote Scanner.bat` - no installation needed!

✅ **Each user installs on their own computer**  

## Requirements✅ **Each user signs in with their own Microsoft account**  

✅ **Each user connects to their own OneNote notebooks**  

**Development:**✅ **All user data stored locally** on their machine  

- Python 3.10+✅ **No shared database** or central server  

- Node.js 16+

- npm### How It Works



**Distribution (portable package):**1. **User A** installs on laptop → Signs in as `alice@school.edu` → Sees Alice's notebooks

- Nothing! The ZIP includes everything.2. **User B** installs on desktop → Signs in as `bob@company.com` → Sees Bob's notebooks  

3. **User C** uses at work → Signs in as `carol@university.edu` → Sees Carol's notebooks

## Project Structure

Each installation is completely isolated and private!

```

OneNote-Whiteboard-Scanner/## 🏗️ Architecture

├── start.bat                    ← Run in development mode

├── test.bat                     ← Test backend```

├── build-portable.bat           ← Build portable package┌─────────────────────────────────┐

├── local-ai-backend/│    Electron Desktop App         │

│   ├── app.py                   ← FastAPI backend│  ───────────────────────────    │

│   ├── onenote_simple.py        ← OneNote API│  • React UI                     │

│   ├── config.py                ← Configuration│  • Native Camera Access         │

│   ├── backend.spec             ← PyInstaller config│  • Sends Image to AI API        │

│   ├── requirements.txt         ← Python dependencies│  • Clipboard Integration        │

│   └── ai/                      ← AI processing modules└──────────────┬──────────────────┘

└── desktop-app/               │ Localhost API (HTTP)

    ├── electron-main.js         ← Electron main process┌──────────────▼──────────────────┐

    ├── preload.js               ← IPC bridge│     Local AI Engine (FastAPI)   │

    ├── package.json             ← Node dependencies│  • Hybrid Extractor (1-3 sec)   │

    └── renderer/│  • Classical CV + Optional AI   │

        ├── App.jsx              ← React app│  • Illumination Correction      │

        ├── index.html           ← HTML shell│  • Whiteboard Detection         │

        └── styles.css           ← Styling│  • Stroke Skeletonization       │

```│  • Returns JSON {svg, metadata} │

└─────────────────────────────────┘

## Features                 │

           ┌─────▼──────┐

- ✅ **Full Camera Access** - No browser restrictions           │  User      │

- ✅ **Local AI Processing** - Fast, private, offline-capable           │  Ctrl+V    │──► OneNote

- ✅ **OneNote Integration** - Browse and upload directly           │  (Paste)   │

- ✅ **Create Sections** - Create new OneNote sections on the fly           └────────────┘

- ✅ **Instant Navigation** - Cached hierarchy for instant browsing```

- ✅ **Clipboard Support** - Copy results to paste anywhere

- ✅ **Native Windows App** - Professional desktop experience## 🚀 Quick Start



## Build Times### Prerequisites



**First build**: ~10-20 minutes (downloads PyTorch, OpenCV, etc.)- **Windows** 10/11

**Subsequent builds**: ~5-10 minutes (uses cached dependencies)- **Node.js** 18+

- **Python** 3.9+

## Distribution- **Webcam** (built-in or external)

- **Visual Studio Code** (recommended)

The portable package is fully self-contained:

- ✅ Standalone backend.exe (~500-800 MB with AI models)### Installation

- ✅ All dependencies bundled

- ✅ No Python needed on target machines#### 1. Clone the Repository

- ✅ No admin rights needed

- ✅ Works on clean Windows 10/11 installations```bash

git clone https://github.com/TheRealManual/OneNote-Whiteboard-Scanner.git

Target machines only need:cd OneNote-Whiteboard-Scanner

- Windows 10/11 (64-bit)```

- ~2 GB disk space

- Internet connection (for OneNote API)#### 2. Set Up Backend (Local AI Engine)



## License```bash

cd local-ai-backend

MITpip install -r requirements.txt


# Or use the quick installer for hybrid mode
cd ..
.\install-hybrid.bat
```

#### 3. Set Up Desktop App

```bash
cd desktop-app
npm install

# Install build dependencies
npm install --save-dev webpack webpack-cli babel-loader @babel/core @babel/preset-react style-loader css-loader

# Install React dependencies
npm install react react-dom

# Build the frontend
npx webpack
```

#### 4. Run the App

```bash
npm start
```

The desktop app will:
- Automatically start the Python backend on `http://127.0.0.1:5000`
- Open the Electron window
- Grant full camera access

## 📖 Usage

### First Time Setup

1. Launch the app (run `.\run.bat` or `npm start` in desktop-app folder)
2. Click **"Connect to OneNote"** in the OneNote Integration panel
3. Browser opens → Sign in with YOUR Microsoft account (free - Outlook, Hotmail, etc.)
4. Select which of YOUR notebooks to send scans to
5. Select which section within that notebook
6. Done! Your preferences are saved locally

### Daily Use

1. Click **"Start Camera"** and allow camera permissions
2. Position your whiteboard in the camera frame
3. Click **"Capture Photo"**
4. Wait 1-3 seconds for AI processing
5. Click **"Send to OneNote"** → Instantly appears in your selected section!
6. Or click **"Copy to Clipboard"** → Paste anywhere with Ctrl+V

### Managing Your Connection

- **Current user**: Shown at top of OneNote panel (`👤 your.email@example.com`)
- **Change destination**: Click "Change Destination" to pick a different notebook/section
- **Switch accounts**: Click "Logout" then reconnect with a different Microsoft account
- **View selection**: Your current destination is always displayed

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Desktop App** | Electron 28, React 18 |
| **Backend** | FastAPI, OpenCV, scikit-image |
| **AI Processing** | Hybrid classical CV + optional U2-Net, skeletonization, RDP vectorization |
| **Acceleration** | OpenVINO (Intel GPU), ONNX DirectML |
| **Data Format** | SVG (Scalable Vector Graphics) |
| **Integration** | Windows Clipboard API |

## 📁 Project Structure

```
OneNote-Whiteboard-Scanner/
│
├── desktop-app/                 # Electron Desktop Application
│   ├── electron-main.js         # Main Electron process
│   ├── preload.js               # Secure IPC bridge
│   ├── package.json
│   ├── webpack.config.js
│   ├── renderer/                # React UI
│   │   ├── index.html
│   │   ├── index.jsx
│   │   ├── App.jsx              # Main React component
│   │   └── styles.css
│   └── README.md
│
├── local-ai-backend/            # Python AI Engine
│   ├── app.py                   # FastAPI entry point
│   ├── config.py                # Configuration
│   ├── config_hybrid.json       # Hybrid extractor settings
│   ├── ai/                      # AI processing modules
│   │   ├── hybrid_extractor.py  # Fast hybrid CV+AI pipeline
│   │   ├── stroke_extract.py    # Stroke objects
│   │   └── vectorize.py         # SVG generation
│   └── requirements.txt
│
└── README.md
```

## 🔧 Configuration

### Environment Variables (`.env` file)

The app uses a **centralized OAuth application** so users don't need Azure accounts:

```env
# Centralized Azure App (shared for all users)
ONENOTE_CLIENT_ID=0ec33887-e96d-4b34-9b66-8871590ad8bb
ONENOTE_CLIENT_SECRET=mul8Q~XLE...

# Backend Settings
BACKEND_HOST=127.0.0.1
BACKEND_PORT=5000

# OAuth Settings
OAUTH_REDIRECT_URI=http://localhost:8888/callback
OAUTH_SCOPES=Notes.ReadWrite Notes.Create offline_access

# Mode
PRODUCTION=false
```

**Important:** Users DON'T need their own Azure apps! They just sign in with their Microsoft account.

### Per-User Configuration (`user_onenote_config.json`)

Auto-created when each user connects. Contains ONLY that user's preferences:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "user_email": "user@example.com",
  "notebook_id": "...",
  "notebook_name": "User's Notebook",
  "section_id": "...",
  "section_name": "Scanned Whiteboards"
}
```

✅ Stored locally on user's machine  
✅ Excluded from git (`.gitignore`)  
✅ Contains only that user's OneNote access  

### Backend Configuration

Edit `local-ai-backend/config_hybrid.json` for processing settings:

```json
{
  "target_size": [960, 540],
  "preserve_aspect_ratio": true,
  "colorize_from_source": false,
  "min_stroke_points": 3
}
```

## 🧪 Testing

```bash
# Backend tests
cd local-ai-backend
pytest tests/

# Frontend tests
cd addin-frontend
npm test
```

## 📊 Performance

- **Average Processing Time**: 1-3 seconds (hybrid mode with Intel GPU)
- **Supported Image Size**: Up to 1280×720 pixels (auto-resized)
- **Supported Formats**: JPEG, PNG
- **Color Detection**: Automatic HSV-based color detection
- **Backend Options**: OpenVINO (Intel GPU), ONNX DirectML, CPU fallback

## 🔐 Security & Privacy

### Data Privacy ✅

- ✅ All image processing happens **locally** on your machine
- ✅ No data sent to external servers (except Microsoft Graph API for OneNote)
- ✅ Your OneNote credentials stored **only on your device**
- ✅ Each user's config file is separate and local
- ✅ Backend only listens on localhost (127.0.0.1)
- ✅ OAuth2 authentication with Microsoft (industry standard)

### Multi-User Security

- **User A** on Computer 1 → `user_onenote_config.json` with Alice's tokens
- **User B** on Computer 2 → `user_onenote_config.json` with Bob's tokens
- **User C** on Computer 3 → `user_onenote_config.json` with Carol's tokens

No shared storage = complete isolation!

### Best Practices

✅ `.env` file excluded from git (contains app secrets)  
✅ `user_onenote_config.json` excluded from git (contains user tokens)  
✅ HTTPS used for all Microsoft API calls  
✅ Refresh tokens for long-term access without re-authentication  
✅ Logout feature to clear credentials when needed  

## 🚀 Deployment for End Users

### For Distribution

1. **Push to GitHub** (secrets automatically excluded via `.gitignore`)
2. **Users clone** the repository
3. **Users run setup:**
   ```bash
   # Backend
   cd local-ai-backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../desktop-app
   npm install
   npm run build
   
   # Run
   cd ..
   .\run.bat
   ```
4. **Users sign in** with their own Microsoft accounts
5. **Each user selects** their own notebooks/sections

### Creating Installers (Optional)

Package as standalone .exe for easier distribution:

```bash
cd desktop-app
npm install electron-builder --save-dev
npm run dist  # Creates Windows installer
```

Users can then:
- Download single `.exe` file
- Install with one click
- Sign in with Microsoft account
- Start scanning!

## ❓ FAQ

**Q: Will other users see my notebooks?**  
A: No! Each user signs in with their own Microsoft account and only sees their own notebooks.

**Q: Do I need to create an Azure app?**  
A: No! The app uses a centralized OAuth app. Just sign in with your Microsoft account (free).

**Q: Where is my data stored?**  
A: Your settings are in `user_onenote_config.json` on your local computer. Not in the cloud.

**Q: Can I use this on multiple devices?**  
A: Yes! Install on each device and sign in. Each maintains its own local config.

**Q: How do I switch Microsoft accounts?**  
A: Click "Logout" in the OneNote panel, then click "Connect to OneNote" and sign in with a different account.

**Q: Is my data safe?**  
A: Yes! Processing is 100% local. Only the final image is sent to Microsoft OneNote via their official API (same as using OneNote normally).

**Q: Do I need internet?**  
A: For AI processing: No. For OneNote sync: Yes (to send scans to cloud).

## 🗺️ Roadmap

- [ ] Real-time video mode (live detection)
- [ ] Handwriting → text recognition (OCR layer)
- [ ] Multi-user whiteboard session sync
- [ ] Cloud fallback mode if local AI unavailable
- [ ] Drawing layer insertion option
- [ ] Mobile app support

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Office.js team
- OpenCV community
- FastAPI framework

## 📞 Support

For issues and questions:
- 🐛 [Report a Bug](https://github.com/TheRealManual/OneNote-Whiteboard-Scanner/issues)
- 💡 [Request a Feature](https://github.com/TheRealManual/OneNote-Whiteboard-Scanner/issues)
- 📧 Email: support@example.com

---

Made with ❤️ for better note-taking
