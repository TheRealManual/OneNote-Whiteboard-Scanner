# OneNote Whiteboard Scanner

A Windows desktop application for digitizing whiteboard content using computer vision and AI. Captures images from a webcam, processes them locally, and outputs vector graphics and Ink Strokes that can be sent directly to OneNote notebooks.

Version: 1.0
License: MIT
Platform: Windows

## Overview

This application processes whiteboard photographs into clean, editable vector graphics. It uses a hybrid computer vision pipeline combining classical CV techniques with optional AI models for improved accuracy. All processing occurs locally on the user's machine.

## Core Capabilities

- Direct webcam access through native desktop application
- Local image processing using OpenCV and scikit-image
- Color-accurate stroke detection and preservation
- SVG vector output for scalable graphics
- Microsoft Graph API integration for OneNote
- Windows clipboard integration
- Hardware acceleration support (OpenVINO, DirectML)

## Technical Features

- Hybrid CV pipeline
- Illumination normalization and perspective correction
- Stroke skeletonization and vectorization
- Multi-backend support (OpenVINO, ONNX DirectML, CPU fallback)
- Offline processing capability
- Per-user credential storage

## System Requirements

- Windows 10 or 11 (64-bit)
- Python 3.9 or higher (development)
- Node.js 18 or higher (development)
- Webcam (built-in or USB)
- Minimum 2 GB available disk space

## Installation (Development)

1. Clone the repository:

```bash
git clone https://github.com/TheRealManual/OneNote-Whiteboard-Scanner.git
cd OneNote-Whiteboard-Scanner
```

2. Install Python dependencies:

```bash
cd local-ai-backend
pip install -r requirements.txt
```

3. Install Node.js dependencies and build frontend:

```bash
cd ../desktop-app
npm install
npm install --save-dev webpack webpack-cli babel-loader @babel/core @babel/preset-react style-loader css-loader
npm install react react-dom
npx webpack
```

4. Start the app:

```bash
npm start
```

## Portable Distribution

Run `build-portable.bat` to create a standalone portable package. The package includes a compiled backend executable and all required assets.

## Usage

1. Launch the application using `start.bat` or `npm start`.
2. In the OneNote integration panel click Connect to OneNote and sign in with a Microsoft account.
3. Select the target notebook and section.
4. Click Start Camera, position the whiteboard, and Capture Photo.
5. Review the processed output and choose to Send to OneNote or Copy to Clipboard.

## Configuration

Create a `.env` file in the repository root with the following variables:

```env
ONENOTE_CLIENT_ID=your_client_id
ONENOTE_CLIENT_SECRET=your_client_secret
BACKEND_HOST=127.0.0.1
BACKEND_PORT=5000
OAUTH_REDIRECT_URI=http://localhost:8888/callback
OAUTH_SCOPES=Notes.ReadWrite Notes.Create offline_access
PRODUCTION=false
```

Edit `local-ai-backend/config_hybrid.json` to set processing parameters.

## Project Structure

```
OneNote-Whiteboard-Scanner/
├── start.bat
├── test.bat
├── build-portable.bat
├── desktop-app/
├── local-ai-backend/
└── tests/
```

## Troubleshooting

- If the backend fails to start, verify Python and dependencies and that port 5000 is free.
- If camera access is denied, verify Windows camera permissions.
- If OneNote connection fails, verify internet connectivity and OAuth settings.

## Tests

Run backend tests:

```bash
cd local-ai-backend
pytest tests/
```

## License

MIT License. See the LICENSE file for details.

## Support

Report issues on the GitHub repository issues page.
