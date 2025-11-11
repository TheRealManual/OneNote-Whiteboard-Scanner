# Exporter - Build & Distribution Tools

This folder contains all scripts and tools for building installers and distributing the OneNote Whiteboard Scanner.

## Build Options

### 1. Light Installer (Recommended) - ~100 MB
**Requires:** User has Python 3.9+ installed

```bash
build-light.bat
```

**What it includes:**
- Electron desktop app
- Python backend source code
- Requirements.txt for dependency installation
- Auto-installs Python packages on first run

**Pros:**
- ✅ Small download size (~100 MB)
- ✅ Easy to update
- ✅ Fast build time

**Cons:**
- ❌ Requires Python pre-installed
- ❌ Needs internet on first run

---

### 2. Portable Build (Standalone) - ~700 MB
**Requires:** Nothing - fully self-contained

```bash
build-portable.bat
```

**What it includes:**
- Electron desktop app
- Embedded Python runtime
- All Python packages pre-installed
- PyTorch model files
- Works 100% offline

**Pros:**
- ✅ No installation required
- ✅ Works offline
- ✅ No dependencies

**Cons:**
- ❌ Large file size (~700 MB)
- ❌ Slow build time (15-20 minutes)
- ❌ May trigger antivirus warnings

---

## Build Requirements

### Light Installer
- Node.js 18+
- npm
- Electron Builder

### Portable Build
- Python 3.9+
- PyInstaller
- All dependencies in local-ai-backend/requirements.txt
- ~2 GB free disk space

---

## Output Files

Builds are created in:
```
exporter/
  └── dist/
      ├── OneNote-Scanner-Setup-1.0.0.exe          (Light installer)
      ├── OneNote-Scanner-Portable-1.0.0.zip       (Portable build)
      └── latest.yml                                (Auto-update metadata)
```

---

## Distribution Checklist

Before releasing:
- [ ] Test light installer on clean Windows machine
- [ ] Test portable build offline
- [ ] Verify Python auto-install works
- [ ] Check file sizes are reasonable
- [ ] Test auto-update mechanism
- [ ] Scan with Windows Defender
- [ ] Sign executables (optional but recommended)

---

## Code Signing (Optional)

To avoid "Unknown Publisher" warnings:

1. Get a code signing certificate ($100-300/year)
2. Set environment variables:
   ```bash
   set CSC_LINK=path\to\certificate.pfx
   set CSC_KEY_PASSWORD=your_password
   ```
3. Rebuild with signing enabled

---

## Quick Reference

| Build Type | Size | Build Time | User Requires | Offline? |
|------------|------|------------|---------------|----------|
| Light      | 100 MB | 5 min    | Python 3.9+   | No       |
| Portable   | 700 MB | 20 min   | Nothing       | Yes      |

---

## Troubleshooting

**"Python not found" error:**
- Ensure Python 3.9+ is in PATH
- Try: `python --version`

**PyInstaller fails:**
- Run: `pip install --upgrade pyinstaller`
- Clear: `rmdir /s /q build dist`

**Large portable size:**
- Normal - PyTorch is huge
- Can't reduce much without removing ML features

**Antivirus blocks portable build:**
- Add exclusion for exporter/dist folder
- Submit false positive report to antivirus vendor
