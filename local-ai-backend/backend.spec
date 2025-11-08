# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for creating standalone backend executable

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files
datas = [
    ('requirements.txt', '.'),
]

# Try to collect torch data files if available
try:
    datas += collect_data_files('torch')
except:
    pass

# Collect all hidden imports for complex packages
hiddenimports = [
    # FastAPI and Uvicorn
    'fastapi',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    
    # Starlette
    'starlette',
    'starlette.applications',
    'starlette.middleware',
    'starlette.middleware.cors',
    'starlette.responses',
    'starlette.routing',
    
    # Pydantic
    'pydantic',
    'pydantic.fields',
    'pydantic.main',
    'pydantic_core',
    
    # Core dependencies
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    
    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'cv2',
    'numpy',
    'scipy',
    'scikit-image',
    'skimage',
    'skimage.transform',
    'skimage.filters',
    
    # AI/ML
    'torch',
    'torchvision',
    'ultralytics',
    'onnxruntime',
    
    # Utilities
    'yaml',
    'psutil',
    'coloredlogs',
    'humanfriendly',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Exclude if not needed
        'tkinter',     # GUI not needed
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console window for backend logging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend',
)
