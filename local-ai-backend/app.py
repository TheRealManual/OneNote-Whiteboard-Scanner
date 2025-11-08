"""
FastAPI Backend for OneNote Whiteboard Scanner
Processes whiteboard images and converts them to SVG vectors using Hybrid Extractor
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add libs folder to Python path for bundled dependencies
script_dir = Path(__file__).parent
libs_dir = script_dir / 'libs'
if libs_dir.exists():
    sys.path.insert(0, str(libs_dir))
    print(f"[OK] Added bundled dependencies from: {libs_dir}")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import logging.config
from typing import Dict, Optional
from datetime import datetime
import traceback
import json
import os

# Load configuration
from config import (
    HOST, PORT, ALLOWED_ORIGINS, MAX_IMAGE_SIZE, 
    LOGGING_CONFIG, validate_config, PRODUCTION,
    USER_CONFIG_FILE
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Validate configuration on startup
validate_config()

# Lazy import for heavy AI dependencies - only load when needed
_extractor = None
def get_ai_dependencies():
    """Lazy load heavy AI dependencies only when processing images"""
    global _extractor
    from ai.hybrid_extractor import get_extractor
    from ai.stroke_extract import Stroke
    from ai.vectorize import strokes_to_svg
    from ai.inkml_export import strokes_to_inkml
    
    if _extractor is None:
        logger.info("Loading AI models (first-time initialization)...")
        _extractor = get_extractor()
        logger.info("AI models loaded successfully")
    
    return {
        'extractor': _extractor,
        'strokes_to_svg': strokes_to_svg,
        'strokes_to_inkml': strokes_to_inkml,
        'Stroke': Stroke
    }

# OneNote integration (optional)
try:
    from onenote_simple import SimpleOneNoteAuth
    ONENOTE_AVAILABLE = True  # Module imported successfully
    logger.info("OneNote integration available")
except ImportError as e:
    ONENOTE_AVAILABLE = False
    logger.warning(f"OneNote integration not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OneNote Whiteboard AI Engine",
    description="Local AI processing for whiteboard capture and vectorization",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add Private Network Access headers
@app.middleware("http")
async def add_private_network_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "OneNote Whiteboard AI Engine",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "max_image_size": MAX_IMAGE_SIZE,
        "onenote_available": ONENOTE_AVAILABLE,
        "ai_models_loaded": _extractor is not None
    }


@app.get("/onenote/config")
async def get_onenote_config() -> Dict:
    """Get current OneNote configuration"""
    if not ONENOTE_AVAILABLE:
        raise HTTPException(status_code=404, detail="OneNote not configured")
    
    try:
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        return {
            "notebook_id": config.get('notebook_id', ''),
            "notebook_name": config.get('notebook_name', ''),
            "section_id": config.get('section_id', ''),
            "section_name": config.get('section_name', ''),
            "user_email": config.get('user_email', ''),
            "notebooks_cache": config.get('notebooks_cache', []),
            "full_hierarchy_cache": config.get('full_hierarchy_cache', {}),
            "cache_timestamp": config.get('cache_timestamp', ''),
            "has_auth": bool(config.get('access_token'))
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration not found")
    except Exception as e:
        logger.error(f"Config load error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load configuration")


@app.post("/onenote/setup")
async def setup_onenote() -> Dict:
    """Trigger OneNote setup and return notebooks"""
    try:
        auth = SimpleOneNoteAuth()
        
        # Check if already configured
        if os.path.exists(str(USER_CONFIG_FILE)):
            with open(str(USER_CONFIG_FILE)) as f:
                config = json.load(f)
            auth.access_token = config.get('access_token')
            auth.refresh_token = config.get('refresh_token')
            
            # Check if we have cached notebooks
            cached_notebooks = config.get('notebooks_cache')
            if cached_notebooks:
                logger.info("Using cached notebooks list")
                user_email = config.get('user_email', auth.get_user_email())
                return {
                    "success": True,
                    "notebooks": cached_notebooks,
                    "user_email": user_email
                }
        else:
            # Need fresh auth
            if not auth.authenticate():
                raise HTTPException(status_code=401, detail="Authentication failed")
            
            # Get user email and notebooks
            user_email = auth.get_user_email()
            notebooks = auth.get_notebooks()
            notebooks_cache = [{"id": nb['id'], "name": nb['displayName']} for nb in notebooks]
            
            # Save tokens and cached notebooks
            config = {
                'access_token': auth.access_token,
                'refresh_token': auth.refresh_token,
                'user_email': user_email,
                'notebooks_cache': notebooks_cache,
                'notebook_id': '',
                'notebook_name': '',
                'section_id': '',
                'section_name': ''
            }
            
            # Ensure directory exists
            USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Config directory: {USER_CONFIG_FILE.parent}, exists: {USER_CONFIG_FILE.parent.exists()}")
            
            try:
                with open(str(USER_CONFIG_FILE), 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Successfully wrote config file to: {USER_CONFIG_FILE}")
                logger.info(f"File exists after write: {USER_CONFIG_FILE.exists()}")
                logger.info(f"Saved OneNote config for user: {user_email}")
            except Exception as write_error:
                logger.error(f"Failed to write config file: {write_error}")
                logger.error(f"Config path: {USER_CONFIG_FILE}")
                logger.error(f"Directory writable: {os.access(str(USER_CONFIG_FILE.parent), os.W_OK)}")
                raise
            
            return {
                "success": True,
                "notebooks": notebooks_cache,
                "user_email": user_email
            }
        
        # If no cache, fetch fresh notebooks
        notebooks = auth.get_notebooks()
        notebooks_cache = [{"id": nb['id'], "name": nb['displayName']} for nb in notebooks]
        user_email = auth.get_user_email()
        
        # Update cache
        if os.path.exists(str(USER_CONFIG_FILE)):
            with open(str(USER_CONFIG_FILE)) as f:
                config = json.load(f)
            config['notebooks_cache'] = notebooks_cache
            config['user_email'] = user_email
            with open(str(USER_CONFIG_FILE), 'w') as f:
                json.dump(config, f, indent=2)
        
        return {
            "success": True,
            "notebooks": notebooks_cache,
            "user_email": user_email
        }
    except Exception as e:
        logger.error(f"Setup error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onenote/logout")
async def logout_onenote() -> Dict:
    """Logout and clear OneNote configuration"""
    try:
        # Read config to get tokens before deleting
        logout_url = None
        if os.path.exists(str(USER_CONFIG_FILE)):
            try:
                with open(str(USER_CONFIG_FILE)) as f:
                    config = json.load(f)
                
                # Build Microsoft logout URL to clear browser session
                # This will invalidate the consent and force fresh permissions next login
                logout_url = "https://login.microsoftonline.com/common/oauth2/v2.0/logout"
                logger.info("Prepared Microsoft logout URL for browser")
            except Exception as read_error:
                logger.warning(f"Could not read config for cleanup: {read_error}")
            
            # Delete the config file - this clears all local tokens and cache
            os.remove(str(USER_CONFIG_FILE))
            logger.info("OneNote configuration cleared - all tokens and cache deleted")
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "logout_url": logout_url  # Frontend can open this to clear Microsoft session
        }
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onenote/refresh-notebooks")
async def refresh_notebooks() -> Dict:
    """Force refresh the notebooks list from Microsoft Graph API"""
    try:
        if not os.path.exists(str(USER_CONFIG_FILE)):
            raise HTTPException(status_code=401, detail="Not connected to OneNote")
        
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        auth = SimpleOneNoteAuth()
        auth.access_token = config.get('access_token')
        auth.refresh_token = config.get('refresh_token')
        
        # Fetch fresh notebooks
        notebooks = auth.get_notebooks()
        notebooks_cache = [{"id": nb['id'], "name": nb['displayName']} for nb in notebooks]
        
        # Update cache
        config['notebooks_cache'] = notebooks_cache
        with open(str(USER_CONFIG_FILE), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Refreshed notebooks cache")
        
        return {
            "success": True,
            "notebooks": notebooks_cache
        }
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onenote/save-selection")
async def save_onenote_selection(request: dict) -> Dict:
    """Save user's notebook and section selection"""
    try:
        if not os.path.exists(str(USER_CONFIG_FILE)):
            raise HTTPException(status_code=401, detail="Not connected to OneNote")
        
        # Load existing config
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        # Update selection
        if 'notebook_id' in request:
            config['notebook_id'] = request['notebook_id']
        if 'notebook_name' in request:
            config['notebook_name'] = request['notebook_name']
        if 'section_id' in request:
            config['section_id'] = request['section_id']
        if 'section_name' in request:
            config['section_name'] = request['section_name']
        
        # Save updated config
        with open(str(USER_CONFIG_FILE), 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            "success": True,
            "message": "Selection saved"
        }
    except Exception as e:
        logger.error(f"Save selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onenote/create-section")
async def create_onenote_section(request: dict) -> Dict:
    """
    Create a new section in a notebook or section group
    
    Args:
        request: {
            notebook_id: str,
            section_name: str,
            section_group_id: str (optional)
        }
    """
    if not ONENOTE_AVAILABLE:
        raise HTTPException(status_code=404, detail="OneNote not available")
    
    try:
        if not os.path.exists(str(USER_CONFIG_FILE)):
            raise HTTPException(status_code=401, detail="Not connected to OneNote")
        
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        # Initialize auth
        auth = SimpleOneNoteAuth()
        auth.access_token = config.get('access_token')
        auth.refresh_token = config.get('refresh_token')
        
        # Create the section
        result = auth.create_section(
            notebook_id=request['notebook_id'],
            section_name=request['section_name'],
            section_group_id=request.get('section_group_id')
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create section")
        
        logger.info(f"Created section: {result['section_name']} (ID: {result['section_id']})")
        
        return {
            "success": True,
            "section_id": result['section_id'],
            "section_name": result['section_name']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create section error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/onenote/sections")
async def get_onenote_sections(notebook_id: str, section_group_id: str = None) -> Dict:
    """
    Get sections for a specific notebook or section group
    
    Args:
        notebook_id: The notebook ID
        section_group_id: Optional section group ID to get sections within a group
    """
    if not ONENOTE_AVAILABLE:
        raise HTTPException(status_code=404, detail="OneNote not available")
    
    try:
        # Try to load config, but it's OK if it doesn't exist yet
        if not os.path.exists(str(USER_CONFIG_FILE)):
            raise HTTPException(status_code=401, detail="Not connected to OneNote. Please connect first.")
        
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        auth = SimpleOneNoteAuth()
        auth.access_token = config.get('access_token')
        auth.refresh_token = config.get('refresh_token')
        
        # Get sections (either from notebook root or from a section group)
        if section_group_id:
            sections = auth.get_sections_in_group(section_group_id)
        else:
            sections = auth.get_sections(notebook_id)
        
        # Get section groups in this notebook (only if getting root-level sections)
        section_groups = []
        if not section_group_id:
            section_groups = auth.get_section_groups(notebook_id)
            logger.info(f"Raw section groups data: {section_groups[:2] if section_groups else []}")
        
        section_groups_formatted = []
        for sg in section_groups:
            formatted = {"id": sg['id'], "name": sg.get('displayName', sg.get('name', 'Unnamed Group'))}
            section_groups_formatted.append(formatted)
            logger.info(f"Section group: {formatted}")
        
        return {
            "success": True,
            "sections": [{"id": sec['id'], "name": sec['displayName']} for sec in sections],
            "section_groups": section_groups_formatted
        }
    except FileNotFoundError:
        raise HTTPException(status_code=401, detail="Not connected to OneNote")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sections error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/onenote/fetch-all")
async def fetch_all_onenote_data() -> Dict:
    """
    Fetch complete OneNote hierarchy (all notebooks, section groups, sections, and pages)
    This is used for initial caching to make navigation instant
    """
    if not ONENOTE_AVAILABLE:
        raise HTTPException(status_code=404, detail="OneNote not available")
    
    try:
        if not os.path.exists(str(USER_CONFIG_FILE)):
            raise HTTPException(status_code=401, detail="Not connected to OneNote")
        
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        # Initialize auth and fetch all data
        auth = SimpleOneNoteAuth()
        
        # Restore tokens from config
        auth.access_token = config.get('access_token')
        auth.refresh_token = config.get('refresh_token')
        
        logger.info("Starting complete OneNote data fetch...")
        all_data = auth.fetch_all_onenote_data()
        
        # Cache the data in config for future loads
        config['full_hierarchy_cache'] = all_data
        config['cache_timestamp'] = datetime.now().isoformat()
        
        with open(str(USER_CONFIG_FILE), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Complete data fetch finished and cached")
        return all_data
        
    except FileNotFoundError:
        raise HTTPException(status_code=401, detail="Not connected to OneNote")
    except Exception as e:
        logger.error(f"Fetch all data error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onenote/send")
async def send_to_onenote(request: dict) -> Dict:
    """
    Send whiteboard ink strokes to OneNote as editable ink
    
    Args:
        request: {inkml_data: InkML XML, section_id: optional}
        
    Returns:
        Created page info
    """
    if not ONENOTE_AVAILABLE:
        raise HTTPException(status_code=501, detail="OneNote not configured. Run: python onenote_simple.py")
    
    try:
        # Load user config
        with open(str(USER_CONFIG_FILE)) as f:
            config = json.load(f)
        
        # Get InkML data and parse it to extract strokes
        inkml_data = request.get('inkml_data', '')
        if not inkml_data:
            raise HTTPException(status_code=400, detail="No InkML data provided")
        
        # Parse InkML to extract stroke data
        import xml.etree.ElementTree as ET
        root = ET.fromstring(inkml_data)
        
        # Find all trace elements (strokes)
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        traces = root.findall('.//inkml:trace', ns)
        
        strokes_data = []
        for idx, trace in enumerate(traces):
            trace_id = trace.get('id', f'stroke-{idx}')
            points_text = trace.text.strip() if trace.text else ''
            
            # Parse points: "x1 y1 x2 y2 ..." format
            coords = points_text.split()
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    try:
                        x = float(coords[i])
                        y = float(coords[i + 1])
                        points.append([x, y])
                    except ValueError:
                        continue
            
            if points:
                strokes_data.append({
                    'id': trace_id,
                    'points': points
                })
        
        logger.info(f"Parsed {len(strokes_data)} strokes from InkML for OneNote")
        
        # Create OneNote auth with saved token
        auth = SimpleOneNoteAuth()
        auth.access_token = config.get('access_token')
        auth.refresh_token = config.get('refresh_token')
        
        # Create page with ink strokes using native OneNote ink API
        section_id = request.get('section_id') or config.get('section_id')
        result = auth.create_page_with_ink_strokes(
            section_id=section_id,
            strokes_data=strokes_data
        )
        
        # Save refreshed tokens if they were updated
        if auth.access_token != config.get('access_token') or auth.refresh_token != config.get('refresh_token'):
            config['access_token'] = auth.access_token
            config['refresh_token'] = auth.refresh_token
            with open(str(USER_CONFIG_FILE), 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Updated tokens after refresh")
        
        if result and result.get('success'):
            return {
                "success": True,
                "page_id": result['page_id'],
                "page_url": result['page_url'],
                "message": f"✓ Sent editable ink to {config.get('section_name', 'OneNote')}"
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to create page'))
            
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="OneNote not configured. Run: python onenote_simple.py")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OneNote send error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)) -> Dict:
    """
    Process whiteboard image and return SVG vectorization
    Uses Hybrid Extractor for fast 1-3 second processing
    """
    try:
        # Lazy load AI dependencies
        ai_deps = get_ai_dependencies()
        extractor = ai_deps['extractor']
        strokes_to_svg = ai_deps['strokes_to_svg']
        strokes_to_inkml = ai_deps['strokes_to_inkml']
        Stroke = ai_deps['Stroke']
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Received image: {img.shape}")
        
        # Process with Hybrid Extractor
        logger.info("Processing with Hybrid Extractor...")
        result = extractor.process_image(img)
        
        if result is None or 'strokes' not in result:
            raise HTTPException(
                status_code=500,
                detail="Hybrid extractor failed to process image"
            )
        
        # Convert hybrid strokes to Stroke objects
        all_strokes = []
        for s in result['strokes']:
            stroke = Stroke(
                points=np.array(s['points'], dtype=np.float32),
                color=s['color'],
                thickness=s.get('thickness', 2.0)
            )
            all_strokes.append(stroke)
        
        logger.info(f"Hybrid extractor found {len(all_strokes)} strokes in {result['metadata']['processing_time']:.2f}s")
        
        # Generate both SVG and InkML
        svg_output = strokes_to_svg(
            all_strokes,
            width=result['rectified'].shape[1],
            height=result['rectified'].shape[0],
            background_color="none"  # Transparent background
        )
        
        inkml_output = strokes_to_inkml(
            all_strokes,
            width=result['rectified'].shape[1],
            height=result['rectified'].shape[0]
        )
        
        # Log output info
        logger.info(f"Generated SVG: {len(svg_output)} chars")
        logger.info(f"Generated InkML: {len(inkml_output)} chars")
        logger.info(f"First 500 chars of SVG:\n{svg_output[:500]}")
        
        return JSONResponse(content={
            "success": True,
            "svg": svg_output,
            "inkml": inkml_output,
            "metadata": {
                "strokes_count": len(all_strokes),
                "processing_time": result['metadata']['processing_time'],
                "backend": result['metadata']['backend'],
                "image_size": {
                    "width": result['rectified'].shape[1],
                    "height": result['rectified'].shape[0]
                }
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview-image")
async def preview_image(file: UploadFile = File(...)) -> Dict:
    """
    Preview image processing without full vectorization
    Useful for debugging
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Process with hybrid extractor
        extractor = get_extractor()
        result = extractor.process_image(img)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Processing failed")
        
        # Encode preprocessed image as base64 for preview
        _, buffer = cv2.imencode('.png', result['rectified'])
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "preview": f"data:image/png;base64,{img_base64}",
            "original_size": {"width": img.shape[1], "height": img.shape[0]},
            "processed_size": {"width": result['rectified'].shape[1], "height": result['rectified'].shape[0]},
            "strokes_count": len(result.get('strokes', []))
        }
        
    except Exception as e:
        logger.error(f"Error in preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("[OK] Backend ready! AI models will load on first use.")
    print(f"[OK] Server starting on {HOST}:{PORT}")
    print("="*60 + "\n")
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
