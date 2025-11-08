"""
Simplified OneNote Integration for Public Use
Users authenticate with your centralized Azure app
"""

import requests
import json
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading
import os

logger = logging.getLogger(__name__)

# Load configuration from environment
try:
    from config import ONENOTE_CLIENT_ID, ONENOTE_CLIENT_SECRET, OAUTH_REDIRECT_URI, OAUTH_SCOPES
    CLIENT_ID = ONENOTE_CLIENT_ID
    CLIENT_SECRET = ONENOTE_CLIENT_SECRET
    REDIRECT_URI = OAUTH_REDIRECT_URI
    SCOPES = OAUTH_SCOPES
except ImportError:
    # Fallback for direct execution (development only)
    CLIENT_ID = os.getenv('ONENOTE_CLIENT_ID', '')
    CLIENT_SECRET = os.getenv('ONENOTE_CLIENT_SECRET', '')
    REDIRECT_URI = os.getenv('OAUTH_REDIRECT_URI', 'http://localhost:8888/callback')
    SCOPES = os.getenv('OAUTH_SCOPES', 'Notes.ReadWrite Notes.Create offline_access')

# Validate configuration (warning only, don't fail)
if not CLIENT_ID or not CLIENT_SECRET:
    logger.warning(
        "OneNote credentials not configured. "
        "OneNote features will not work until credentials are set. "
        "Set ONENOTE_CLIENT_ID and ONENOTE_CLIENT_SECRET environment variables."
    )

AUTH_ENDPOINT = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
TOKEN_ENDPOINT = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_API_BETA = "https://graph.microsoft.com/beta"


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP server handler for OAuth callback"""
    auth_code = None
    auth_error = None
    
    def do_GET(self):
        """Handle the OAuth callback"""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        if 'code' in params:
            AuthCallbackHandler.auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
                <html>
                <head>
                    <title>Authentication Successful</title>
                    <style>
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                            text-align: center;
                            padding: 50px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                        }
                        .success-box {
                            background: white;
                            color: #333;
                            padding: 40px;
                            border-radius: 12px;
                            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                            max-width: 400px;
                            margin: 0 auto;
                        }
                        .checkmark {
                            font-size: 64px;
                            color: #4CAF50;
                            margin-bottom: 20px;
                        }
                        h1 { margin: 0 0 10px 0; color: #333; }
                        p { color: #666; margin: 10px 0 20px 0; }
                        .close-btn {
                            background: #667eea;
                            color: white;
                            border: none;
                            padding: 12px 30px;
                            border-radius: 6px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background 0.3s;
                        }
                        .close-btn:hover {
                            background: #764ba2;
                        }
                        .auto-close {
                            font-size: 12px;
                            color: #999;
                            margin-top: 15px;
                        }
                    </style>
                </head>
                <body>
                    <div class="success-box">
                        <div class="checkmark">✓</div>
                        <h1>Connected to OneNote!</h1>
                        <p>You can now return to the app</p>
                        <button class="close-btn" onclick="closeWindow()">Close This Tab</button>
                        <div class="auto-close" id="countdown">This tab will close automatically in 3 seconds...</div>
                    </div>
                    <script>
                        let countdown = 3;
                        const countdownEl = document.getElementById('countdown');
                        
                        function closeWindow() {
                            // Try multiple methods to close the window
                            window.open('', '_self').close();
                            window.close();
                            
                            // If still open after 100ms, show message
                            setTimeout(() => {
                                if (!window.closed) {
                                    countdownEl.textContent = 'Please close this tab manually (Ctrl+W or ⌘+W)';
                                    countdownEl.style.color = '#666';
                                }
                            }, 100);
                        }
                        
                        // Countdown timer
                        const timer = setInterval(() => {
                            countdown--;
                            if (countdown > 0) {
                                countdownEl.textContent = `This tab will close automatically in ${countdown} second${countdown !== 1 ? 's' : ''}...`;
                            } else {
                                clearInterval(timer);
                                closeWindow();
                            }
                        }, 1000);
                        
                        // Try to close immediately in background (silent fail if blocked)
                        setTimeout(() => {
                            window.open('', '_self').close();
                        }, 100);
                    </script>
                </body>
                </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            AuthCallbackHandler.auth_error = params.get('error_description', ['Unknown error'])[0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
                <html>
                <head><title>Error</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: red;">Authentication Failed</h1>
                    <p>{AuthCallbackHandler.auth_error}</p>
                </body>
                </html>
            """
            self.wfile.write(html.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass


class SimpleOneNoteAuth:
    """Simple OneNote authentication - just click a button!"""
    
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.user_info = None
    
    def authenticate(self) -> bool:
        """
        One-click authentication with Microsoft
        Opens browser, user signs in, done!
        
        Returns:
            True if successful
        """
        # DEBUG: Log CLIENT_ID value
        logger.info(f"[DEBUG] CLIENT_ID at authenticate(): '{CLIENT_ID}'")
        logger.info(f"[DEBUG] CLIENT_SECRET at authenticate(): '{CLIENT_SECRET[:10] if CLIENT_SECRET else 'EMPTY'}...'")
        logger.info(f"[DEBUG] REDIRECT_URI at authenticate(): '{REDIRECT_URI}'")
        
        # Build authorization URL
        # Convert scopes to space-separated string if it's a list
        scopes_str = ' '.join(SCOPES) if isinstance(SCOPES, list) else SCOPES
        
        auth_params = {
            'client_id': CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': REDIRECT_URI,
            'scope': scopes_str,
            'response_mode': 'query',
            'prompt': 'consent'  # Force consent screen to show all requested permissions
        }
        auth_url = f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(auth_params)}"
        logger.info(f"[DEBUG] Authorization URL: {auth_url[:100]}...")
        
        # Reset callback handler
        AuthCallbackHandler.auth_code = None
        AuthCallbackHandler.auth_error = None
        
        # Start local server to receive callback
        server = HTTPServer(('localhost', 8888), AuthCallbackHandler)
        server_thread = threading.Thread(target=server.handle_request)
        server_thread.start()
        
        # Open browser for authentication
        print("\n>>> Opening browser for Microsoft login...")
        print("    Please sign in with your Microsoft account")
        webbrowser.open(auth_url)
        
        # Wait for callback
        server_thread.join(timeout=120)  # 2 minute timeout
        
        if AuthCallbackHandler.auth_error:
            logger.error(f"Authentication error: {AuthCallbackHandler.auth_error}")
            return False
        
        if not AuthCallbackHandler.auth_code:
            logger.error("Authentication timed out")
            return False
        
        # Exchange code for access token
        logger.info(f"[DEBUG] CLIENT_ID at token exchange: '{CLIENT_ID}'")
        logger.info(f"[DEBUG] CLIENT_SECRET at token exchange: '{CLIENT_SECRET[:10] if CLIENT_SECRET else 'EMPTY'}...'")
        
        token_data = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code': AuthCallbackHandler.auth_code,
            'redirect_uri': REDIRECT_URI,
            'grant_type': 'authorization_code'
        }
        
        logger.info(f"[DEBUG] Token request data keys: {list(token_data.keys())}")
        logger.info(f"[DEBUG] Token request data: client_id={token_data.get('client_id', 'MISSING')}")
        
        try:
            logger.info("Attempting token exchange with Microsoft...")
            response = requests.post(TOKEN_ENDPOINT, data=token_data, timeout=30)
            response.raise_for_status()
            tokens = response.json()
            
            self.access_token = tokens.get('access_token')
            self.refresh_token = tokens.get('refresh_token')
            logger.info("Token exchange successful")
            
            # Get user info
            self._get_user_info()
            
            user_name = self.user_info.get('displayName', 'User') if self.user_info else 'User'
            logger.info(f"Authenticated as: {user_name}")
            return True
            
        except requests.exceptions.Timeout:
            logger.error("Token exchange timed out - Microsoft servers not responding")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Connection error - Cannot reach Microsoft servers")
            return False
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_user_info(self):
        """Get basic user info"""
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            response = requests.get(f"{GRAPH_API_BASE}/me", headers=headers, timeout=15)
            logger.info(f"User info API status: {response.status_code}")
            if response.status_code == 200:
                self.user_info = response.json()
                logger.info(f"User info retrieved: {self.user_info.get('displayName', 'N/A')}, email: {self.user_info.get('userPrincipalName', 'N/A')}")
            else:
                logger.error(f"User info API failed: {response.text}")
                self.user_info = {}
        except requests.exceptions.Timeout:
            logger.error("User info request timed out")
            self.user_info = {}
        except Exception as e:
            logger.warning(f"Failed to get user info: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.user_info = {}
    
    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token
        
        Returns:
            True if successful, False otherwise
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        # Convert scopes to space-separated string if it's a list
        scopes_str = ' '.join(SCOPES) if isinstance(SCOPES, list) else SCOPES
        
        token_data = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token',
            'scope': scopes_str
        }
        
        try:
            response = requests.post(TOKEN_ENDPOINT, data=token_data)
            response.raise_for_status()
            tokens = response.json()
            
            self.access_token = tokens.get('access_token')
            # Sometimes a new refresh token is provided
            if 'refresh_token' in tokens:
                self.refresh_token = tokens.get('refresh_token')
            
            logger.info("Access token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_user_email(self) -> str:
        """Get user's email address"""
        if not self.user_info:
            logger.info("User info not set, fetching...")
            self._get_user_info()
        
        if not self.user_info:
            logger.error("User info is still empty after fetch!")
            return ''
        
        email = self.user_info.get('userPrincipalName', '') or self.user_info.get('mail', '')
        logger.info(f"Retrieved user email: {email}")
        return email
    
    def get_notebooks(self) -> list:
        """Get user's OneNote notebooks"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/notebooks?$select=id,displayName",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get notebooks: {e}")
            return []
    
    def get_sections(self, notebook_id: str) -> list:
        """Get sections in a notebook"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/notebooks/{notebook_id}/sections?$select=id,displayName",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get sections: {e}")
            return []
    
    def get_section_groups(self, notebook_id: str) -> list:
        """Get section groups in a notebook"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/notebooks/{notebook_id}/sectionGroups?$select=id,displayName",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get section groups: {e}")
            return []
    
    def get_sections_in_group(self, section_group_id: str) -> list:
        """Get sections within a section group"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/sectionGroups/{section_group_id}/sections?$select=id,displayName",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get sections in group: {e}")
            return []
    
    def create_section(self, notebook_id: str, section_name: str, section_group_id: str = None) -> dict:
        """
        Create a new section in a notebook or section group
        
        Args:
            notebook_id: The notebook ID
            section_name: Name for the new section
            section_group_id: Optional section group ID to create section within a group
            
        Returns:
            Dict with section_id and section_name
        """
        if not self.access_token:
            return None
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            # Determine the endpoint based on whether it's in a section group or notebook
            if section_group_id:
                url = f"{GRAPH_API_BASE}/me/onenote/sectionGroups/{section_group_id}/sections"
            else:
                url = f"{GRAPH_API_BASE}/me/onenote/notebooks/{notebook_id}/sections"
            
            payload = {
                'displayName': section_name
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Successfully created section: {section_name}")
            
            return {
                'section_id': result['id'],
                'section_name': result.get('displayName', section_name)
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error creating section: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            return None
        except Exception as e:
            logger.error(f"Failed to create section: {e}")
            logger.error(f"URL: {url}")
            logger.error(f"Payload: {payload}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_nested_section_groups(self, section_group_id: str) -> list:
        """Get nested section groups within a section group"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/sectionGroups/{section_group_id}/sectionGroups?$select=id,displayName",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get nested section groups: {e}")
            return []
    
    def get_pages_in_section(self, section_id: str) -> list:
        """Get pages within a section"""
        if not self.access_token:
            return []
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me/onenote/sections/{section_id}/pages?$select=id,title",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('value', [])
        except Exception as e:
            logger.error(f"Failed to get pages: {e}")
            return []
    
    def fetch_all_onenote_data(self) -> dict:
        """
        Recursively fetch ALL OneNote data and build a complete hierarchy
        Returns a structured dict with notebooks, section groups, sections, and pages
        """
        if not self.access_token:
            return {}
        
        logger.info("Starting complete OneNote data fetch...")
        
        def fetch_section_group_hierarchy(section_group_id: str, parent_path: str = "") -> dict:
            """Recursively fetch section group contents"""
            path = f"{parent_path}/SectionGroup"
            logger.info(f"Fetching section group {section_group_id} at {path}")
            
            # Get sections in this group
            sections = self.get_sections_in_group(section_group_id)
            sections_data = []
            for section in sections:
                section_path = f"{path}/{section.get('displayName', 'Unnamed')}"
                logger.info(f"  Fetching pages for section: {section_path}")
                pages = self.get_pages_in_section(section['id'])
                sections_data.append({
                    'id': section['id'],
                    'name': section.get('displayName', section.get('name', 'Unnamed Section')),
                    'pages': [{'id': p['id'], 'name': p.get('title', 'Untitled')} for p in pages]
                })
            
            # Get nested section groups
            nested_groups = self.get_nested_section_groups(section_group_id)
            nested_groups_data = []
            for group in nested_groups:
                group_data = fetch_section_group_hierarchy(group['id'], path)
                group_data['id'] = group['id']
                group_data['name'] = group.get('displayName', group.get('name', 'Unnamed Group'))
                nested_groups_data.append(group_data)
            
            return {
                'sections': sections_data,
                'section_groups': nested_groups_data
            }
        
        # Start with notebooks
        notebooks = self.get_notebooks()
        notebooks_data = []
        
        for nb in notebooks:
            notebook_name = nb.get('displayName', nb.get('name', 'Unnamed Notebook'))
            logger.info(f"Fetching notebook: {notebook_name}")
            
            # Get root-level sections
            sections = self.get_sections(nb['id'])
            sections_data = []
            for section in sections:
                section_path = f"{notebook_name}/{section.get('displayName', 'Unnamed')}"
                logger.info(f"  Fetching pages for section: {section_path}")
                pages = self.get_pages_in_section(section['id'])
                sections_data.append({
                    'id': section['id'],
                    'name': section.get('displayName', section.get('name', 'Unnamed Section')),
                    'pages': [{'id': p['id'], 'name': p.get('title', 'Untitled')} for p in pages]
                })
            
            # Get root-level section groups
            section_groups = self.get_section_groups(nb['id'])
            section_groups_data = []
            for group in section_groups:
                group_path = f"{notebook_name}/{group.get('displayName', 'Unnamed')}"
                logger.info(f"  Fetching section group: {group_path}")
                group_hierarchy = fetch_section_group_hierarchy(group['id'], group_path)
                group_hierarchy['id'] = group['id']
                group_hierarchy['name'] = group.get('displayName', group.get('name', 'Unnamed Group'))
                section_groups_data.append(group_hierarchy)
            
            notebooks_data.append({
                'id': nb['id'],
                'name': notebook_name,
                'sections': sections_data,
                'section_groups': section_groups_data
            })
        
        logger.info(f"Complete data fetch finished. {len(notebooks_data)} notebooks loaded.")
        return {'notebooks': notebooks_data}
    
    def create_page_with_image(self, section_id: str, image_data: bytes, page_title: str = None) -> dict:
        """
        Create a OneNote page with an image
        
        Args:
            section_id: Target section ID
            image_data: PNG image bytes
            page_title: Optional page title (default: timestamp)
        
        Returns:
            Page info dict with page_id and page_url
        """
        if not self.access_token:
            return None
        
        from datetime import datetime
        if not page_title:
            page_title = f"Whiteboard Scan - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
        
        # Create HTML content for the page
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{page_title}</title>
        </head>
        <body>
            <h1>{page_title}</h1>
            <img src="image1" alt="Scanned Whiteboard" />
        </body>
        </html>
        """
        
        # Prepare multipart request
        boundary = "----Boundary" + str(id(image_data))
        
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="Presentation"\r\n'
            f'Content-Type: text/html\r\n\r\n'
            f'{html_content}\r\n'
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="image1"\r\n'
            f'Content-Type: image/png\r\n\r\n'
        ).encode('utf-8')
        
        body += image_data
        body += f'\r\n--{boundary}--\r\n'.encode('utf-8')
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
        
        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/me/onenote/sections/{section_id}/pages",
                headers=headers,
                data=body
            )
            
            # If 401 Unauthorized, try to refresh token and retry once
            if response.status_code == 401:
                logger.info("Access token expired, refreshing...")
                if self.refresh_access_token():
                    # Retry with new token
                    headers['Authorization'] = f'Bearer {self.access_token}'
                    response = requests.post(
                        f"{GRAPH_API_BASE}/me/onenote/sections/{section_id}/pages",
                        headers=headers,
                        data=body
                    )
            
            response.raise_for_status()
            page_data = response.json()
            
            return {
                'success': True,
                'page_id': page_data.get('id'),
                'page_url': page_data.get('links', {}).get('oneNoteWebUrl', {}).get('href'),
                'title': page_title
            }
            
        except Exception as e:
            logger.error(f"Failed to create page: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_page_with_inkml(self, section_id: str, inkml_data: str, page_title: str = None) -> dict:
        """
        Create a OneNote page with InkML (editable ink strokes)
        
        Args:
            section_id: Target section ID
            inkml_data: InkML XML string
            page_title: Optional page title (default: timestamp)
        
        Returns:
            Page info dict with page_id and page_url
        """
        if not self.access_token:
            return None
        
        from datetime import datetime
        if not page_title:
            page_title = f"Whiteboard Scan - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
        
        # Create HTML content with embedded InkML
        # Use <img> tag with src="name:inkml1" to reference the InkML part
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{page_title}</title>
        </head>
        <body>
            <h1>{page_title}</h1>
            <img src="name:inkml1" alt="Whiteboard Ink" />
        </body>
        </html>
        """
        
        # Prepare multipart request
        boundary = "----Boundary" + str(id(inkml_data))
        
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="Presentation"\r\n'
            f'Content-Type: text/html\r\n\r\n'
            f'{html_content}\r\n'
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="inkml1"\r\n'
            f'Content-Type: application/inkml+xml\r\n\r\n'
            f'{inkml_data}\r\n'
            f'--{boundary}--\r\n'
        ).encode('utf-8')
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
        
        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/me/onenote/sections/{section_id}/pages",
                headers=headers,
                data=body
            )
            response.raise_for_status()
            page_data = response.json()
            
            return {
                'success': True,
                'page_id': page_data.get('id'),
                'page_url': page_data.get('links', {}).get('oneNoteWebUrl', {}).get('href'),
                'title': page_title
            }
            
        except Exception as e:
            logger.error(f"Failed to create page with InkML: {e}")
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
            return {'success': False, 'error': str(e)}
    
    def create_page_with_ink_strokes(self, section_id: str, strokes_data: list, page_title: str = None) -> dict:
        """
        Create a OneNote page with editable ink strokes using Microsoft's InkML format
        
        This follows the official Microsoft Graph InkML beta API specification.
        Reference: https://learn.microsoft.com/en-us/previous-versions/office/office-365-api/
        
        Args:
            section_id: Target section ID
            strokes_data: List of stroke dicts with 'id' and 'points' [[x,y],...]
            page_title: Optional page title (default: timestamp)
        
        Returns:
            Page info dict with page_id and page_url
        """
        if not self.access_token:
            return None
        
        from datetime import datetime
        if not page_title:
            page_title = f"Whiteboard Scan - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
        
        # Build InkML following Microsoft's spec
        # Convert pixels to himetric (1 himetric = 1/1000 cm)
        # Calculation: 96 DPI = 96 pixels/inch = 96 pixels / 2.54cm = 37.8 pixels/cm
        # So: 1 pixel = 1000/37.8 = 26.5 himetric
        PIXELS_TO_HIMETRIC = 26.5
        
        inkml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<inkml:ink xmlns:inkml="http://www.w3.org/2003/InkML" xmlns:msink="http://schemas.microsoft.com/ink/2010/main">',
            ' <inkml:definitions>',
            '  <inkml:context xml:id="ctxCoordinatesWithPressure">',
            '   <inkml:inkSource xml:id="inkSrcCoordinatesWithPressure">',
            '    <inkml:traceFormat>',
            '     <inkml:channel name="X" type="integer" max="32767" units="himetric"/>',
            '     <inkml:channel name="Y" type="integer" max="32767" units="himetric"/>',
            '     <inkml:channel name="F" type="integer" max="32767" units="dev"/>',
            '    </inkml:traceFormat>',
            '    <inkml:channelProperties>',
            '     <inkml:channelProperty channel="X" name="resolution" value="1" units="1/himetric"/>',
            '     <inkml:channelProperty channel="Y" name="resolution" value="1" units="1/himetric"/>',
            '     <inkml:channelProperty channel="F" name="resolution" value="1" units="1/dev"/>',
            '    </inkml:channelProperties>',
            '   </inkml:inkSource>',
            '  </inkml:context>',
            '  <inkml:brush xml:id="br0">',
            '   <inkml:brushProperty name="width" value="100" units="himetric"/>',
            '   <inkml:brushProperty name="height" value="100" units="himetric"/>',
            '   <inkml:brushProperty name="color" value="#000000"/>',
            '   <inkml:brushProperty name="transparency" value="0"/>',
            '   <inkml:brushProperty name="tip" value="ellipse"/>',
            '   <inkml:brushProperty name="rasterOp" value="copyPen"/>',
            '   <inkml:brushProperty name="ignorePressure" value="false"/>',
            '   <inkml:brushProperty name="antiAliased" value="true"/>',
            '   <inkml:brushProperty name="fitToCurve" value="false"/>',
            '  </inkml:brush>',
            ' </inkml:definitions>',
            ' <inkml:traceGroup>'
        ]
        
        # Add each stroke as a trace
        for idx, stroke in enumerate(strokes_data):
            stroke_id = stroke.get('id', f'st{idx}')
            points = stroke.get('points', [])
            
            # Convert points to himetric with pressure
            # Format: "X1 Y1 F1, X2 Y2 F2, ..."
            point_strs = []
            for pt in points:
                x_himetric = int(pt[0] * PIXELS_TO_HIMETRIC)
                y_himetric = int(pt[1] * PIXELS_TO_HIMETRIC)
                pressure = 7168  # Mid-range pressure
                point_strs.append(f'{x_himetric} {y_himetric} {pressure}')
            
            points_text = ', '.join(point_strs)
            inkml_lines.append(f'  <inkml:trace xml:id="{stroke_id}" contextRef="#ctxCoordinatesWithPressure" brushRef="#br0">{points_text}</inkml:trace>')
        
        inkml_lines.extend([
            ' </inkml:traceGroup>',
            '</inkml:ink>'
        ])
        
        inkml_data = '\n'.join(inkml_lines)
        
        # Create HTML
        html_content = f"""<html>
<head>
<title>{page_title}</title>
</head>
<body>
<h1>{page_title}</h1>
</body>
</html>"""
        
        # Create multipart form with HTML and InkML
        # CRITICAL: Content-Disposition name must be "presentation-onenote-inkml"
        boundary = f"----Boundary{id(strokes_data)}"
        body = (
            f'--{boundary}\r\n'
            f'Content-Type: text/html\r\n'
            f'Content-Disposition: form-data; name="Presentation"\r\n\r\n'
            f'{html_content}\r\n'
            f'--{boundary}\r\n'
            f'Content-Type: application/inkml+xml\r\n'
            f'Content-Disposition: form-data; name="presentation-onenote-inkml"\r\n\r\n'
            f'{inkml_data}\r\n'
            f'--{boundary}--\r\n'
        ).encode('utf-8')
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
        
        try:
            # Use beta endpoint for ink support and specify the section
            response = requests.post(
                f"{GRAPH_API_BETA}/me/onenote/sections/{section_id}/pages",
                headers=headers,
                data=body
            )
            
            # If 401 Unauthorized, try to refresh token and retry once
            if response.status_code == 401:
                logger.info("Access token expired, refreshing...")
                if self.refresh_access_token():
                    # Retry with new token
                    headers['Authorization'] = f'Bearer {self.access_token}'
                    response = requests.post(
                        f"{GRAPH_API_BETA}/me/onenote/sections/{section_id}/pages",
                        headers=headers,
                        data=body
                    )
            
            # Log the response for debugging
            logger.info(f"OneNote API response status: {response.status_code}")
            if response.status_code != 201:
                logger.error(f"OneNote API error response: {response.text}")
            
            response.raise_for_status()
            page_data = response.json()
            
            logger.info(f"Created OneNote page with {len(strokes_data)} editable ink strokes")
            logger.info(f"Page URL: {page_data.get('links', {}).get('oneNoteWebUrl', {}).get('href', 'N/A')}")
            
            return {
                'success': True,
                'page_id': page_data.get('id'),
                'page_url': page_data.get('links', {}).get('oneNoteWebUrl', {}).get('href'),
                'title': page_title,
                'strokes_count': len(strokes_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to create page with ink strokes: {e}")
            if 'response' in locals():
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response body: {response.text}")
            return {'success': False, 'error': str(e)}


# Simple command-line setup for users
def setup_simple():
    """Simple setup wizard for end users"""
    print("=" * 70)
    print(" " * 20 + "OneNote Connection Setup")
    print("=" * 70 + "\n")
    
    auth = SimpleOneNoteAuth()
    
    print("Step 1: Sign in to Microsoft")
    print("-" * 70)
    if not auth.authenticate():
        print("\n>>> Sign-in failed. Please try again.")
        return
    
    user_name = auth.user_info.get('displayName', 'User') if auth.user_info else 'User'
    print(f"\n>>> Signed in as: {user_name}")
    
    print("\nStep 2: Select your OneNote notebook")
    print("─" * 70)
    notebooks = auth.get_notebooks()
    
    if not notebooks:
        print(">>> No notebooks found")
        return
    
    print(f"\nFound {len(notebooks)} notebooks:\n")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {nb['displayName']}")
    
    choice = input(f"\nSelect notebook (1-{len(notebooks)}): ").strip()
    try:
        notebook = notebooks[int(choice) - 1]
    except:
        print(">>> Invalid selection")
        return
    
    print(f"\nStep 3: Select section in '{notebook['displayName']}'")
    print("─" * 70)
    sections = auth.get_sections(notebook['id'])
    
    if not sections:
        print(">>> No sections found")
        return
    
    print(f"\nFound {len(sections)} sections:\n")
    for i, sec in enumerate(sections, 1):
        print(f"  {i}. {sec['displayName']}")
    
    choice = input(f"\nSelect section (1-{len(sections)}): ").strip()
    try:
        section = sections[int(choice) - 1]
    except:
        print(">>> Invalid selection")
        return
    
    # Save configuration
    user_name = auth.user_info.get('displayName', 'User') if auth.user_info else 'User'
    user_email = auth.get_user_email()
    config = {
        'access_token': auth.access_token,
        'refresh_token': auth.refresh_token,
        'notebook_id': notebook['id'],
        'notebook_name': notebook['displayName'],
        'section_id': section['id'],
        'section_name': section['displayName'],
        'user_name': user_name,
        'user_email': user_email
    }
    
    with open('user_onenote_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print(">>> Setup Complete!")
    print("=" * 70)
    print(f"Scans will be sent to:")
    print(f"  [Notebook] {notebook['displayName']}")
    print(f"  [Section] {section['displayName']}")
    print("\nConfiguration saved. You're ready to scan whiteboards!")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_simple()
