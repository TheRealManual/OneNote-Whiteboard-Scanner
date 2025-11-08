import React, { useState, useRef, useEffect } from 'react';
import './styles.css';

const BACKEND_URL = 'http://127.0.0.1:5000';

function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({ message: '', type: 'info' });
  const [oneNoteAvailable, setOneNoteAvailable] = useState(false);
  const [oneNoteConnected, setOneNoteConnected] = useState(false);
  const [notebooks, setNotebooks] = useState([]);
  const [selectedNotebook, setSelectedNotebook] = useState(null);
  const [sections, setSections] = useState([]);
  const [sectionGroups, setSectionGroups] = useState([]);
  const [currentSectionGroup, setCurrentSectionGroup] = useState(null); // Track which group we're viewing
  const [selectedSection, setSelectedSection] = useState(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [hierarchyCache, setHierarchyCache] = useState(null); // Complete OneNote hierarchy cache
  const [cacheLoading, setCacheLoading] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  console.log('App render - cameraActive:', cameraActive, 'processing:', processing, 'result:', result);

  // Check backend health on startup (with retries)
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 60; // Try for 60 seconds
    const retryDelay = 1000; // 1 second between retries
    let timeoutId = null;
    
    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`, {
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✓ Backend connected successfully!');
        setBackendReady(true);
        setBackendError(null);
        setOneNoteAvailable(data.onenote_available || false);
        
        // Hide the initial HTML loader now that backend is ready
        const initialLoader = document.getElementById('initial-loader');
        if (initialLoader) {
          initialLoader.style.display = 'none';
        }
        
        // Auto-load OneNote config if user was previously connected
        loadOneNoteConfig();
      } catch (error) {
        console.log(`Backend check ${retryCount + 1}/${maxRetries}: ${error.message}`);
        
        retryCount++;
        if (retryCount < maxRetries) {
          // Keep trying
          timeoutId = setTimeout(checkBackend, retryDelay);
        } else {
          setBackendError('Backend failed to start after 60 seconds. Please restart the application.');
          setBackendReady(false);
          
          // Hide initial loader and show error in React overlay
          const initialLoader = document.getElementById('initial-loader');
          if (initialLoader) {
            initialLoader.style.display = 'none';
          }
        }
      }
    };
    
    checkBackend();
    
    // Cleanup timeout on unmount
    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  // Load OneNote configuration (auto-login on startup if previously authenticated)
  const loadOneNoteConfig = async () => {
    try {
      console.log('Checking for existing OneNote authentication...');
      const response = await fetch(`${BACKEND_URL}/onenote/config`);
      if (response.ok) {
        const config = await response.json();
        
        // Check if user has valid auth tokens (email is optional - may not have User.Read permission)
        if (config.has_auth) {
          const displayEmail = config.user_email || 'Microsoft Account';
          console.log('Auto-logging in user:', displayEmail);
          setOneNoteConnected(true);
          setUserEmail(displayEmail);
          
          // Load cached hierarchy if available
          if (config.full_hierarchy_cache && config.full_hierarchy_cache.notebooks) {
            console.log('📦 Loading cached OneNote hierarchy...');
            setHierarchyCache(config.full_hierarchy_cache);
            const cachedNotebooks = config.full_hierarchy_cache.notebooks;
            setNotebooks(cachedNotebooks.map(nb => ({ id: nb.id, name: nb.name })));
            console.log('✅ Loaded complete hierarchy with', cachedNotebooks.length, 'notebooks from cache');
            
            // Debug: Show cache structure
            console.log('🔍 Cache structure sample:');
            if (cachedNotebooks[0]) {
              console.log('  First notebook:', cachedNotebooks[0].name);
              console.log('  Sections:', cachedNotebooks[0].sections?.length || 0);
              console.log('  Section groups:', cachedNotebooks[0].section_groups?.length || 0);
              if (cachedNotebooks[0].section_groups?.[0]) {
                console.log('  First group:', cachedNotebooks[0].section_groups[0].name);
                console.log('    Has sections:', cachedNotebooks[0].section_groups[0].sections?.length || 0);
                console.log('    Has nested groups:', cachedNotebooks[0].section_groups[0].section_groups?.length || 0);
              }
            }
          } else if (config.notebooks_cache && config.notebooks_cache.length > 0) {
            // Fallback to old notebook-only cache
            setNotebooks(config.notebooks_cache);
            console.log('Loaded', config.notebooks_cache.length, 'cached notebooks (no full hierarchy)');
            
            // Trigger background fetch of complete hierarchy
            fetchCompleteHierarchy();
          } else {
            // No cache at all - fetch everything
            console.log('No cache found - fetching complete hierarchy...');
            fetchCompleteHierarchy();
          }
          
          // Restore selected notebook/section if exists
          if (config.notebook_id && config.section_id) {
            setSelectedNotebook({ id: config.notebook_id, name: config.notebook_name });
            setSelectedSection({ id: config.section_id, name: config.section_name });
            
            // Load sections from cache if available, otherwise fetch
            if (config.full_hierarchy_cache && config.full_hierarchy_cache.notebooks) {
              const notebook = config.full_hierarchy_cache.notebooks.find(nb => nb.id === config.notebook_id);
              if (notebook) {
                setSections(notebook.sections || []);
                setSectionGroups(notebook.section_groups || []);
                console.log('Loaded sections and groups from cache for restored notebook');
              }
            } else {
              // Fallback to API fetch
              try {
                const sectionsResponse = await fetch(`${BACKEND_URL}/onenote/sections?notebook_id=${config.notebook_id}`);
                if (sectionsResponse.ok) {
                  const sectionsData = await sectionsResponse.json();
                  setSections(sectionsData.sections || []);
                  setSectionGroups(sectionsData.section_groups || []);
                  console.log('Loaded', sectionsData.sections?.length || 0, 'sections for restored notebook');
                }
              } catch (err) {
                console.warn('Could not load sections for restored notebook:', err);
              }
            }
            
            console.log('Restored destination:', config.notebook_name, '→', config.section_name);
            setStatus({ 
              message: `Welcome back! Ready to scan to: ${config.notebook_name} → ${config.section_name}`, 
              type: 'success' 
            });
          } else {
            // User is logged in but hasn't selected destination yet
            setShowSidebar(true);
            setStatus({ 
              message: 'Please select a notebook and section to continue', 
              type: 'info' 
            });
          }
        } else {
          console.log('No saved authentication found');
        }
      }
    } catch (error) {
      // Config doesn't exist yet - user needs to login
      console.log('No existing OneNote config - user must sign in');
    }
  };

  // Fetch complete OneNote hierarchy and cache it
  const fetchCompleteHierarchy = async () => {
    if (cacheLoading) return; // Prevent duplicate requests
    
    try {
      setCacheLoading(true);
      console.log('🔄 Fetching complete OneNote hierarchy...');
      setStatus({ message: 'Loading OneNote data...', type: 'info' });
      
      const response = await fetch(`${BACKEND_URL}/onenote/fetch-all`);
      if (response.ok) {
        const data = await response.json();
        setHierarchyCache(data);
        
        // Update notebooks list
        if (data.notebooks) {
          setNotebooks(data.notebooks.map(nb => ({ id: nb.id, name: nb.name })));
          console.log('✅ Complete hierarchy cached:', data.notebooks.length, 'notebooks');
        }
        
        setStatus({ message: 'OneNote data loaded!', type: 'success' });
      } else {
        console.warn('Failed to fetch complete hierarchy');
      }
    } catch (error) {
      console.error('Error fetching complete hierarchy:', error);
    } finally {
      setCacheLoading(false);
    }
  };

  // Connect to OneNote
  const connectToOneNote = async () => {
    if (!backendReady) {
      setStatus({ message: 'Backend is still starting up...', type: 'error' });
      return;
    }
    
    try {
      setStatus({ message: 'Opening OneNote authentication...', type: 'info' });
      
      // Trigger backend setup
      const response = await fetch(`${BACKEND_URL}/onenote/setup`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        setNotebooks(data.notebooks || []);
        setUserEmail(data.user_email || '');
        setOneNoteConnected(true);
        setShowSidebar(true); // Open sidebar to select notebook/section
        setStatus({ message: 'Connected! Loading all your OneNote data...', type: 'info' });
        
        // Immediately fetch complete hierarchy after login
        fetchCompleteHierarchy();
      } else {
        throw new Error('Authentication failed');
      }
    } catch (error) {
      console.error('OneNote connection error:', error);
      setStatus({ message: 'OneNote connection failed. Please try again.', type: 'error' });
    }
  };

  // Logout from OneNote
  const logoutFromOneNote = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/onenote/logout`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Clear all local state including hierarchy cache
        setOneNoteConnected(false);
        setUserEmail('');
        setNotebooks([]);
        setSelectedNotebook(null);
        setSections([]);
        setSectionGroups([]);
        setCurrentSectionGroup(null);
        setSelectedSection(null);
        setShowSidebar(false);
        setHierarchyCache(null); // Clear cached hierarchy
        
        // Open Microsoft logout URL to clear browser session and cached consent
        if (data.logout_url) {
          window.open(data.logout_url, '_blank');
          console.log('Opened Microsoft logout page to clear session');
        }
        
        setStatus({ message: 'Logged out - all data cleared', type: 'success' });
      }
    } catch (error) {
      console.error('Logout error:', error);
      setStatus({ message: 'Logout failed', type: 'error' });
    }
  };

  // Fetch sections when notebook is selected
  const selectNotebook = async (notebook) => {
    setSelectedNotebook(notebook);
    setSelectedSection(null); // Reset section selection
    setSectionGroups([]);
    setCurrentSectionGroup(null); // Reset to root level
    
    // Try to load from cache first for instant response
    if (hierarchyCache && hierarchyCache.notebooks) {
      const cachedNotebook = hierarchyCache.notebooks.find(nb => nb.id === notebook.id);
      if (cachedNotebook) {
        console.log('📦 Loading notebook data from cache (instant)');
        setSections(cachedNotebook.sections || []);
        setSectionGroups(cachedNotebook.section_groups || []);
        
        // Save notebook selection
        await fetch(`${BACKEND_URL}/onenote/save-selection`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            notebook_id: notebook.id,
            notebook_name: notebook.name
          })
        });
        
        return; // Don't fetch from API if cache hit
      }
    }
    
    // Fallback to API if not in cache
    try {
      console.log('⚠️ Cache miss - fetching from API');
      const response = await fetch(`${BACKEND_URL}/onenote/sections?notebook_id=${notebook.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch sections');
      }
      const data = await response.json();
      console.log('Sections data:', data.sections);
      console.log('Section groups data:', data.section_groups);
      setSections(data.sections || []);
      setSectionGroups(data.section_groups || []);
      
      // Save notebook selection
      await fetch(`${BACKEND_URL}/onenote/save-selection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          notebook_id: notebook.id,
          notebook_name: notebook.name
        })
      });
    } catch (error) {
      console.error('Failed to load sections:', error);
      setStatus({ message: 'Failed to load sections. Please reconnect to OneNote.', type: 'error' });
    }
  };

  // Navigate into a section group (drill down)
  const openSectionGroup = async (sectionGroup) => {
    console.log('🔍 Opening section group:', sectionGroup.id, sectionGroup.name);
    
    // Helper function to find section group in cache hierarchy
    const findSectionGroupInCache = (groups, targetId) => {
      if (!groups) return null;
      for (const group of groups) {
        console.log('  Checking group:', group.id, group.name);
        if (group.id === targetId) {
          console.log('  ✅ FOUND!');
          return group;
        }
        if (group.section_groups && group.section_groups.length > 0) {
          const found = findSectionGroupInCache(group.section_groups, targetId);
          if (found) return found;
        }
      }
      return null;
    };
    
    // Try cache first
    if (hierarchyCache && hierarchyCache.notebooks) {
      console.log('🔍 Searching in cache...');
      const cachedNotebook = hierarchyCache.notebooks.find(nb => nb.id === selectedNotebook.id);
      if (cachedNotebook) {
        console.log('📖 Found notebook in cache:', cachedNotebook.name);
        console.log('  Section groups in notebook:', cachedNotebook.section_groups?.length || 0);
        
        if (cachedNotebook.section_groups) {
          const cachedGroup = findSectionGroupInCache(cachedNotebook.section_groups, sectionGroup.id);
          if (cachedGroup) {
            console.log('📦 Loading section group from cache (instant)');
            console.log('  Sections:', cachedGroup.sections?.length || 0);
            console.log('  Nested groups:', cachedGroup.section_groups?.length || 0);
            setSections(cachedGroup.sections || []);
            setSectionGroups(cachedGroup.section_groups || []);
            setCurrentSectionGroup(sectionGroup);
            return; // Cache hit - no API call needed
          } else {
            console.log('❌ Section group not found in cache');
          }
        }
      } else {
        console.log('❌ Notebook not found in cache');
      }
    } else {
      console.log('❌ No hierarchy cache available');
    }
    
    // Fallback to API
    console.log('⚠️ Section group cache miss - fetching from API');
    try {
      const response = await fetch(`${BACKEND_URL}/onenote/sections?notebook_id=${selectedNotebook.id}&section_group_id=${sectionGroup.id}`);
      if (response.ok) {
        const data = await response.json();
        console.log('API returned:', data.sections?.length || 0, 'sections,', data.section_groups?.length || 0, 'groups');
        setSections(data.sections || []);
        setSectionGroups(data.section_groups || []);
        setCurrentSectionGroup(sectionGroup);
      }
    } catch (error) {
      console.error('Failed to load section group contents:', error);
      setStatus({ message: 'Failed to open section group', type: 'error' });
    }
  };

  // Go back to parent level
  const goBack = async () => {
    // Go back to notebook root
    setCurrentSectionGroup(null);
    
    // Use cache if available
    if (hierarchyCache && hierarchyCache.notebooks) {
      const cachedNotebook = hierarchyCache.notebooks.find(nb => nb.id === selectedNotebook.id);
      if (cachedNotebook) {
        console.log('📦 Navigating back using cache');
        setSections(cachedNotebook.sections || []);
        setSectionGroups(cachedNotebook.section_groups || []);
        return;
      }
    }
    
    // Fallback to API
    try {
      const response = await fetch(`${BACKEND_URL}/onenote/sections?notebook_id=${selectedNotebook.id}`);
      if (response.ok) {
        const data = await response.json();
        setSections(data.sections || []);
        setSectionGroups(data.section_groups || []);
      }
    } catch (error) {
      console.error('Failed to navigate back:', error);
    }
  };

  // Select a section
  const selectSection = async (section) => {
    setSelectedSection(section);
    setStatus({ 
      message: `Selected: ${selectedNotebook.name} → ${section.name}`, 
      type: 'success' 
    });
    
    // Save section selection
    try {
      await fetch(`${BACKEND_URL}/onenote/save-selection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          section_id: section.id,
          section_name: section.name
        })
      });
    } catch (error) {
      console.error('Failed to save selection:', error);
    }
    
    // Close sidebar after selecting section
    setShowSidebar(false);
  };

  // Create new section
  const createNewSection = async (e) => {
    e.stopPropagation(); // Prevent any parent click handlers
    console.log('🆕 Create new section clicked');
    
    if (!selectedNotebook) {
      setStatus({ message: 'Please select a notebook first', type: 'error' });
      return;
    }

    // Close sidebar immediately to prevent multiple clicks
    setShowSidebar(false);

    try {
      // Generate section name with current date and time for uniqueness
      const now = new Date();
      const dateStr = now.toLocaleDateString('en-US', { 
        month: '2-digit', 
        day: '2-digit', 
        year: 'numeric' 
      }).replace(/\//g, '-');
      
      const timeStr = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      }).replace(/:/g, '-');
      
      // Use timestamp to ensure uniqueness
      const timestamp = Date.now();
      const sectionName = `${dateStr} ${timeStr}`;

      setStatus({ message: `Creating section "${sectionName}"...`, type: 'info' });

      // Determine parent ID (section group or notebook)
      const parentId = currentSectionGroup ? currentSectionGroup.id : selectedNotebook.id;
      const isInGroup = !!currentSectionGroup;

      // Call backend to create section
      const response = await fetch(`${BACKEND_URL}/onenote/create-section`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          notebook_id: selectedNotebook.id,
          section_group_id: isInGroup ? parentId : null,
          section_name: sectionName
        })
      });

      if (response.ok) {
        const data = await response.json();
        const newSection = { id: data.section_id, name: data.section_name };
        
        // Add to local sections list
        setSections([...sections, newSection]);
        
        // Auto-select the new section
        selectSection(newSection);
        
        setStatus({ message: `Section "${sectionName}" created!`, type: 'success' });
      } else {
        throw new Error('Failed to create section');
      }
    } catch (error) {
      console.error('Failed to create section:', error);
      setStatus({ message: 'Failed to create section. Please try again.', type: 'error' });
    }
  };

  // Start camera
  const startCamera = async () => {
    console.log('startCamera function called');
    setCameraLoading(true);
    try {
      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'environment' }
      });
      
      console.log('Camera stream obtained:', stream);
      streamRef.current = stream;
      setCameraActive(true);
      setCameraLoading(false);
      setStatus({ message: 'Camera ready! Position your whiteboard and click Capture.', type: 'success' });
      console.log('Camera activated, state updated');
    } catch (error) {
      console.error('Camera error:', error);
      setCameraLoading(false);
      setStatus({ message: `Camera error: ${error.message}`, type: 'error' });
    }
  };

  // Effect to attach stream to video element when camera becomes active
  useEffect(() => {
    if (cameraActive && streamRef.current && videoRef.current) {
      console.log('Attaching stream to video element');
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().then(() => {
        console.log('Video playing successfully');
      }).catch(err => {
        console.error('Error playing video:', err);
      });
    }
  }, [cameraActive]);

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setCameraActive(false);
      setStatus({ message: 'Camera stopped.', type: 'info' });
    }
  };

  // Capture photo
  const capturePhoto = () => {
    console.log('capturePhoto called');
    if (!videoRef.current || !canvasRef.current) {
      console.error('Video or canvas ref is null');
      setStatus({ message: 'Camera not ready', type: 'error' });
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
    
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.error('Video has no dimensions yet');
      setStatus({ message: 'Please wait for camera to initialize', type: 'error' });
      return;
    }

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    console.log('Photo captured, canvas size:', canvas.width, 'x', canvas.height);
    
    // Stop camera after capture
    stopCamera();
    setStatus({ message: 'Photo captured! Processing...', type: 'info' });
    processImage();
  };

  // Process image with AI
  const processImage = async () => {
    console.log('processImage called');
    if (!canvasRef.current) {
      console.error('Canvas ref is null');
      return;
    }

    setProcessing(true);
    setResult(null);

    try {
      console.log('Creating blob from canvas...');
      const blob = await new Promise(resolve => canvasRef.current.toBlob(resolve, 'image/jpeg', 0.95));
      console.log('Blob created, size:', blob.size, 'bytes');
      
      const formData = new FormData();
      formData.append('file', blob, 'whiteboard.jpg');

      console.log('Sending request to backend...');
      const response = await fetch(`${BACKEND_URL}/process-image`, {
        method: 'POST',
        body: formData
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error response:', errorText);
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Processing complete, result:', data);
      
      // Create preview image from SVG
      if (data.svg) {
        const svgBlob = new Blob([data.svg], { type: 'image/svg+xml' });
        const svgUrl = URL.createObjectURL(svgBlob);
        data.previewUrl = svgUrl;
      }
      
      setResult(data);
      setStatus({ 
        message: 'Processing complete! Send to OneNote or save locally.', 
        type: 'success' 
      });
    } catch (error) {
      console.error('Processing error:', error);
      setStatus({ message: `Processing failed: ${error.message}`, type: 'error' });
    } finally {
      setProcessing(false);
    }
  };

  // Copy to clipboard
  const copyToClipboard = async () => {
    if (!result || !result.svg) {
      setStatus({ message: 'No result to copy', type: 'error' });
      return;
    }

    try {
      console.log('Converting SVG to PNG for clipboard...');
      
      // Create image from SVG
      const img = new Image();
      const svgBlob = new Blob([result.svg], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      img.onload = async () => {
        console.log('SVG loaded, creating canvas...');
        const canvas = document.createElement('canvas');
        
        // Use metadata dimensions or default
        const width = result.metadata?.image_size?.width || 1280;
        const height = result.metadata?.image_size?.height || 720;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        
        const dataUrl = canvas.toDataURL('image/png');
        console.log('PNG created, copying to clipboard...');
        
        // Use Electron API to copy to clipboard
        if (window.electronAPI) {
          const clipboardResult = await window.electronAPI.copyToClipboard(dataUrl);
          console.log('Clipboard result:', clipboardResult);
          setStatus({ message: clipboardResult.message, type: clipboardResult.success ? 'success' : 'error' });
        } else {
          setStatus({ message: 'Clipboard API not available', type: 'error' });
        }
        
        URL.revokeObjectURL(url);
      };
      
      img.onerror = (err) => {
        console.error('Error loading SVG:', err);
        setStatus({ message: 'Failed to convert SVG to image', type: 'error' });
        URL.revokeObjectURL(url);
      };
      
      img.src = url;
    } catch (error) {
      console.error('Clipboard error:', error);
      setStatus({ message: `Copy failed: ${error.message}`, type: 'error' });
    }
  };

  // Download InkML file
  const downloadInkML = () => {
    if (!result || !result.inkml) {
      setStatus({ message: 'No InkML data available', type: 'error' });
      return;
    }

    try {
      // Create blob and download link
      const blob = new Blob([result.inkml], { type: 'application/inkml+xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `whiteboard-${Date.now()}.inkml`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setStatus({ 
        message: 'InkML downloaded! Insert it into OneNote via Insert → File', 
        type: 'success' 
      });
    } catch (error) {
      console.error('Download error:', error);
      setStatus({ message: `Download failed: ${error.message}`, type: 'error' });
    }
  };

  // Send to OneNote
  const sendToOneNote = async () => {
    if (!result || !result.inkml) {
      setStatus({ message: 'No InkML data to send', type: 'error' });
      return;
    }

    // Check if OneNote is connected
    if (!oneNoteConnected) {
      setStatus({ message: 'Please connect to OneNote first', type: 'error' });
      await connectToOneNote();
      return;
    }

    if (!selectedSection) {
      setStatus({ message: 'Please select a section first', type: 'error' });
      setShowSidebar(true);
      return;
    }

    try {
      setStatus({ message: 'Sending InkML to OneNote...', type: 'info' });
      
      // Send InkML data to backend
      const response = await fetch(`${BACKEND_URL}/onenote/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          inkml_data: result.inkml,
          section_id: selectedSection.id
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to send to OneNote');
      }

      const data = await response.json();
      setStatus({ 
        message: `Sent to ${selectedNotebook.name} → ${selectedSection.name}`, 
        type: 'success' 
      });
    } catch (error) {
      console.error('OneNote error:', error);
      setStatus({ message: `OneNote failed: ${error.message}`, type: 'error' });
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera();
  }, []);

  return (
    <div className="app-container">
      {/* Backend Error Overlay (only shows on error after backend fails) */}
      {backendError && (
        <div className="loading-overlay">
          <div className="loading-content">
            <div className="error-icon">⚠️</div>
            <h2>Backend Error</h2>
            <p>{backendError}</p>
            <button className="button" onClick={() => window.location.reload()}>
              Retry
            </button>
          </div>
        </div>
      )}
      
      {/* Header */}
      <div className="header" style={{ opacity: backendReady ? 1 : 0 }}>
        <div>
          <h1>📸 Whiteboard Scanner</h1>
          <div className="subtitle">AI-powered whiteboard digitization for OneNote</div>
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="main-layout" style={{ opacity: backendReady ? 1 : 0 }}>
        {/* Left Column - Camera & Preview */}
        <div className="camera-column">
          <div className="camera-container">
            {/* Camera Placeholder / Loading */}
            {!cameraActive && !result && !processing && (
              <div className="camera-placeholder">
                <div className="placeholder-content">
                  {cameraLoading ? (
                    <>
                      <div className="spinner-large"></div>
                      <h2>Starting Camera...</h2>
                      <p>Requesting camera permissions...</p>
                    </>
                  ) : (
                    <>
                      <div className="placeholder-icon">📷</div>
                      <h2>Ready to Scan</h2>
                      <p>Position your whiteboard in frame</p>
                      <button 
                        className="button button-large button-primary" 
                        onClick={startCamera}
                        disabled={!oneNoteConnected || !backendReady}
                      >
                        {oneNoteConnected ? '📷 Start Camera' : '🔒 Sign In First'}
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Live Camera Feed */}
            {cameraActive && !result && (
              <div className="camera-active">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline
                  className="camera-feed"
                />
                <div className="camera-overlay">
                  <button 
                    className="button button-capture" 
                    onClick={capturePhoto}
                    disabled={processing}
                  >
                    📸 Capture Photo
                  </button>
                </div>
              </div>
            )}

            {/* Hidden Canvas */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Processing State */}
            {processing && (
              <div className="processing-overlay">
                <div className="spinner"></div>
                <p>Processing whiteboard with AI...</p>
              </div>
            )}

            {/* Result Preview */}
            {result && (
              <div className="result-view">
                {result.previewUrl && (
                  <img 
                    src={result.previewUrl} 
                    alt="Processed whiteboard" 
                    className="result-image"
                  />
                )}
                
                <div className="result-overlay">
                  <div className="result-actions">
                    <button 
                      className="button button-secondary" 
                      onClick={copyToClipboard}
                    >
                      📋 Copy
                    </button>
                    <button 
                      className="button button-primary" 
                      onClick={sendToOneNote}
                    >
                      📘 Send to OneNote
                    </button>
                    <button 
                      className="button" 
                      onClick={downloadInkML}
                    >
                      💾 Download
                    </button>
                    <button 
                      className="button button-secondary" 
                      onClick={() => {
                        setResult(null);
                        startCamera();
                      }}
                    >
                      🔄 Scan Another
                    </button>
                  </div>
                  
                  {result.metadata && (
                    <div className="result-metadata">
                      <span>✓ {result.metadata.strokes_count} strokes</span>
                      <span>✓ {result.metadata.colors_detected} colors</span>
                      <span>✓ {result.metadata.image_size?.width} × {result.metadata.image_size?.height}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Sidebar */}
        <div className="sidebar-column">
          {/* Instructions Card */}
          <div className="sidebar-card">
            <h3>📋 Quick Guide</h3>
            <ol className="compact-list">
              <li>Sign in to OneNote</li>
              <li>Select destination</li>
              <li>Start camera</li>
              <li>Capture whiteboard</li>
              <li>Send to OneNote</li>
            </ol>
          </div>

          {/* OneNote Connection Card */}
          {oneNoteAvailable && (
            <div className="sidebar-card onenote-card">
              <h3>📘 OneNote</h3>
              
              {!oneNoteConnected ? (
                <div className="connect-section">
                  <p className="info-text">Sign in to send scans directly to OneNote</p>
                  <button 
                    className="button button-primary button-block" 
                    onClick={connectToOneNote}
                    disabled={!backendReady}
                  >
                    🔑 Sign In with Microsoft
                  </button>
                </div>
              ) : (
                <div className="connected-section">
                  {/* User Info */}
                  <div className="user-badge">
                    <span className="user-icon">👤</span>
                    <div className="user-details">
                      <div className="user-email">{userEmail || 'Connected'}</div>
                      <button className="link-button" onClick={logoutFromOneNote}>
                        Logout
                      </button>
                    </div>
                  </div>

                  {/* Current Destination */}
                  {selectedNotebook && selectedSection ? (
                    <div className="destination-info">
                      <div className="destination-label">Sending to:</div>
                      <div className="destination-path">
                        📓 {selectedNotebook.name}
                        <span className="path-separator">→</span>
                        📄 {selectedSection.name}
                      </div>
                    </div>
                  ) : (
                    <div className="destination-info warning">
                      <div className="destination-label">⚠️ No destination selected</div>
                    </div>
                  )}

                  {/* Change Destination Button */}
                  <button 
                    className="button button-secondary button-block" 
                    onClick={() => setShowSidebar(!showSidebar)}
                  >
                    📂 Change Destination
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Status Messages */}
          {status.message && (
            <div className="sidebar-card status-card">
              <div className={`status-bar status-${status.type}`}>
                {status.message}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Full Screen Destination Selector */}
      {showSidebar && oneNoteConnected && (
        <div className="fullscreen-modal">
          <div className="modal-header">
            <h2>Select Destination</h2>
            {selectedSection && (
              <button className="close-button" onClick={() => setShowSidebar(false)}>
                ✕ Close
              </button>
            )}
          </div>
          <div className="modal-content">
            <div className="selection-grid">
              {/* Notebooks */}
              <div className="selection-section">
                <h4>📓 Notebooks</h4>
                <div className="item-grid">
                  {notebooks.map(nb => (
                    <div 
                      key={nb.id}
                      className={`grid-item ${selectedNotebook?.id === nb.id ? 'selected' : ''}`}
                      onClick={() => selectNotebook(nb)}
                    >
                      <div className="item-icon">📓</div>
                      <div className="item-name">{nb.name}</div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Sections and Section Groups */}
              {(sections.length > 0 || sectionGroups.length > 0) && (
                <div className="selection-section">
                  <h4>📄 Sections in {selectedNotebook?.name}</h4>
                  
                  {/* Back button if we're inside a section group */}
                  {currentSectionGroup && (
                    <div className="back-button grid-item" onClick={goBack}>
                      <div className="item-icon">⬅️</div>
                      <div className="item-name">Back to {selectedNotebook.name}</div>
                    </div>
                  )}
                  
                  <div className="item-grid">
                    {/* Section Groups first (folders) */}
                    {sectionGroups.map(group => (
                      <div 
                        key={group.id}
                        className="grid-item section-group-item"
                        onClick={() => openSectionGroup(group)}
                      >
                        <div className="item-icon">📁</div>
                        <div className="item-name">{group.name}</div>
                        <div className="expand-indicator">▶</div>
                      </div>
                    ))}
                    
                    {/* Then root-level sections */}
                    {sections.map(sec => (
                      <div 
                        key={sec.id}
                        className={`grid-item ${selectedSection?.id === sec.id ? 'selected' : ''}`}
                        onClick={() => selectSection(sec)}
                      >
                        <div className="item-icon">📄</div>
                        <div className="item-name">{sec.name}</div>
                      </div>
                    ))}
                    
                    {/* Create New Section button */}
                    {selectedNotebook && (
                      <div 
                        className="grid-item create-new-section"
                        onClick={createNewSection}
                      >
                        <div className="item-icon">➕</div>
                        <div className="item-name">Create New Section</div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {selectedNotebook && sections.length === 0 && sectionGroups.length === 0 && !currentSectionGroup && (
                <div className="selection-section">
                  <p className="loading-text">Loading sections...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
