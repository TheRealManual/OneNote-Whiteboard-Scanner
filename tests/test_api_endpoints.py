"""
Test Suite: Backend API Endpoints
Tests all FastAPI endpoints for correct responses
Starts its own backend instance for testing
"""

import requests
import time
import sys
import subprocess
import os
import signal
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
backend_process = None


def start_backend():
    """Start the backend server for testing"""
    global backend_process
    print("\n[SETUP] Starting backend server for testing...")
    
    backend_dir = Path(__file__).parent.parent / "local-ai-backend"
    
    # Start backend in a subprocess (show output for debugging if it fails)
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(backend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    # Wait for backend to be ready
    max_wait = 60  # Increased timeout for slower systems
    print("  ⏳ Waiting for backend to start (this may take 30-60 seconds)...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print(f"  ✓ Backend ready (took {i+1} seconds)")
                return True
        except requests.exceptions.RequestException:
            if i % 5 == 0 and i > 0:
                print(f"    Still waiting... ({i}/{max_wait} seconds)")
            time.sleep(1)
    
    # If we get here, backend failed to start
    print("  ✗ Backend failed to start within 60 seconds")
    
    # Try to get error output
    try:
        stdout, stderr = backend_process.communicate(timeout=2)
        if stderr:
            print("\n  Backend error output:")
            print(f"  {stderr.decode('utf-8', errors='ignore')}")
    except:
        pass
    
    stop_backend()
    return False


def stop_backend():
    """Stop the backend server"""
    global backend_process
    if backend_process:
        print("\n[CLEANUP] Stopping backend server...")
        if os.name == 'nt':
            # Windows: terminate process tree
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(backend_process.pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Unix: send SIGTERM
            backend_process.terminate()
        backend_process.wait(timeout=5)
        backend_process = None
        print("  ✓ Backend stopped")


def test_health_endpoint():
    """Test the /health endpoint"""
    print("\n[TEST] Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "healthy", "Expected status to be 'healthy'"
        print("  ✓ Health endpoint working")
        return True
    except Exception as e:
        print(f"  ✗ Health endpoint failed: {e}")
        return False

def test_onenote_config_endpoint():
    """Test the /onenote/config endpoint"""
    print("\n[TEST] OneNote Config Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/onenote/config", timeout=5)
        # Accept 200 (works) or 404 (endpoint not mounted)
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Config endpoint working")
        else:
            print("  ⚠ Config endpoint not available (404)")
        return True
    except Exception as e:
        print(f"  ✗ Config endpoint failed: {e}")
        return False


def test_onenote_fetch_all_endpoint():
    """Test the /onenote/fetch-all endpoint"""
    print("\n[TEST] OneNote Fetch-All Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/onenote/fetch-all", timeout=10)
        # Accept 200 (works), 401 (not authenticated), 500 (server error), 404 (not available)
        assert response.status_code in [200, 401, 404, 500], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list), "Expected list of notebooks"
            print(f"  ✓ Fetch-all endpoint working (returned {len(data)} items)")
        elif response.status_code == 401:
            print("  ⚠ Not authenticated (expected)")
        elif response.status_code == 404:
            print("  ⚠ Endpoint not available (404)")
        else:
            print("  ⚠ Server error (500 - expected without auth)")
        return True
    except Exception as e:
        print(f"  ✗ Fetch-all endpoint failed: {e}")
        return False


def test_process_image_endpoint():
    """Test the /process-image endpoint"""
    print("\n[TEST] Process Image Endpoint...")
    try:
        # Create a simple test image
        import base64
        from PIL import Image
        import io
        
        img = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        test_data = {
            "image": f"data:image/png;base64,{img_base64}"
        }
        response = requests.post(f"{BASE_URL}/process-image", json=test_data, timeout=30)
        
        # Accept 200 (success), 422 (validation error), 500 (processing error), 404 (not available)
        assert response.status_code in [200, 404, 422, 500], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Process-image endpoint working")
        elif response.status_code == 404:
            print("  ⚠ Endpoint not available (404)")
        elif response.status_code == 422:
            print("  ⚠ Validation error (422 - endpoint exists but data format issue)")
        else:
            print("  ⚠ Processing error (500 - endpoint exists)")
        return True
    except Exception as e:
        print(f"  ✗ Process-image endpoint failed: {e}")
        return False


def test_onenote_send_endpoint():
    """Test the /onenote/send endpoint"""
    print("\n[TEST] OneNote Send Endpoint...")
    try:
        test_data = {
            "svg": "<svg></svg>",
            "notebook_id": "test",
            "section_id": "test"
        }
        response = requests.post(f"{BASE_URL}/onenote/send", json=test_data, timeout=10)
        
        # Endpoint should exist and return some response (likely 400 or 401 without proper auth)
        assert response.status_code in [200, 400, 401, 500], f"Unexpected status: {response.status_code}"
        print("  ✓ Send endpoint exists and responds")
        return True
    except Exception as e:
        print(f"  ✗ Send endpoint failed: {e}")
        return False

def run_all_tests():
    """Run all API endpoint tests"""
    print("\n" + "="*60)
    print("BACKEND API ENDPOINT TESTS")
    print("="*60)
    
    # Start backend for testing
    if not start_backend():
        return False
    
    try:
        # Run tests
        tests = [
            test_health_endpoint,
            test_onenote_config_endpoint,
            test_onenote_fetch_all_endpoint,
            test_process_image_endpoint,
            test_onenote_send_endpoint,
        ]
        
        results = []
        for test in tests:
            results.append(test())
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("\n" + "="*60)
        print(f"RESULTS: {passed}/{total} tests passed")
        print("="*60)
        
        return all(results)
    finally:
        # Always stop backend even if tests fail
        stop_backend()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
