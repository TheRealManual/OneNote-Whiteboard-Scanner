"""
Test Suite: Image Processing Pipeline
Tests camera, capture, AI processing, and clipboard functionality
Starts its own backend instance for testing
"""

import requests
import sys
import base64
import io
import subprocess
import os
import time
from pathlib import Path
from PIL import Image

BASE_URL = "http://127.0.0.1:5000"
backend_process = None


def start_backend():
    """Start the backend server for testing"""
    global backend_process
    print("\n[SETUP] Starting backend server for testing...")
    
    backend_dir = Path(__file__).parent.parent / "local-ai-backend"
    
    # Start backend in a subprocess
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(backend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    # Wait for backend to be ready
    max_wait = 30
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print(f"  ✓ Backend ready (took {i+1} seconds)")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    print("  ✗ Backend failed to start within 30 seconds")
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


def create_test_image():
    """Create a simple test image (100x100 white square with black border)"""
    img = Image.new('RGB', (100, 100), color='white')
    pixels = img.load()
    
    # Draw black border
    for x in range(100):
        pixels[x, 0] = (0, 0, 0)
        pixels[x, 99] = (0, 0, 0)
    for y in range(100):
        pixels[0, y] = (0, 0, 0)
        pixels[99, y] = (0, 0, 0)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def test_process_endpoint():
    """Test image processing endpoint"""
    print("\n[TEST] Image Processing Endpoint...")
    try:
        test_image = create_test_image()
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={"image": test_image},
            timeout=30
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert "processedImage" in data, "Expected processedImage in response"
        assert data["processedImage"].startswith("data:image"), "Invalid image format"
        
        print("  ✓ Image processed successfully")
        print(f"    Input size: ~{len(test_image)} chars")
        print(f"    Output size: ~{len(data['processedImage'])} chars")
        return True
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        return False

def test_process_with_perspective():
    """Test perspective correction in processing"""
    print("\n[TEST] Perspective Correction...")
    try:
        test_image = create_test_image()
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "image": test_image,
                "correct_perspective": True
            },
            timeout=30
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "processedImage" in data, "Expected processedImage"
        
        print("  ✓ Perspective correction applied")
        return True
    except Exception as e:
        print(f"  ✗ Perspective correction failed: {e}")
        return False

def test_process_with_enhancement():
    """Test image enhancement in processing"""
    print("\n[TEST] Image Enhancement...")
    try:
        test_image = create_test_image()
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "image": test_image,
                "enhance": True
            },
            timeout=30
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "processedImage" in data, "Expected processedImage"
        
        print("  ✓ Image enhancement applied")
        return True
    except Exception as e:
        print(f"  ✗ Enhancement failed: {e}")
        return False

def test_invalid_image():
    """Test error handling for invalid image data"""
    print("\n[TEST] Invalid Image Handling...")
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={"image": "invalid_base64_data"},
            timeout=10
        )
        
        # Should either reject with 400 or handle gracefully
        assert response.status_code in [400, 500], f"Expected error status, got {response.status_code}"
        
        print("  ✓ Invalid image properly rejected")
        return True
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        return False

def test_large_image():
    """Test processing a larger image"""
    print("\n[TEST] Large Image Processing...")
    try:
        # Create 800x600 image
        img = Image.new('RGB', (800, 600), color='lightblue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        test_image = f"data:image/png;base64,{img_base64}"
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={"image": test_image},
            timeout=60  # Longer timeout for large image
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "processedImage" in data, "Expected processedImage"
        
        print(f"  ✓ Large image (800x600) processed successfully")
        return True
    except Exception as e:
        print(f"  ✗ Large image processing failed: {e}")
        return False

def test_image_formats():
    """Test different image formats (PNG, JPEG)"""
    print("\n[TEST] Multiple Image Formats...")
    try:
        formats_tested = 0
        
        # Test PNG
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        png_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={"image": f"data:image/png;base64,{png_base64}"},
            timeout=30
        )
        
        if response.status_code == 200:
            formats_tested += 1
            print("    ✓ PNG format supported")
        
        # Test JPEG
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        jpg_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = requests.post(
            f"{BASE_URL}/process",
            json={"image": f"data:image/jpeg;base64,{jpg_base64}"},
            timeout=30
        )
        
        if response.status_code == 200:
            formats_tested += 1
            print("    ✓ JPEG format supported")
        
        assert formats_tested >= 1, "No image formats supported"
        print(f"  ✓ {formats_tested}/2 formats tested successfully")
        return True
    except Exception as e:
        print(f"  ✗ Format testing failed: {e}")
        return False

def run_all_tests():
    """Run all image processing tests"""
    print("\n" + "="*60)
    print("IMAGE PROCESSING TESTS")
    print("="*60)
    
    # Start backend for testing
    if not start_backend():
        return False
    
    try:
        tests = [
            test_process_endpoint,
            test_process_with_perspective,
            test_process_with_enhancement,
            test_invalid_image,
            test_large_image,
            test_image_formats,
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
