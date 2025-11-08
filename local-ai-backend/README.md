# Local AI Backend

Python FastAPI server for processing whiteboard images and converting them to SVG vectors.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`

## API Endpoints

### Health Check
```
GET /
GET /health
```

### Process Whiteboard Image
```
POST /process-image
Content-Type: multipart/form-data

Parameters:
  - file: Image file (JPEG or PNG)

Response:
{
  "success": true,
  "svg": "<svg>...</svg>",
  "metadata": {
    "strokes_count": 42,
    "colors_detected": 3,
    "image_size": {"width": 1280, "height": 720}
  }
}
```

### Preview Preprocessing
```
POST /preview-image
Content-Type: multipart/form-data

Parameters:
  - file: Image file

Response:
{
  "success": true,
  "preview": "data:image/png;base64,...",
  "original_size": {"width": 1920, "height": 1080},
  "processed_size": {"width": 1280, "height": 720}
}
```

## Configuration

Edit `config.py` to customize processing parameters:

- `MAX_IMAGE_SIZE`: Maximum image resolution
- `NUM_COLORS`: Number of marker colors to detect
- `MIN_CONTOUR_AREA`: Minimum stroke size
- `EPSILON_FACTOR`: Curve smoothing factor

## Architecture

```
local-ai-backend/
├── app.py                  # FastAPI application
├── config.py               # Configuration settings
├── ai/
│   ├── preprocess.py       # Image preprocessing
│   ├── color_segment.py    # Color detection
│   ├── stroke_extract.py   # Stroke extraction
│   └── vectorize.py        # SVG generation
└── requirements.txt
```

## Testing

```bash
# Test with curl
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:5000/process-image

# Or use the provided test script
python test_api.py
```

## Troubleshooting

**OpenCV not loading:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**Port already in use:**
Edit `config.py` and change `PORT = 5000` to another port.

**Memory issues with large images:**
Reduce `MAX_IMAGE_SIZE` in `config.py`.
