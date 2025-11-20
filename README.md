# NotebookLM Watermark Remover

A Flask-based web app specifically designed to automatically remove the NotebookLM watermark from videos and replace it with your custom logo. One-click processing with before/after comparison.

## ✨ Features

### 🎯 Automatic Watermark Detection
- **No manual selection required** - automatically detects NotebookLM watermark in bottom-right corner
- Pre-configured for NotebookLM's rainbow logo position
- Works with any video resolution (proportional scaling)

### 🎨 Custom Logo Replacement
- Automatically overlays your `Logo.png` after watermark removal
- Full transparency (alpha channel) support
- Auto-scales to fit watermark area
- Maintains aspect ratio

### ⚡ Optimized Performance
- **Multiprocessing** - Parallel frame processing across CPU cores (2-4x speedup)
- **2 FPS processing** with frame deduplication (100x+ faster than original)
- Ultra-fast frame comparison using downsampled hashing
- 720p downscaling for faster encoding
- Batch file copying for duplicate frames
- Detailed timing diagnostics

### 🎬 Before/After Comparison
- Side-by-side video comparison with slider
- Interactive swipe to compare original vs processed
- Independent playback controls
- No scrolling required

### 🎨 Modern UI
- One-button workflow
- Gradient purple theme
- Responsive design (mobile-friendly)
- Toast notifications
- Clean upload/results flow

## 🚀 Quick Start

### Requirements
- **Python**: 3.9+
- **FFmpeg**: Installed and available on PATH (`ffmpeg` and `ffprobe`)

On macOS using Homebrew:
```bash
brew install ffmpeg
```

### Installation

```bash
cd notebooklm-watermark-remover
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Add Your Logo
Place your custom logo as `Logo.png` in the project root directory. For best results:
- Use PNG format with alpha channel (transparency)
- Recommended size: ~220px width
- Square or landscape orientation works best

### Run the App
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## 📖 Usage

1. **Upload** - Select or drag & drop a NotebookLM video
2. **Configure** (Optional) - Choose quality: Fast (default), Normal, or Ultra
3. **Process** - Click "Remove Watermark & Add Logo"
4. **Compare** - Use the slider to compare before/after
5. **Download** - Get your processed video

That's it! The watermark is automatically detected and replaced.

## ⚙️ Quality Modes

- **Fast** (Default): x264 veryfast, CRF 23 - Quick processing, good quality
- **Normal**: x264 medium, CRF 20 - Balanced quality/speed
- **Ultra**: x264 veryslow, CRF 16 - Near-lossless, slower processing

## 📁 Project Structure

```text
notebooklm-watermark-remover/
  app.py                 # Flask app & API endpoints
  requirements.txt       # Python dependencies
  Logo.png              # Your custom logo (add this!)
  Dockerfile            # Docker configuration
  DEPLOYMENT.md         # Deployment guide (Railway, Render, Fly.io)
  CHANGES.md            # Detailed changelog
  USAGE.md              # Extended usage guide
  templates/
    index.html          # UI
  static/
    app.js              # Frontend logic
    style.css           # Styling
  utils/
    __init__.py
    video.py            # Video processing & watermark removal
  uploads/              # Uploaded videos (auto-created, gitignored)
  outputs/              # Processed outputs (auto-created, gitignored)
  temp/                 # Temp frames (auto-created, gitignored)
```

## 🐳 Deployment

### Local Development
```bash
python app.py
```

### Docker
```bash
docker build -t notebooklm-watermark-remover .
docker run -p 5000:5000 notebooklm-watermark-remover
```

### Railway.app (Recommended)
See `DEPLOYMENT.md` for detailed instructions on deploying to Railway.app, which supports:
- Large file uploads (2GB+)
- Long processing times (unlimited)
- FFmpeg binary
- Persistent storage

**Note:** Vercel is **not suitable** for this application due to 4.5MB request body limit.

## 🛠️ Technical Details

### Watermark Detection Algorithm
```python
# Automatically calculates ROI based on video resolution
# Reference: 1470x956 video
# NotebookLM watermark: 200x60px @ (1240, 850)
# Scales proportionally for any resolution
```

### Processing Pipeline
1. Upload video to `/uploads/`
2. Auto-detect ROI in bottom-right (200x60px proportional)
3. Extract and process frames at 2 FPS
4. Detect duplicate frames using fast hashing
5. Remove watermark using OpenCV inpainting (Telea/Navier-Stokes)
6. Overlay Logo.png with alpha transparency
7. Write processed frames (720p downscaled)
8. Batch-copy duplicate frames
9. Re-encode with FFmpeg at original FPS
10. Mux with original audio
11. Serve before/after comparison

### Performance Optimizations
- **Multiprocessing**: Parallel frame processing using all CPU cores (default: CPU count - 1)
  - Processes multiple frames simultaneously
  - Automatic load balancing across workers
  - 2-4x speedup on 4-8 core systems
- **2 FPS Processing**: Process 1 frame per 15 (for 30fps video), then duplicate
- **Frame Hashing**: Ultra-fast duplicate detection (16x9 downsampled comparison)
- **720p Encoding**: Reduces PNG write time by ~70%
- **Batch Copying**: File copy for duplicates instead of re-encoding
- **Result**: ~15-30 seconds for a 40-second video (was ~60+ minutes before optimization)

## 🔧 API Reference

### POST `/upload`
Upload a video file
- **Body**: `multipart/form-data` with `video` field
- **Returns**: `{ filename, videoUrl }`

### POST `/process`
Process the uploaded video
- **Body**: `{ filename, method?, quality? }`
- **Returns**: `{ downloadUrl, outputFilename, videoUrl }`

### GET `/video/<filename>`
Stream original uploaded video

### GET `/output/<filename>`
Stream processed output video

### GET `/download/<filename>`
Download processed video

## 🐛 Troubleshooting

### "ffmpeg failed"
- Ensure `ffmpeg` and `ffprobe` are installed: `ffmpeg -version`
- On macOS: `brew install ffmpeg`
- On Linux: `sudo apt install ffmpeg`

### Processing is slow
- Use "Fast" quality mode
- Ensure optimizations are enabled (check `utils/video.py` for 2 FPS processing)
- Process shorter videos first to test

### Watermark not fully removed
- Check `Logo.png` positioning in output
- The ROI is auto-calculated based on video resolution
- For custom positioning, modify `auto_detect_bottom_right_roi()` in `utils/video.py`

### Logo looks pixelated
- Use a higher resolution logo (minimum 220px width recommended)
- Ensure logo has transparent background (PNG with alpha channel)

### Upload fails on deployed version
- Vercel has 4.5MB upload limit - **not suitable for videos**
- Use Railway.app, Render.com, or Fly.io instead (see `DEPLOYMENT.md`)

## 🔒 Security & Privacy

- All processing is done **locally** on your server
- No external API calls or data sharing
- Files are stored temporarily in `uploads/`, `outputs/`, and `temp/`
- Clean up old files periodically if deploying publicly:
  ```bash
  rm -rf uploads/* outputs/* temp/*
  ```

## 📄 License

MIT License - feel free to use this for personal or commercial projects!

## 🤝 Contributing

Pull requests welcome! Areas for improvement:
- Support for other watermark positions
- Multiple logo positions
- Batch processing
- Progress streaming via WebSocket
- GPU acceleration for faster processing

## 🙏 Credits

Built with:
- **Flask** - Web framework
- **OpenCV** - Video processing & inpainting
- **FFmpeg** - Video encoding/decoding
- **NumPy** - Array operations

---

**Made for NotebookLM users** 🎙️ | [View Demo](https://github.com/NickgotGitHub/notebooklm-watermark-remover) | [Report Issues](https://github.com/NickgotGitHub/notebooklm-watermark-remover/issues)
