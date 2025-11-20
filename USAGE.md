# NotebookLM Watermark Remover - Usage Guide

## Overview
This application automatically detects and removes the NotebookLM watermark from videos (located in the bottom-right corner) and replaces it with your custom logo.

## Features

✨ **Automatic Detection**: No manual selection needed - automatically detects the NotebookLM watermark in the bottom-right corner (approximately 18% width × 10% height from bottom-right)

🎨 **Logo Replacement**: Automatically overlays your Logo.png file after watermark removal with full transparency support

📊 **Before & After Comparison**: Side-by-side video comparison without scrolling

⚡ **One-Click Processing**: Simple one-button workflow - upload and process

🎥 **High Quality**: Lossless processing with ultra quality mode (default)

## How to Use

1. **Open the App**: Navigate to http://127.0.0.1:5000

2. **Upload Video**: 
   - Drag & drop your video file, or click "Browse Files"
   - Supported formats: MP4, MOV, WEBM, MKV, AVI

3. **Configure Settings** (optional):
   - **Method**: 
     - Telea (fast, good quality) - Default
     - Navier-Stokes (slower, better quality)
   - **Quality**:
     - Ultra (lossless) - Default
     - Best, Better, Balanced, Fast

4. **Process**: Click "Remove Watermark & Add Logo"

5. **Review & Download**: 
   - View before/after comparison side-by-side
   - Videos play in sync
   - Click "Download Processed Video" to save

## Technical Details

### Watermark Detection ROI
- **Position**: Bottom-right corner
- **Width**: 18% of video width
- **Height**: 10% of video height
- **Padding**: 10px from edges

### Logo Requirements
- **File**: `Logo.png` in project root
- **Format**: PNG with alpha channel (transparency) supported
- **Scaling**: Automatically resized to fit watermark area

### Processing Pipeline
1. Auto-detect NotebookLM watermark region
2. Remove watermark using OpenCV inpainting
3. Overlay custom logo with transparency
4. Export lossless frames as PNG
5. Re-encode with FFmpeg (ultra quality by default)

## Customization

To change the watermark detection area, edit `utils/video.py`:

```python
def auto_detect_bottom_right_roi(width: int, height: int) -> Tuple[int, int, int, int]:
    w = int(width * 0.18)   # Adjust width percentage
    h = int(height * 0.10)  # Adjust height percentage
    x = width - w - 10      # Adjust right padding
    y = height - h - 10     # Adjust bottom padding
    return (x, y, w, h)
```

## Troubleshooting

**Logo not appearing?**
- Ensure `Logo.png` exists in project root
- Check logo has correct format (PNG recommended)

**Watermark not fully removed?**
- Try increasing ROI dimensions in `auto_detect_bottom_right_roi()`
- Switch to Navier-Stokes method for better inpainting

**Processing slow?**
- Reduce quality setting
- Use Telea method instead of Navier-Stokes
- Consider video resolution

## Notes

- Original videos are never modified
- Processing time depends on video length and quality settings
- Temporary frames are automatically cleaned up after processing
- All processing happens locally - no external services

