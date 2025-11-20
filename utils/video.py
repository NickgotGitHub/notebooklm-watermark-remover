import cv2
import numpy as np
from typing import Tuple, Optional
import os
import time
import shutil


def auto_detect_bottom_right_roi(width: int, height: int) -> Tuple[int, int, int, int]:
    """Auto-detect ROI for NotebookLM watermark in bottom right corner.
    
    Based on reference dimensions: 1470x956 video with watermark at (1263, 860) to (1425, 890)
    Scales proportionally for any video resolution.
    
    Returns:
        (x, y, w, h): ROI coordinates for the watermark area.
    """
    # Reference dimensions from user's video (moved down 100px from original)
    ref_width = 1470
    ref_height = 956
    ref_x = 1240
    ref_y = 850  # Was 760, moved down by 100px
    ref_w = 200  # 1425 - 1263
    ref_h = 60   # 890 - 860
    
    # Scale proportionally to current video dimensions
    scale_x = width / ref_width
    scale_y = height / ref_height
    
    x = int(ref_x * scale_x)
    y = int(ref_y * scale_y)
    w = int(ref_w * scale_x)
    h = int(ref_h * scale_y)
    
    return (x, y, w, h)


def overlay_logo(frame: np.ndarray, logo_path: str, position: Tuple[int, int, int, int]) -> np.ndarray:
    """Overlay a logo onto a frame at the specified position, maintaining aspect ratio.
    
    Args:
        frame: The video frame to overlay onto
        logo_path: Path to the logo image (PNG with transparency)
        position: (x, y, w, h) bounding box where to place the logo
        
    Returns:
        Frame with logo overlaid
    """
    if not os.path.exists(logo_path):
        return frame
    
    x, y, w, h = position
    
    # Load logo with alpha channel
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return frame
    
    # Get original logo dimensions
    logo_h, logo_w = logo.shape[:2]
    logo_aspect = logo_w / logo_h
    
    # Calculate new dimensions maintaining aspect ratio to fit within ROI
    # The logo should fit within the height of the ROI (30px based on reference)
    new_h = h
    new_w = int(new_h * logo_aspect)
    
    # If width exceeds ROI width, scale based on width instead
    if new_w > w:
        new_w = w
        new_h = int(new_w / logo_aspect)
    
    # Center the logo within the ROI area
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2
    
    # Resize logo maintaining aspect ratio
    logo_resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate actual position for centered logo
    actual_x = x + offset_x
    actual_y = y + offset_y
    
    # Handle different logo formats
    if logo_resized.shape[2] == 4:  # Has alpha channel
        alpha = logo_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[actual_y:actual_y+new_h, actual_x:actual_x+new_w, c] = (
                alpha * logo_resized[:, :, c] +
                (1 - alpha) * frame[actual_y:actual_y+new_h, actual_x:actual_x+new_w, c]
            )
    else:  # No alpha channel, just overlay
        frame[actual_y:actual_y+new_h, actual_x:actual_x+new_w] = logo_resized[:, :, :3]
    
    return frame


def remove_watermark_roi(input_video_path: str, output_video_path: str, roi: Tuple[int, int, int, int], inpaint_method: str = 'telea') -> None:
    """Remove a rectangular watermark using inpainting across all frames and write a temporary video.

    Note: This writes a lossy intermediate. Prefer remove_watermark_roi_to_frames for best quality.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to open input video')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError('Failed to open output video for writing')

    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    flags = cv2.INPAINT_TELEA if inpaint_method.lower() == 'telea' else cv2.INPAINT_NS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        inpainted = cv2.inpaint(frame, mask, 3, flags)
        out.write(inpainted)

    cap.release()
    out.release()


def remove_watermark_roi_to_frames(
    input_video_path: str, 
    output_frames_dir: str, 
    roi: Optional[Tuple[int, int, int, int]] = None, 
    inpaint_method: str = 'telea',
    logo_path: Optional[str] = None
) -> Tuple[float, int, int]:
    """Remove watermark and write lossless PNG frames to a directory.
    Optimized for still-frame videos by detecting and reusing duplicate frames.
    Maintains original FPS and duration for perfect audio sync.
    
    Args:
        input_video_path: Path to input video
        output_frames_dir: Directory to write output frames
        roi: ROI (x, y, w, h) for watermark. If None, auto-detects bottom right.
        inpaint_method: 'telea' or 'ns' for inpainting
        logo_path: Optional path to logo to overlay after watermark removal
        
    Returns:
        Tuple of (fps, output_width, output_height) for proper encoding.
    """
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to open input video')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # Process at 2 FPS for speed, then duplicate frames to maintain original FPS
    target_process_fps = 2.0
    frame_skip = int(fps / target_process_fps)
    if frame_skip < 1:
        frame_skip = 1
    
    # Determine output resolution - downscale if too large for faster PNG writing
    # Target max 720p for good balance of quality and speed
    max_height = 720
    if height > max_height:
        output_scale = max_height / height
        output_width = int(width * output_scale)
        output_height = max_height
        # Make dimensions even for video encoding
        if output_width % 2 == 1:
            output_width -= 1
        if output_height % 2 == 1:
            output_height -= 1
        print(f"Video: {width}x{height} → downscaling to {output_width}x{output_height} for faster processing", flush=True)
    else:
        output_width = width
        output_height = height
        print(f"Video: {width}x{height} (no downscaling needed)", flush=True)
    
    print(f"Processing: {fps:.1f} FPS, every {frame_skip} frames (effective {fps/frame_skip:.1f} FPS)", flush=True)

    # Auto-detect ROI if not provided
    if roi is None:
        roi = auto_detect_bottom_right_roi(width, height)
    
    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    flags = cv2.INPAINT_TELEA if inpaint_method.lower() == 'telea' else cv2.INPAINT_NS

    # Optimization for still frames: track previous frame and reuse processed result
    prev_frame_hash = None
    prev_processed_frame = None
    frames_skipped = 0
    frames_processed = 0
    frames_duplicated = 0
    
    # Timing trackers
    time_reading = 0
    time_comparison = 0
    time_mask_creation = 0
    time_inpainting = 0
    time_logo_overlay = 0
    time_writing_encode = 0  # PNG encoding time
    time_writing_copy = 0    # File copy time
    time_total_start = time.time()
    
    input_frame_idx = 0
    output_idx = 0
    
    while True:
        # Time: Frame reading
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        time_reading += time.time() - t_start
        
        # Only process every Nth frame (2 FPS processing)
        if input_frame_idx % frame_skip == 0:
            # Time: Frame comparison
            t_start = time.time()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_tiny = cv2.resize(frame_gray, (16, 9), interpolation=cv2.INTER_NEAREST)
            frame_hash = (frame_tiny.mean(), frame_tiny.std(), frame_tiny[0,0], frame_tiny[8,8], frame_tiny[4,4])
            time_comparison += time.time() - t_start
            
            # Check if frame is identical to previous frame
            if prev_frame_hash is not None and frame_hash == prev_frame_hash and prev_processed_frame is not None:
                # Reuse previous processed frame - no need to inpaint again!
                inpainted = prev_processed_frame
                frames_skipped += 1
            else:
                # Different frame - process it
                # Time: Mask creation
                t_start = time.time()
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                time_mask_creation += time.time() - t_start
                
                # Time: Inpainting (watermark removal)
                t_start = time.time()
                inpainted = cv2.inpaint(frame, mask, 3, flags)
                time_inpainting += time.time() - t_start
                
                # Time: Logo overlay
                if logo_path and os.path.exists(logo_path):
                    t_start = time.time()
                    inpainted = overlay_logo(inpainted, logo_path, (x, y, w, h))
                    time_logo_overlay += time.time() - t_start
                
                # Cache for next iteration
                prev_frame_hash = frame_hash
                prev_processed_frame = inpainted.copy()
                frames_processed += 1
            
            # Downscale before writing if needed (much faster PNG compression)
            if output_width != width or output_height != height:
                inpainted_resized = cv2.resize(inpainted, (output_width, output_height), interpolation=cv2.INTER_AREA)
            else:
                inpainted_resized = inpainted
            
            # Write this processed frame multiple times to maintain original FPS
            # Optimization: Write first frame with PNG encoding, then just copy the file for duplicates
            
            # Write the first frame (PNG encoding required)
            t_encode = time.time()
            first_frame_path = os.path.join(output_frames_dir, f"frame_{output_idx:06d}.png")
            ok = cv2.imwrite(first_frame_path, inpainted_resized)
            if not ok:
                cap.release()
                raise RuntimeError(f'Failed to write frame {first_frame_path}')
            time_writing_encode += time.time() - t_encode
            output_idx += 1
            
            # For duplicate frames, just copy the file (much faster than re-encoding!)
            t_copy = time.time()
            for dup in range(1, frame_skip):
                duplicate_path = os.path.join(output_frames_dir, f"frame_{output_idx:06d}.png")
                shutil.copy2(first_frame_path, duplicate_path)
                output_idx += 1
                frames_duplicated += 1
            time_writing_copy += time.time() - t_copy
            
            # Progress indicator every 30 output frames
            if output_idx % 90 == 0:
                elapsed = time.time() - time_total_start
                fps_current = output_idx / elapsed if elapsed > 0 else 0
                print(f"Progress: {output_idx} frames written ({input_frame_idx} read, {frames_processed} processed) in {elapsed:.1f}s ({fps_current:.1f} fps)", flush=True)
        
        input_frame_idx += 1

    cap.release()
    time_total = time.time() - time_total_start
    
    # Print detailed timing breakdown with flush to ensure it appears in logs
    print("\n" + "="*60, flush=True)
    print("DETAILED TIMING BREAKDOWN", flush=True)
    print("="*60, flush=True)
    print(f"Input frames read: {input_frame_idx}", flush=True)
    print(f"Output frames written: {output_idx}", flush=True)
    print(f"Actual frames processed: {frames_processed}", flush=True)
    print(f"Duplicate frames reused: {frames_skipped}", flush=True)
    print(f"Frames duplicated (2 FPS): {frames_duplicated}", flush=True)
    if frames_processed > 0:
        reduction = ((input_frame_idx - frames_processed) / input_frame_idx * 100) if input_frame_idx > 0 else 0
        print(f"Processing reduction: {reduction:.1f}% (only processed {frames_processed}/{input_frame_idx} frames)", flush=True)
    print(flush=True)
    print(f"{'Operation':<25} {'Time (s)':<12} {'% of Total':<12} {'Per Frame (ms)':<15}", flush=True)
    print("-"*60, flush=True)
    
    time_writing_total = time_writing_encode + time_writing_copy
    frames_encoded = frames_processed + frames_skipped
    
    operations = [
        ("Frame Reading", time_reading, input_frame_idx),
        ("Frame Comparison", time_comparison, frames_processed + frames_skipped),
        ("Mask Creation", time_mask_creation, frames_processed),
        ("Inpainting (removal)", time_inpainting, frames_processed),
        ("Logo Overlay", time_logo_overlay, frames_processed),
        ("PNG Encoding", time_writing_encode, frames_encoded),
        ("File Copying (dupes)", time_writing_copy, frames_duplicated),
    ]
    
    for op_name, op_time, op_count in operations:
        pct = (op_time / time_total * 100) if time_total > 0 else 0
        per_frame = (op_time / op_count * 1000) if op_count > 0 else 0
        print(f"{op_name:<25} {op_time:<12.2f} {pct:<12.1f} {per_frame:<15.1f}", flush=True)
    
    print("-"*60, flush=True)
    print(f"{'TOTAL TIME':<25} {time_total:<12.2f} {'100.0':<12} {(time_total/output_idx*1000):<15.1f}", flush=True)
    print("="*60, flush=True)
    print(f"\nProcessing speed: {output_idx/time_total:.2f} output frames/second", flush=True)
    print(f"Effective speedup: {input_frame_idx/(frames_processed+frames_skipped):.1f}x (processed {frames_processed+frames_skipped} instead of {input_frame_idx})", flush=True)
    print(flush=True)
    
    return float(fps), output_width, output_height  # Return FPS and output dimensions
