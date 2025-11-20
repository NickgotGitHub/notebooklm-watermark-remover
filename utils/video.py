import cv2
import numpy as np
from typing import Tuple, Optional
import os
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


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


def _process_single_frame(args):
    """Worker function to process a single frame (for multiprocessing).
    
    Args:
        args: Tuple of (frame_data, roi, inpaint_method, logo_path, frame_idx)
    
    Returns:
        Tuple of (frame_idx, processed_frame)
    """
    frame, roi, inpaint_method, logo_path, frame_idx = args
    
    height, width = frame.shape[:2]
    x, y, w, h = roi
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    
    # Inpaint
    flags = cv2.INPAINT_TELEA if inpaint_method.lower() == 'telea' else cv2.INPAINT_NS
    inpainted = cv2.inpaint(frame, mask, 3, flags)
    
    # Overlay logo if provided
    if logo_path and os.path.exists(logo_path):
        inpainted = overlay_logo(inpainted, logo_path, (x, y, w, h))
    
    return (frame_idx, inpainted)


def remove_watermark_roi_to_frames(
    input_video_path: str, 
    output_frames_dir: str, 
    roi: Optional[Tuple[int, int, int, int]] = None, 
    inpaint_method: str = 'telea',
    logo_path: Optional[str] = None,
    use_multiprocessing: bool = True,
    num_workers: Optional[int] = None
) -> Tuple[float, int, int]:
    """Remove watermark and write lossless PNG frames to a directory.
    Optimized for still-frame videos by detecting and reusing duplicate frames.
    Maintains original FPS and duration for perfect audio sync.
    Uses multiprocessing to parallelize frame processing across CPU cores.
    
    Args:
        input_video_path: Path to input video
        output_frames_dir: Directory to write output frames
        roi: ROI (x, y, w, h) for watermark. If None, auto-detects bottom right.
        inpaint_method: 'telea' or 'ns' for inpainting
        logo_path: Optional path to logo to overlay after watermark removal
        use_multiprocessing: Enable parallel processing (default: True)
        num_workers: Number of worker processes. If None, uses CPU count - 1.
        
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

    # Determine number of workers for multiprocessing
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core for system
    
    if use_multiprocessing:
        print(f"Multiprocessing: Enabled with {num_workers} worker processes", flush=True)
    else:
        print(f"Multiprocessing: Disabled (sequential processing)", flush=True)

    # Timing trackers
    time_reading = 0
    time_comparison = 0
    time_processing = 0  # Includes mask, inpainting, logo
    time_writing_encode = 0  # PNG encoding time
    time_writing_copy = 0    # File copy time
    time_total_start = time.time()
    
    # Step 1: Read all frames to process and compute hashes
    print("Step 1: Reading frames and detecting duplicates...", flush=True)
    t_read_start = time.time()
    
    frames_to_process = []  # List of (input_idx, frame, hash, output_batch_start)
    frame_hash_map = {}  # hash -> first occurrence index
    input_frame_idx = 0
    output_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame (2 FPS processing)
        if input_frame_idx % frame_skip == 0:
            # Compute frame hash for deduplication
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_tiny = cv2.resize(frame_gray, (16, 9), interpolation=cv2.INTER_NEAREST)
            frame_hash = (frame_tiny.mean(), frame_tiny.std(), frame_tiny[0,0], frame_tiny[8,8], frame_tiny[4,4])
            
            # Check if we've seen this frame before
            if frame_hash not in frame_hash_map:
                # New unique frame - need to process it
                frame_hash_map[frame_hash] = len(frames_to_process)
                frames_to_process.append((input_frame_idx, frame.copy(), frame_hash, output_idx))
            
            output_idx += frame_skip  # Account for duplicates we'll create later
        
        input_frame_idx += 1
    
    cap.release()
    time_reading = time.time() - t_read_start
    
    total_input_frames = input_frame_idx
    total_output_frames = output_idx
    unique_frames = len(frames_to_process)
    duplicate_frames = (total_input_frames // frame_skip) - unique_frames
    
    print(f"Read {total_input_frames} input frames in {time_reading:.2f}s", flush=True)
    print(f"Found {unique_frames} unique frames to process ({duplicate_frames} duplicates skipped)", flush=True)
    print(f"Will output {total_output_frames} frames total", flush=True)
    
    # Step 2: Process unique frames (with or without multiprocessing)
    print("\nStep 2: Processing unique frames...", flush=True)
    t_process_start = time.time()
    
    processed_frames = {}  # frame_idx -> processed_frame
    
    if use_multiprocessing and unique_frames > 1:
        # Prepare arguments for worker processes
        process_args = [
            (frame, roi, inpaint_method, logo_path, idx)
            for idx, frame, _, _ in frames_to_process
        ]
        
        # Process frames in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_frame, args): args[4] for args in process_args}
            
            completed = 0
            for future in as_completed(futures):
                frame_idx, processed_frame = future.result()
                processed_frames[frame_idx] = processed_frame
                completed += 1
                
                if completed % 10 == 0 or completed == unique_frames:
                    print(f"  Processed {completed}/{unique_frames} unique frames", flush=True)
    else:
        # Sequential processing (fallback or single frame)
        flags = cv2.INPAINT_TELEA if inpaint_method.lower() == 'telea' else cv2.INPAINT_NS
        for i, (idx, frame, _, _) in enumerate(frames_to_process):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            inpainted = cv2.inpaint(frame, mask, 3, flags)
            
            if logo_path and os.path.exists(logo_path):
                inpainted = overlay_logo(inpainted, logo_path, (x, y, w, h))
            
            processed_frames[idx] = inpainted
            
            if (i + 1) % 10 == 0 or (i + 1) == unique_frames:
                print(f"  Processed {i + 1}/{unique_frames} unique frames", flush=True)
    
    time_processing = time.time() - t_process_start
    print(f"Processing completed in {time_processing:.2f}s ({unique_frames/time_processing:.2f} frames/sec)", flush=True)
    
    # Step 3: Write all output frames in order
    print("\nStep 3: Writing output frames...", flush=True)
    t_write_start = time.time()
    
    output_idx = 0
    frames_duplicated = 0
    
    # Map input indices to output batches
    input_to_output = {}  # input_idx -> (first_output_idx, count)
    for input_idx, _, frame_hash, output_batch_start in frames_to_process:
        first_idx = frame_hash_map[frame_hash]
        if first_idx != len([x for x in frames_to_process if frames_to_process.index(x) < frames_to_process.index((input_idx, _, frame_hash, output_batch_start))]):
            # This is a duplicate - map to the first occurrence
            first_input_idx = frames_to_process[first_idx][0]
            input_to_output[input_idx] = input_to_output[first_input_idx]
        else:
            # This is the first occurrence
            input_to_output[input_idx] = (output_batch_start, frame_skip)
    
    # Write frames in output order
    for input_idx, _, _, output_batch_start in frames_to_process:
        processed_frame = processed_frames[input_idx]
        
        # Downscale if needed
        if output_width != width or output_height != height:
            processed_frame = cv2.resize(processed_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        
        # Write first frame
        t_encode = time.time()
        first_frame_path = os.path.join(output_frames_dir, f"frame_{output_batch_start:06d}.png")
        ok = cv2.imwrite(first_frame_path, processed_frame)
        if not ok:
            raise RuntimeError(f'Failed to write frame {first_frame_path}')
        time_writing_encode += time.time() - t_encode
        
        # Copy for duplicates to maintain FPS
        t_copy = time.time()
        for dup in range(1, frame_skip):
            duplicate_path = os.path.join(output_frames_dir, f"frame_{output_batch_start + dup:06d}.png")
            shutil.copy2(first_frame_path, duplicate_path)
            frames_duplicated += 1
        time_writing_copy += time.time() - t_copy
        
        output_idx += frame_skip
        
        if output_idx % 90 == 0:
            print(f"  Written {output_idx}/{total_output_frames} output frames", flush=True)
    
    time_writing = time.time() - t_write_start
    print(f"Writing completed in {time_writing:.2f}s", flush=True)
    
    time_total = time.time() - time_total_start
    
    frames_processed = unique_frames
    frames_skipped = duplicate_frames
    
    # Print detailed timing breakdown with flush to ensure it appears in logs
    print("\n" + "="*60, flush=True)
    print("DETAILED TIMING BREAKDOWN", flush=True)
    print("="*60, flush=True)
    print(f"Input frames read: {total_input_frames}", flush=True)
    print(f"Output frames written: {total_output_frames}", flush=True)
    print(f"Unique frames processed: {frames_processed}", flush=True)
    print(f"Duplicate frames reused: {frames_skipped}", flush=True)
    print(f"Frames duplicated (2 FPS): {frames_duplicated}", flush=True)
    if frames_processed > 0:
        reduction = ((total_input_frames // frame_skip - frames_processed) / (total_input_frames // frame_skip) * 100) if total_input_frames > 0 else 0
        print(f"Processing reduction: {reduction:.1f}% (only processed {frames_processed}/{total_input_frames // frame_skip} candidate frames)", flush=True)
    print(flush=True)
    print(f"{'Operation':<25} {'Time (s)':<12} {'% of Total':<12} {'Throughput':<15}", flush=True)
    print("-"*60, flush=True)
    
    operations = [
        ("Frame Reading", time_reading, f"{total_input_frames/time_reading:.1f} fps" if time_reading > 0 else "N/A"),
        ("Frame Processing", time_processing, f"{frames_processed/time_processing:.1f} fps" if time_processing > 0 else "N/A"),
        ("PNG Encoding", time_writing_encode, f"{unique_frames/time_writing_encode:.1f} fps" if time_writing_encode > 0 else "N/A"),
        ("File Copying (dupes)", time_writing_copy, f"{frames_duplicated/time_writing_copy:.1f} fps" if time_writing_copy > 0 else "N/A"),
    ]
    
    for op_name, op_time, throughput in operations:
        pct = (op_time / time_total * 100) if time_total > 0 else 0
        print(f"{op_name:<25} {op_time:<12.2f} {pct:<12.1f} {throughput:<15}", flush=True)
    
    print("-"*60, flush=True)
    print(f"{'TOTAL TIME':<25} {time_total:<12.2f} {'100.0':<12} {total_output_frames/time_total:.1f} fps", flush=True)
    print("="*60, flush=True)
    if use_multiprocessing:
        print(f"\nMultiprocessing: {num_workers} workers, {frames_processed/time_processing:.2f} frames/sec", flush=True)
    print(f"Overall speed: {total_output_frames/time_total:.2f} output frames/second", flush=True)
    print(f"Effective speedup: {(total_input_frames // frame_skip)/frames_processed:.1f}x via deduplication", flush=True)
    print(flush=True)
    
    return float(fps), output_width, output_height  # Return FPS and output dimensions
