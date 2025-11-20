import json
import logging
import os
import shutil
import subprocess
import uuid

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from utils.video import remove_watermark_roi_to_frames

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTRO_PATH = os.path.join(APP_ROOT, 'public', 'intellibus-outro.mp4')

# Resolve data root:
# 1) honor explicit DATA_ROOT (recommended: /app/data volume on Railway)
# 2) fall back to TMPDIR when present (serverless/ephemeral)
# 3) default to app directory for local dev
DATA_ROOT = (
    os.environ.get('DATA_ROOT')
    or (os.environ.get('TMPDIR', '/tmp') if os.environ.get('VERCEL') else APP_ROOT)
)

UPLOAD_FOLDER = os.path.join(DATA_ROOT, 'uploads')
OUTPUT_FOLDER = os.path.join(DATA_ROOT, 'outputs')
TEMP_FOLDER = os.path.join(DATA_ROOT, 'temp')
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.webm', '.avi', '.mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

app.logger.info(
    "App startup: DATA_ROOT=%s (uploads=%s, outputs=%s, temp=%s) PORT=%s",
    DATA_ROOT,
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    TEMP_FOLDER,
    os.environ.get('PORT'),
)


def is_allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not is_allowed(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    base, ext = os.path.splitext(filename)
    stored_name = f"{base}_{unique_id}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(save_path)

    return jsonify({
        'filename': stored_name,
        'videoUrl': f"/video/{stored_name}"
    })


@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)


@app.route('/download/<path:filename>')
def download_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json(force=True)
    filename = data.get('filename')
    method = data.get('method', 'telea')
    quality = data.get('quality', 'ultra')

    if not filename:
        return jsonify({'error': 'filename is required'}), 400

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found'}), 404

    output_basename = f"processed_{uuid.uuid4().hex}.mp4"
    final_output_path = os.path.join(OUTPUT_FOLDER, output_basename)
    interim_output_path = os.path.join(OUTPUT_FOLDER, f"processed_{uuid.uuid4().hex}_main.mp4")
    frames_dir = os.path.join(TEMP_FOLDER, f"frames_{uuid.uuid4().hex}")

    # Logo path for replacement watermark
    logo_path = os.path.join(APP_ROOT, 'Logo.png')

    try:
        # Auto-detect ROI (None triggers auto-detection in the function)
        fps, width, height = remove_watermark_roi_to_frames(
            input_video_path=input_path,
            output_frames_dir=frames_dir,
            roi=None,  # Auto-detect bottom right watermark
            inpaint_method=method,
            logo_path=logo_path if os.path.exists(logo_path) else None
        )

        # Use the output dimensions from processing (may be downscaled)
        scale_filter = build_scale_filter(width, height)

        encode_frames_and_mux(frames_dir, fps, input_path, interim_output_path, quality, scale_filter)

        # Trim last 4s and append outro clip
        append_outro(interim_output_path, final_output_path, width, height, outro_path=OUTRO_PATH, trim_seconds=4.0)

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        try:
            if os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir)
        except Exception:
            pass
        # Cleanup interim if still present
        try:
            if os.path.exists(interim_output_path):
                os.remove(interim_output_path)
        except Exception:
            pass

    return jsonify({
        'downloadUrl': f"/download/{output_basename}",
        'outputFilename': output_basename,
        'videoUrl': f"/output/{output_basename}"
    })


@app.route('/output/<path:filename>')
def serve_output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


def probe_duration_seconds(path: str) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def probe_resolution(path: str):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return (1920, 1080)
    try:
        info = json.loads(proc.stdout)
        stream = info['streams'][0]
        return (int(stream['width']), int(stream['height']))
    except Exception:
        return (1920, 1080)


def build_scale_filter(src_w: int, src_h: int) -> str:
    # Never upscale; downscale large sources to 1080p max for speed/footprint.
    if src_h > 1080:
        target_h = 1080
        target_w = int(round(src_w * (target_h / src_h)))
        if target_w % 2 == 1:
            target_w += 1
        return f'scale={target_w}:{target_h}:flags=lanczos'
    return 'scale=iw:ih:flags=lanczos'


def encode_frames_and_mux(frames_dir: str, fps: float, source_with_audio: str, output_path: str, quality: str, scale_filter: str) -> None:
    # Quality map: include ultra (lossless x264)
    presets = {
        'fast': ('veryfast', '23', None),
        'balanced': ('medium', '20', None),
        'better': ('slow', '18', None),
        'best': ('veryslow', '16', None),
        'ultra': ('placebo', None, '0')  # qp=0 lossless
    }
    preset, crf, qp = presets.get(quality, ('placebo', None, '0'))

    pattern = os.path.join(frames_dir, 'frame_%06d.png')

    cmd = [
        'ffmpeg', '-y',
        # Limit global threads to avoid OOM on small containers
        '-threads', os.environ.get('FFMPEG_THREADS', '2'),
        '-framerate', str(fps), '-i', pattern,
        '-i', source_with_audio,
        '-map', '0:v:0', '-map', '1:a:0?',
        '-filter:v', scale_filter,
        '-c:v', 'libx264', '-preset', preset,
    ]
    if qp is not None:
        cmd += ['-qp', qp]
    else:
        cmd += ['-crf', crf]

    cmd += [
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ]

    run_ffmpeg(cmd)


def append_outro(main_video_path: str, final_output_path: str, target_w: int, target_h: int, outro_path: str, trim_seconds: float = 4.0) -> None:
    """Trim the last `trim_seconds` from main_video_path and append outro_path."""
    if not os.path.exists(outro_path):
        shutil.move(main_video_path, final_output_path)
        return

    duration = probe_duration_seconds(main_video_path)
    if duration <= trim_seconds + 0.1:
        shutil.move(main_video_path, final_output_path)
        return

    keep_duration = max(duration - trim_seconds, 0.1)
    filter_complex = (
        f"[0:v]trim=0:{keep_duration},setpts=PTS-STARTPTS[v0];"
        f"[0:a]atrim=0:{keep_duration},asetpts=PTS-STARTPTS[a0];"
        f"[1:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h},setsar=1[v1];"
        f"[1:a]aresample=async=1[a1];"
        f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', main_video_path,
        '-i', outro_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]', '-map', '[outa]',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        final_output_path
    ]

    run_ffmpeg(cmd)


def run_ffmpeg(cmd: list) -> None:
    env = os.environ.copy()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        # Log full context server-side for diagnostics
        app.logger.error(
            "FFmpeg failed (exit %s)\ncmd: %s\nstdout:\n%s\nstderr:\n%s",
            proc.returncode,
            " ".join(cmd),
            proc.stdout,
            proc.stderr,
        )
        # Return a trimmed but informative error to the client
        err_snippet = proc.stderr[:1600] if proc.stderr else "no stderr"
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}). {err_snippet}")


if __name__ == '__main__':
    # Bind to 0.0.0.0 so platforms like Railway can expose the app,
    # and respect PORT env var if provided.
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
