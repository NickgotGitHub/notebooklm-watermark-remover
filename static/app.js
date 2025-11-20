const dropzone = document.getElementById('dropzone');
const browseBtn = document.getElementById('browseBtn');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const uploadProgress = document.getElementById('uploadProgress');
const uploadProgressBar = document.getElementById('uploadProgressBar');

const methodSelect = document.getElementById('method');
const qualitySelect = document.getElementById('quality');
const processBtn = document.getElementById('processBtn');
const processStatus = document.getElementById('processStatus');
const processProgress = document.getElementById('processProgress');
const downloadLink = document.getElementById('downloadLink');
const toasts = document.getElementById('toasts');

const comparisonCard = document.getElementById('comparisonCard');
const videoOriginal = document.getElementById('videoOriginal');
const videoProcessed = document.getElementById('videoProcessed');
const compareSlider = document.getElementById('compareSlider');
const compareOverlay = document.getElementById('compareOverlay');
const compareDivider = document.getElementById('compareDivider');
const playToggle = document.getElementById('playToggle');
const themeToggle = document.getElementById('themeToggle');
const root = document.documentElement;

let uploadedFilename = null;
let originalVideoUrl = null;
let isPlaying = false;

// Toasts
function toast(message, type = 'success', timeout = 3000) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = message;
  toasts.appendChild(t);
  setTimeout(() => { t.remove(); }, timeout);
}

// Theme toggle
const storedTheme = localStorage.getItem('theme');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
function setTheme(theme) {
  root.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  if (themeToggle) {
    themeToggle.textContent = theme === 'dark' ? '🌙' : '☀️';
    themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }
}
setTheme(storedTheme || (prefersDark ? 'dark' : 'light'));
themeToggle?.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  setTheme(next);
});

// Compare slider - use clip-path to avoid resizing the processed video
function updateSlider(val) {
  if (!compareOverlay) return;
  const pct = Math.min(100, Math.max(0, Number(val) || 0));
  // Use clip-path instead of width so video doesn't resize
  compareOverlay.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  if (compareDivider) {
    compareDivider.style.left = `${pct}%`;
  }
}
if (compareSlider) {
  updateSlider(compareSlider.value || 55);
  compareSlider.addEventListener('input', (e) => updateSlider(e.target.value));
}

function pauseBoth() {
  isPlaying = false;
  playToggle?.setAttribute('aria-label', 'Play preview');
  if (playToggle) playToggle.textContent = '▶';
  videoOriginal?.pause();
  videoProcessed?.pause();
}

async function playBoth() {
  if (!videoOriginal || !videoProcessed) return;
  try {
    // Play processed video with audio, keep original muted for visual sync
    videoProcessed.muted = false;
    videoOriginal.muted = true;
    videoOriginal.currentTime = videoProcessed.currentTime;
    await videoProcessed.play();
    await videoOriginal.play();
    isPlaying = true;
    playToggle?.setAttribute('aria-label', 'Pause preview');
    if (playToggle) playToggle.textContent = '❚❚';
  } catch (e) {
    // Autoplay might be blocked, fallback to manual play
    isPlaying = false;
  }
}

function syncCurrentTime() {
  if (!videoOriginal || !videoProcessed) return;
  const delta = Math.abs(videoOriginal.currentTime - videoProcessed.currentTime);
  if (delta > 0.12) {
    videoProcessed.currentTime = videoOriginal.currentTime;
  }
}

videoOriginal?.addEventListener('timeupdate', syncCurrentTime);
videoOriginal?.addEventListener('ended', pauseBoth);
videoProcessed?.addEventListener('ended', pauseBoth);

playToggle?.addEventListener('click', () => {
  if (!videoOriginal?.src || !videoProcessed?.src) return;
  if (isPlaying) pauseBoth();
  else playBoth();
});

// Dropzone
['dragenter', 'dragover'].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation();
  dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation();
  dropzone.classList.remove('dragover');
}));
dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (!files || !files.length) return;
  fileInput.files = files;
  handleFileSelect();
});

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

async function handleFileSelect() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = `Selected: ${file.name}`;
  uploadProgress.hidden = false;
  uploadProgressBar.style.width = '0%';
  processBtn.disabled = true;

  const form = new FormData();
  form.append('video', file);

  try {
    const data = await uploadWithProgress('/upload', form, (pct) => {
      uploadProgressBar.style.width = `${pct}%`;
    });
    uploadedFilename = data.filename;
    originalVideoUrl = data.videoUrl;
    uploadStatus.textContent = `✓ ${file.name} uploaded`;
    toast('Video uploaded successfully!');
    processBtn.disabled = false;
  } catch (e) {
    uploadStatus.textContent = 'Upload failed';
    toast(e.message || 'Upload failed', 'error');
    processBtn.disabled = true;
  } finally {
    setTimeout(() => { uploadProgress.hidden = true; }, 500);
  }
}

function uploadWithProgress(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress(pct);
      }
    };
    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new Error(data.error || 'Upload failed'));
      } catch {
        reject(new Error('Invalid server response'));
      }
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send(form);
  });
}

// Process
processBtn.addEventListener('click', async () => {
  if (!uploadedFilename) { 
    toast('Please upload a video first', 'error'); 
    return; 
  }

  processStatus.textContent = 'Processing... This may take a few minutes.';
  processProgress.hidden = false;
  processBtn.disabled = true;
  comparisonCard.hidden = true;

  try {
    const res = await fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        filename: uploadedFilename, 
        method: methodSelect.value, 
        quality: qualitySelect.value || 'fast' 
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Processing failed');
    
    processStatus.textContent = '✓ Processing complete!';
    
    // Hide upload card, show comparison
    document.getElementById('uploadCard').hidden = true;
    
    // Show before/after comparison
    videoOriginal.src = originalVideoUrl;
    videoProcessed.src = data.videoUrl;
    videoOriginal.load();
    videoProcessed.load();
    videoProcessed.muted = false;
    videoOriginal.muted = true;
    videoOriginal.currentTime = 0;
    videoProcessed.currentTime = 0;
    downloadLink.href = data.downloadUrl;
    downloadLink.download = data.outputFilename;
    downloadLink.hidden = false;
    comparisonCard.hidden = false;
    pauseBoth();
    updateSlider(compareSlider?.value || 100);
    
    toast('Watermark removed successfully! ✨', 'success', 4000);
  } catch (e) {
    processStatus.textContent = '✗ Processing failed';
    toast(e.message || 'Processing failed', 'error', 5000);
  } finally {
    processProgress.hidden = true;
    processBtn.disabled = false;
  }
});

// Handle "New Video" button
document.getElementById('newVideoBtn').addEventListener('click', () => {
  comparisonCard.hidden = true;
  document.getElementById('uploadCard').hidden = false;
  fileInput.value = '';
  uploadStatus.textContent = '';
  videoOriginal.src = '';
  videoProcessed.src = '';
  pauseBoth();
  uploadedFilename = null;
  processStatus.textContent = '';
  processBtn.disabled = true;
  downloadLink.hidden = true;
  if (compareSlider) {
    compareSlider.value = 100;
    updateSlider(compareSlider.value);
  }
});
