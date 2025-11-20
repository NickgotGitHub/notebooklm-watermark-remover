# Deployment Guide - Video Watermark Remover

## ⚠️ Important: Vercel is NOT Compatible

**This application CANNOT run on Vercel** due to fundamental limitations:

### Why Vercel Doesn't Work:

| Requirement | This App | Vercel Limit | Status |
|-------------|----------|--------------|--------|
| File Upload Size | Up to 2GB | 4.5MB | ❌ FAIL |
| Processing Time | 5-60+ seconds | 10-50 seconds | ❌ FAIL |
| Persistent Storage | Required | None | ❌ FAIL |
| FFmpeg Binary | Required | Not available | ❌ FAIL |
| Long-running Process | Required | Serverless only | ❌ FAIL |

### Error 413 Explanation:
- **Your video file** is larger than Vercel's 4.5MB limit
- **Vercel is serverless** - designed for APIs, not file processing
- **No workaround exists** for these fundamental limits

---

## ✅ Recommended Deployment Platforms

### 1. **Railway.app** (⭐ BEST OPTION)

**Why Railway:**
- ✓ Persistent storage (volumes)
- ✓ No timeout limits
- ✓ FFmpeg support
- ✓ Docker support
- ✓ Free tier available
- ✓ Easy deployment from GitHub

**Steps:**
```bash
# 1. Create account at railway.app
# 2. New Project → Deploy from GitHub
# 3. Select this repository
# 4. Add Nixpacks/Dockerfile
# 5. Configure environment variables
# 6. Deploy!
```

**Dockerfile for Railway:**
```dockerfile
FROM python:3.11-slim

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

---

### 2. **Render.com**

**Why Render:**
- ✓ Persistent disk storage
- ✓ Docker support
- ✓ Auto-deploy from GitHub
- ✓ Generous free tier

**render.yaml:**
```yaml
services:
  - type: web
    name: video-watermark-remover
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 5000
    disk:
      name: video-storage
      mountPath: /app/data
      sizeGB: 10
```

---

### 3. **Fly.io**

**Why Fly:**
- ✓ Full VMs (not serverless)
- ✓ Persistent volumes
- ✓ Global edge network
- ✓ Generous free tier

**Deploy:**
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Initialize app
flyctl launch

# Deploy
flyctl deploy
```

---

### 4. **DigitalOcean App Platform**

**Why DigitalOcean:**
- ✓ Simple setup
- ✓ Managed platform
- ✓ Predictable pricing
- ✓ Volume storage

---

### 5. **Google Cloud Run / AWS Lambda (with adjustments)**

**Requires:**
- S3/Cloud Storage for files
- Increased timeout limits
- API Gateway configuration
- More complex setup

---

## 📦 What to Deploy

### Required Files:
- ✓ `app.py`
- ✓ `requirements.txt`
- ✓ `utils/`
- ✓ `templates/`
- ✓ `static/`
- ✓ `Logo.png`

### What NOT to Deploy:
- ❌ `uploads/` (created at runtime)
- ❌ `outputs/` (created at runtime)
- ❌ `temp/` (created at runtime)
- ❌ `.venv/`
- ❌ `__pycache__/`

---

## 🛠️ Quick Start: Railway Deployment

1. **Create Dockerfile** (use the one above)

2. **Push to GitHub:**
```bash
git add .
git commit -m "Prepare for Railway deployment"
git push
```

3. **Deploy on Railway:**
- Go to railway.app
- New Project → Deploy from GitHub
- Select repository
- Railway auto-detects Python
- Click Deploy

4. **Configure:**
- Add volume for persistent storage
- Set PORT environment variable
- Wait for deployment

5. **Access:**
- Railway provides a public URL
- Your app is live! 🎉

---

## 💡 Local Development

**Always works locally:**
```bash
cd video-watermark-remover
source .venv/bin/activate
python app.py
# Open http://127.0.0.1:5000
```

---

## 🆘 Need Help?

**Common Issues:**

1. **FFmpeg not found:**
   - Install: `apt-get install ffmpeg` (Linux)
   - Install: `brew install ffmpeg` (macOS)

2. **OpenCV errors:**
   - Install: `apt-get install libgl1-mesa-glx`

3. **Permission errors:**
   - Ensure uploads/outputs/temp folders exist
   - Check write permissions

---

## 📊 Platform Comparison

| Platform | Setup | Storage | Timeout | FFmpeg | Price/mo |
|----------|-------|---------|---------|--------|----------|
| **Railway** | ⭐⭐⭐⭐⭐ | ✓ | ∞ | ✓ | $5+ |
| **Render** | ⭐⭐⭐⭐ | ✓ | ∞ | ✓ | $7+ |
| **Fly.io** | ⭐⭐⭐ | ✓ | ∞ | ✓ | $0-10 |
| **Vercel** | ⭐⭐⭐⭐⭐ | ❌ | 50s | ❌ | ❌ Won't work |
| **Netlify** | ⭐⭐⭐⭐⭐ | ❌ | 26s | ❌ | ❌ Won't work |

**Recommendation:** Use **Railway.app** for easiest deployment! 🚀

