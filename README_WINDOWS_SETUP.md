# Windows Setup Instructions

## Current Issue

You're getting an error because the `av` (PyAV) package needs to be compiled from source on Windows, which requires Visual Studio C++ Build Tools.

---

## ✅ SOLUTION: Use the Fix Script

### Step 1: Run the installation fix script

```cmd
install_fix.bat
```

This will:
- Install dependencies in the correct order
- Try to get precompiled versions of problematic packages
- Automatically handle the `av` package issue

---

## Alternative: Manual Installation

If the batch script doesn't work, try this:

### Option A: Install without strict version locks

```cmd
venv\Scripts\activate
pip install -r requirements-simple.txt
```

### Option B: Install Visual Studio Build Tools

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Then run: `pip install -r requirements.txt`

---

## Verify Installation

After installation completes, run:

```cmd
venv\Scripts\activate
python quick_test.py
```

This will verify all dependencies are installed correctly.

---

## Running the Application

### Start the FastAPI Server

```cmd
venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

Then visit: http://localhost:8000/docs

### Test with CLI

```cmd
venv\Scripts\activate
python test_transcription.py sample_audio/your_audio.wav base.en
```

---

## Getting Sample Audio

You need actual speech audio for transcription. Options:

1. **Record your own:**
   - Use Windows Voice Recorder
   - Save as WAV or MP3
   - Place in `sample_audio/` folder

2. **Download sample audio:**
   - LibriVox: https://librivox.org/
   - Open Speech Repository: https://www.voiptroubleshooter.com/open_speech/american.html
   - Sample audio: https://filesamples.com/formats/wav

3. **Use YouTube:**
   - Find a doctor-patient consultation video
   - Extract audio using online tools
   - Place in `sample_audio/` folder

---

## Quick Start After Setup

```cmd
# Activate venv
venv\Scripts\activate

# Verify installation
python quick_test.py

# Start server
python -m uvicorn app.main:app --reload

# In another terminal, test the API
curl -X POST http://localhost:8000/api/v1/transcribe -F "file=@sample_audio/test.wav"
```

---

## Common Issues & Solutions

### Issue: "av" fails to install

**Solution 1:** Install precompiled wheel
```cmd
pip install av --only-binary :all:
```

**Solution 2:** Install specific version
```cmd
pip install av==11.0.0
```

**Solution 3:** Install Build Tools (link above)

### Issue: "ffmpeg not found"

**Solution:** Install ffmpeg
```cmd
choco install ffmpeg
```

Or download from: https://ffmpeg.org/download.html

### Issue: Whisper model takes long to download

**Explanation:** Models download on first use. They're cached locally after that.
- tiny.en: ~39 MB
- base.en: ~74 MB (recommended)
- small.en: ~244 MB

---

## Project Structure

```
├── app/
│   ├── audio/              # Audio loading & preprocessing
│   ├── transcription/      # Whisper transcription
│   ├── speaker/            # Diarization (placeholder)
│   ├── api/                # FastAPI routes
│   └── main.py             # FastAPI app
├── sample_audio/           # Place audio files here
├── output/                 # Generated transcripts
├── test_transcription.py   # CLI test script
├── quick_test.py           # Verify installation
└── install_fix.bat         # Windows installation fix
```

---

## Need Help?

Check `INSTALLATION_GUIDE.md` for detailed troubleshooting steps.
