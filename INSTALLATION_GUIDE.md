# Installation Guide - Windows

## Issue You're Facing

The `av` package (PyAV) needs to be compiled from source, which requires **Microsoft Visual C++ 14.0 or greater**. This is a common Windows issue.

---

## Quick Fix Solutions

### Option 1: Use install_fix.bat (Recommended)

Simply run the provided batch file:

```cmd
install_fix.bat
```

This will install all dependencies in the correct order and try to get precompiled versions.

---

### Option 2: Manual Installation with Flexible Versions

```cmd
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-simple.txt
```

---

### Option 3: Install Visual Studio Build Tools (If above fails)

1. **Download Visual Studio Build Tools:**
   - Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download "Build Tools for Visual Studio 2022"

2. **Install with C++ Build Tools:**
   - Run the installer
   - Select "Desktop development with C++"
   - Install (requires ~7GB space)

3. **Then install requirements:**
   ```cmd
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Verify Installation

After successful installation, verify it works:

```cmd
venv\Scripts\activate
python -c "import faster_whisper; print('✓ Faster-Whisper OK')"
python -c "import fastapi; print('✓ FastAPI OK')"
python -c "import librosa; print('✓ Librosa OK')"
```

---

## Alternative: Use OpenAI Whisper Instead

If you continue to face issues with faster-whisper (which requires av/PyAV), you can use OpenAI's whisper instead:

### Modified requirements:
```txt
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
openai-whisper>=20231117  # Alternative to faster-whisper
pydub>=0.25.1
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.26.0,<2.0.0
python-dotenv>=1.0.0
```

This requires modifying `app/transcription/whisper_service.py` (I can help with that if needed).

---

## Troubleshooting

### Issue: "av" still fails to install

**Solution:** Try installing a specific precompiled version:
```cmd
pip install av==11.0.0 --only-binary :all:
```

If that fails, try:
```cmd
pip install av --only-binary :all:
```

### Issue: "ffmpeg not found"

**Solution:** Install ffmpeg:
```cmd
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### Issue: Other packages fail

**Solution:** Clear pip cache and retry:
```cmd
pip cache purge
pip install -r requirements-simple.txt
```

---

## Next Steps After Successful Installation

1. **Test the installation:**
   ```cmd
   python -c "from app.audio.audio_loader import AudioLoader; print('✓ All imports working!')"
   ```

2. **Download a sample audio file** and place it in `sample_audio/` folder

3. **Start the FastAPI server:**
   ```cmd
   python -m uvicorn app.main:app --reload
   ```

4. **Access the API docs:** http://localhost:8000/docs

5. **Or test with CLI:**
   ```cmd
   python test_transcription.py sample_audio/your_audio.wav base.en
   ```
