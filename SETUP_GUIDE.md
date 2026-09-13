# Setup Guide — Clinical Scribe Project

Follow these steps to get the project running on your machine from scratch.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/SahilKhaire2006/Accion_lab_team_M3.git
cd Accion_lab_team_M3
```

---

## Step 2 — Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This will install FastAPI, Whisper, pyannote, sounddevice, Groq client, and all other required packages.

---

## Step 4 — Install ffmpeg (Required)

ffmpeg is required for audio format conversion (MP3, M4A, etc.).

**Option A — Using Chocolatey:**
```bash
choco install ffmpeg
```

**Option B — Manual download:**
Download from: https://ffmpeg.org/download.html
Extract it and add the `bin` folder to your system PATH.

**Verify installation:**
```bash
ffmpeg -version
```

---

## Step 5 — Get HuggingFace Token (For Speaker Diarization)

Speaker diarization (who spoke when) uses a model from HuggingFace that requires a free account and token.

### 5a — Create Account
Go to: https://huggingface.co/join

### 5b — Generate Token
1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Give it any name, set role to **Read**
4. Click **Generate** and copy the token
   - Format will be: `hf_xxxxxxxxxxxxxxxxxxxxxxxx`

### 5c — Accept Model Terms (Important!)
You MUST visit both these links and click **"Agree and access repository"**:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

> Without accepting these terms, diarization will not work even if your token is correct.

---

## Step 6 — Get Groq API Key (For LLM Role Correction)

Groq is used to verify and correct Doctor/Patient role assignments using an LLM.

1. Create a free account at: https://console.groq.com
2. In the left sidebar click **"API Keys"**
3. Click **"Create API Key"**
4. Copy the key
   - Format will be: `gsk_xxxxxxxxxxxxxxxxxxxxxxxx`

---

## Step 7 — Create Your .env File

A `.env.example` file is already in the repo. Copy it:

```bash
copy .env.example .env
```

Now open `.env` and fill in your actual tokens:

```env
HF_TOKEN=hf_yahan_apna_token_daalo
GROQ_API_KEY=gsk_yahan_apna_key_daalo
GROQ_MODEL=openai/gpt-oss-20b
GROQ_FALLBACK_MODELS=openai/gpt-oss-120b,groq/compound
```

> **Warning:** Never push your `.env` file to GitHub. It contains private tokens.
> It is already listed in `.gitignore` so it will not be committed accidentally.

---

## Step 8 — Start the Server

```bash
python -m uvicorn app.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> **First run note:** The diarization model (~300MB) will download automatically on the first recording. This is normal — it only happens once and is cached locally after that.

---

## Step 9 — Open in Browser

```
http://localhost:8000
```

- Click **"Start Conversation"** to begin recording
- Speak into your microphone (ideally two people — doctor and patient)
- Click **"End Conversation"** when done
- Wait for processing (30–60 seconds depending on audio length)
- A popup will appear with the transcript file location
- Click **"View in Chat Format"** to see the Doctor/Patient chat view

---

## Quick Reference — All Links

| Resource | Link |
|---|---|
| GitHub Repo | https://github.com/SahilKhaire2006/Accion_lab_team_M3.git |
| HuggingFace Signup | https://huggingface.co/join |
| HuggingFace Tokens | https://huggingface.co/settings/tokens |
| Diarization Model Terms | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| Segmentation Model Terms | https://huggingface.co/pyannote/segmentation-3.0 |
| Groq Console | https://console.groq.com |
| ffmpeg Download | https://ffmpeg.org/download.html |
| Local Server | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## Common Issues

**"HF_TOKEN not set" warning on startup**
- Check your `.env` file has the actual token, not the placeholder text
- Restart the server after editing `.env`

**All speakers show as UNKNOWN**
- You forgot to accept model terms on both HuggingFace links (Step 5c)
- Or your HF_TOKEN is incorrect

**ffmpeg not found error**
- Make sure ffmpeg is installed and added to PATH
- Restart your terminal after installing

**pip install fails on `pyannote.audio`**
- Install PyTorch separately first:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- Then run `pip install -r requirements.txt` again

**Port 8000 already in use**
```bash
python -m uvicorn app.main:app --reload --port 8001
```
Then open: http://localhost:8001
