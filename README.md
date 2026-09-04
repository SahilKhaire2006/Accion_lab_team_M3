# Privacy-Preserving Clinical Scribe - Audio & Transcription Module

## Overview

This is the **Team A Module** for the Privacy-Preserving Clinical Scribe and Diagnostic Copilot project. It implements **on-device audio processing and transcription** using local Whisper models, ensuring complete privacy by keeping all audio processing offline.

### Key Features

✅ **Local Whisper Transcription** - Speech-to-text using Faster-Whisper  
✅ **Audio Preprocessing** - Mono conversion, normalization, 16kHz resampling  
✅ **Timestamp Extraction** - Precise start/end times for each segment  
✅ **Structured JSON Output** - Clean, formatted transcripts  
✅ **Speaker Diarization Interface** - Architecture ready for Week 3 implementation  
✅ **FastAPI Endpoint** - Integration point for Team B  
✅ **100% Privacy-Preserving** - All processing happens locally, no cloud APIs

---

## Architecture

```
project/
├── app/
│   ├── audio/
│   │   ├── audio_loader.py          # Load audio files (wav, mp3, etc.)
│   │   └── preprocessing.py         # Normalize, resample, convert to mono
│   │
│   ├── transcription/
│   │   ├── whisper_service.py       # Local Whisper transcription
│   │   └── transcript_formatter.py  # JSON output formatting
│   │
│   ├── speaker/
│   │   └── diarization_interface.py # Speaker diarization (placeholder)
│   │
│   ├── api/
│   │   └── routes.py                # FastAPI endpoints
│   │
│   └── main.py                      # FastAPI application
│
├── sample_audio/                    # Sample audio files
├── output/                          # Generated transcripts
├── test_transcription.py            # CLI test script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Component Interaction

```
Audio File
    ↓
[AudioLoader] → Load and validate
    ↓
[AudioPreprocessor] → Normalize, resample, mono
    ↓
[WhisperService] → Transcribe with timestamps
    ↓
[DiarizationInterface] → Assign speakers (placeholder)
    ↓
[TranscriptFormatter] → Generate structured JSON
    ↓
Output JSON File
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- ffmpeg (required by pydub)

#### Install ffmpeg

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### Installation

1. **Create and activate virtual environment:**

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Verify installation:**

```bash
python -c "import faster_whisper; print('Faster-Whisper installed successfully!')"
```

---

## Usage

### Option 1: FastAPI Server

**Start the server:**

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Test the API:**

```bash
# Check health
curl http://localhost:8000/api/v1/health

# List available models
curl http://localhost:8000/api/v1/models

# Transcribe audio
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@sample_audio/your_audio.wav" \
  -F "model=base.en" \
  -F "session_id=session_001"
```

**Interactive API documentation:**
- Open browser: http://localhost:8000/docs
- Try out endpoints directly from the browser

### Option 2: CLI Test Script

**Run transcription test:**

```bash
python test_transcription.py sample_audio/your_audio.wav base.en
```

**Output will be saved to:**
```
output/session_YYYYMMDD_HHMMSS.json
```

---

## API Reference

### POST /api/v1/transcribe

Transcribe an audio file using local Whisper model.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (file, required): Audio file (wav, mp3, m4a, flac, ogg)
  - `session_id` (string, optional): Session identifier
  - `model` (string, optional): Whisper model (default: "base.en")
  - `save_output` (boolean, optional): Save to output directory (default: true)

**Response:**

```json
{
  "session_id": "session_001",
  "created_at": "2026-09-01T10:00:00Z",
  "model": "whisper-local",
  "segments": [
    {
      "speaker": "UNKNOWN",
      "start": 0.0,
      "end": 3.2,
      "text": "Good morning doctor."
    },
    {
      "speaker": "UNKNOWN",
      "start": 3.3,
      "end": 6.8,
      "text": "How can I help you today?"
    }
  ],
  "metadata": {
    "total_segments": 2,
    "total_duration": 6.8,
    "speaker_diarization": "not_implemented"
  },
  "saved_to": "output/session_20260901_100000.json"
}
```

### GET /api/v1/health

Health check endpoint.

### GET /api/v1/models

List available Whisper models and recommendations.

---

## Whisper Models

| Model      | Size  | Speed    | Accuracy | Use Case           |
|------------|-------|----------|----------|--------------------|
| tiny.en    | 39 MB | Fastest  | Good     | Quick testing      |
| base.en    | 74 MB | Fast     | Better   | **Recommended**    |
| small.en   | 244 MB| Medium   | Best     | High accuracy      |
| medium.en  | 769 MB| Slow     | Excellent| Production quality |

**Configuration:**

```python
# In code
whisper_service = WhisperService(model_name="base.en")

# Via API
curl -X POST /api/v1/transcribe -F "model=small.en" -F "file=@audio.wav"
```

---

## Output Format

### JSON Structure

```json
{
  "session_id": "session_20260901_100000",
  "created_at": "2026-09-01T10:00:00Z",
  "model": "whisper-local",
  "segments": [
    {
      "speaker": "UNKNOWN",
      "start": 0.0,
      "end": 3.2,
      "text": "Transcript text here"
    }
  ],
  "metadata": {
    "total_segments": 10,
    "total_duration": 45.6,
    "speaker_diarization": "not_implemented"
  }
}
```

### Field Descriptions

- **session_id**: Unique identifier for the session
- **created_at**: ISO 8601 timestamp
- **model**: Transcription model used
- **segments**: Array of transcript segments
  - **speaker**: Speaker ID (currently "UNKNOWN", Week 3 will implement)
  - **start**: Start timestamp in seconds
  - **end**: End timestamp in seconds
  - **text**: Transcribed text
- **metadata**: Additional information about the transcript

---

## Speaker Diarization (Week 3)

Currently, all speakers are labeled as `"UNKNOWN"`. The architecture is ready for Week 3 implementation.

### Placeholder Implementation

```python
from app.speaker.diarization_interface import create_diarization_service

diarization = create_diarization_service("placeholder")
speaker_labels = diarization.assign_speakers(audio_data, segments)
```

### Planned Implementation (Week 3)

- Integrate pyannote.audio or similar library
- Speaker clustering and segmentation
- Alignment with transcript segments
- Label speakers as SPEAKER_1, SPEAKER_2, etc.

---

## Testing

### Sample Audio Files

Place sample audio files in `sample_audio/` directory.

**Supported formats:**
- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)

### CLI Testing

```bash
# Test with tiny model (fastest)
python test_transcription.py sample_audio/test.wav tiny.en

# Test with base model (recommended)
python test_transcription.py sample_audio/test.wav base.en

# Test with small model (more accurate)
python test_transcription.py sample_audio/test.wav small.en
```

### API Testing with curl

```bash
# Transcribe audio
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@sample_audio/test.wav" \
  -F "model=base.en" \
  -F "session_id=test_001" \
  -o transcript.json

# View transcript
cat transcript.json | python -m json.tool
```

---

## Project Status

### ✅ Completed (Weeks 1-2)

- [x] Audio loading and validation
- [x] Audio preprocessing (mono, normalize, resample)
- [x] Local Whisper transcription
- [x] Timestamp extraction
- [x] Structured JSON output
- [x] FastAPI endpoints
- [x] Modular architecture
- [x] Speaker diarization interface

### 🚧 Planned (Week 3)

- [ ] Implement actual speaker diarization
- [ ] Speaker clustering and identification
- [ ] Enhanced audio preprocessing (noise reduction)
- [ ] Performance optimization

### 📋 Future Enhancements

- [ ] Real-time streaming transcription
- [ ] Multiple language support
- [ ] Custom vocabulary and medical terminology
- [ ] Audio quality analysis

---

## Troubleshooting

### Issue: "ffmpeg not found"

**Solution:**
```bash
# Install ffmpeg (see Prerequisites section)
# Verify installation
ffmpeg -version
```

### Issue: "Model download slow"

**Solution:**
- Models are downloaded on first use
- Downloaded models are cached locally
- Use `tiny.en` for faster initial testing

### Issue: "Out of memory"

**Solution:**
- Use smaller models (tiny.en or base.en)
- Process shorter audio clips
- Increase system RAM

### Issue: "Transcription is slow"

**Solution:**
- Use GPU if available: `WhisperService(device="cuda")`
- Use smaller models for faster processing
- Consider audio length and model size trade-offs

---

## Team Integration

### For Team B (Clinical NLP)

This module provides a `/transcribe` endpoint that returns structured JSON transcripts. Team B can:

1. Call the transcription endpoint
2. Receive JSON with speaker labels and timestamps
3. Process the transcript for SOAP note generation
4. Extract clinical entities and generate prescriptions

**Integration point:**
```python
# Team B can call
response = requests.post(
    "http://localhost:8000/api/v1/transcribe",
    files={"file": audio_file},
    data={"model": "base.en"}
)
transcript = response.json()
# Process transcript for clinical NLP
```

---

## Privacy & Security

🔒 **All processing happens locally**
- No audio data sent to cloud services
- No external API calls
- Complete HIPAA-compliant data handling
- Audio files can be processed on air-gapped systems

---

## License

Internal project - Team A Module

---

## Contact

For questions about this module, contact Team A.

---

## Appendix: Example Output

```json
{
  "session_id": "session_20260901_143022",
  "created_at": "2026-09-01T14:30:22Z",
  "model": "whisper-local",
  "segments": [
    {
      "speaker": "UNKNOWN",
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you feeling today?"
    },
    {
      "speaker": "UNKNOWN",
      "start": 2.8,
      "end": 5.3,
      "text": "I've been having headaches for the past week."
    },
    {
      "speaker": "UNKNOWN",
      "start": 5.5,
      "end": 8.2,
      "text": "Can you describe the pain? Is it sharp or dull?"
    }
  ],
  "metadata": {
    "total_segments": 3,
    "total_duration": 8.2,
    "speaker_diarization": "not_implemented"
  }
}
```
