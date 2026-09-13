# Privacy-Preserving Clinical Scribe — Complete Project Documentation

> **Team A Module | EDI Accion Lab | VIT 3rd Semester**
> Audio Processing, Transcription, Speaker Diarization & Role Mapping

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Tech Stack & Libraries](#3-tech-stack--libraries)
4. [Project Architecture](#4-project-architecture)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
   - 5.1 [Audio Loading](#51-audio-loading--appaudioaudio_loaderpy)
   - 5.2 [Audio Preprocessing](#52-audio-preprocessing--appaudiopreprocessingpy)
   - 5.3 [Whisper Transcription](#53-whisper-transcription--apptranscriptionwhisper_servicepy)
   - 5.4 [Speaker Diarization](#54-speaker-diarization--appservicesdiarizationpy)
   - 5.5 [Merge Logic](#55-merge-logic--appservicesmergepy)
   - 5.6 [Role Mapping](#56-role-mapping--appservicesrolespy)
   - 5.7 [LLM Role Correction](#57-llm-role-correction--appservicesllm_correctionpy)
   - 5.8 [Garbled Text Cleanup](#58-garbled-text-cleanup--appservicestext_cleanuppy)
   - 5.9 [API Routes](#59-api-routes--appapiliveroutes_py--appapiRoutespy)
   - 5.10 [Frontend Dashboard](#510-frontend-dashboard--staticindexhtml)
6. [Full Processing Pipeline](#6-full-processing-pipeline)
7. [API Reference](#7-api-reference)
8. [Output JSON Format](#8-output-json-format)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)
10. [Bug Fixes Implemented](#10-bug-fixes-implemented)
11. [Feature Additions](#11-feature-additions)
12. [Team B Integration](#12-team-b-integration)
13. [Privacy & Security Design](#13-privacy--security-design)
14. [Development Timeline](#14-development-timeline)
15. [Current Project Status](#15-current-project-status)
16. [Q&A — Questions Your Mentor May Ask](#16-qa--questions-your-mentor-may-ask)
17. [Questions You Can Ask Your Mentor](#17-questions-you-can-ask-your-mentor)

---

## 1. Project Overview

This project is **Team A's module** in a larger system called the **Privacy-Preserving Clinical Scribe and Diagnostic Copilot**. The overall system aims to automate clinical documentation — specifically converting doctor-patient conversations into structured medical notes without sending any sensitive audio or text to external cloud services.

Team A is responsible for the **audio pipeline**: recording live audio, processing it, transcribing speech to text, identifying who is speaking (doctor vs patient), and delivering a structured JSON transcript to Team B (Clinical NLP team), who then generates SOAP notes and prescriptions.

The key design philosophy is **zero cloud dependency** — all models run locally on the machine, making it compliant with healthcare data privacy laws like HIPAA.

---

## 2. Problem Statement

In a typical clinic:
- Doctors spend 30–40% of their time on documentation, not patient care.
- Manual transcription is expensive and slow.
- Cloud-based speech services (like Google Speech API or AWS Transcribe) are not HIPAA-compliant by default and require patient data to leave the hospital's network.

Our solution:
- Captures real-time doctor-patient conversation audio.
- Runs all AI models **locally** (on-device) for complete privacy.
- Produces a structured, role-labeled transcript that can be fed to downstream NLP systems for SOAP note generation.

---

## 3. Tech Stack & Libraries

| Library / Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Core language |
| **FastAPI** | 0.109.0 | REST API framework |
| **Uvicorn** | 0.27.0 | ASGI server to run FastAPI |
| **faster-whisper** | 1.0.0 | Local Whisper model for speech-to-text (CTranslate2 backend, faster than OpenAI Whisper) |
| **pyannote.audio** | 3.1.1 | Neural speaker diarization (who spoke when) |
| **torch** | >=2.0.0 | PyTorch — backend for pyannote.audio |
| **pydub** | 0.25.1 | Audio format conversion (MP3, M4A, etc.) |
| **librosa** | 0.10.1 | Audio resampling and analysis |
| **soundfile** | 0.12.1 | Fast WAV/FLAC loading and saving |
| **sounddevice** | 0.4.6 | Real-time microphone capture (live recording) |
| **numpy** | 1.26.3 | Numerical operations on audio arrays |
| **groq** | >=0.4.0 | Groq Cloud API client for LLM role correction |
| **python-dotenv** | 1.0.0 | Load `.env` config file |
| **python-multipart** | 0.0.6 | File upload support in FastAPI |
| **pytest** | 7.4.3 | Unit testing framework |
| **ffmpeg** | (bundled) | Audio codec support for pydub (MP3, M4A, AAC) |

### Why faster-whisper over standard Whisper?
OpenAI's original Whisper library is slow on CPU. `faster-whisper` uses the CTranslate2 inference engine with INT8 quantization, making it **2–4x faster** on CPU with the same accuracy. It also supports word-level timestamps natively, which we use for segment splitting in the merge step.

### Why pyannote.audio?
It is the state-of-the-art open-source speaker diarization library based on neural networks. Version 3.1 supports speaker count constraints (min/max speakers), which we use to enforce the two-speaker clinical conversation assumption.

### Why Groq for LLM?
Groq provides extremely fast inference (tokens/second) for open models. We use it for two purposes:
1. Role refinement — correcting wrong Doctor/Patient assignments by reading conversation context.
2. Garbled text cleanup — fixing repeated-word Whisper hallucinations using surrounding context.

---

## 4. Project Architecture

```
EDI_Accion_lab/
│
├── app/
│   ├── main.py                         ← FastAPI app entry point, CORS, routing
│   │
│   ├── api/
│   │   ├── routes.py                   ← File upload transcription endpoint
│   │   └── live_routes.py              ← Live recording + full pipeline (7-step)
│   │
│   ├── audio/
│   │   ├── audio_loader.py             ← Load WAV/MP3/M4A/FLAC/OGG files
│   │   └── preprocessing.py            ← Convert to mono, resample to 16kHz, normalize
│   │
│   ├── transcription/
│   │   ├── whisper_service.py          ← Faster-Whisper local model (word timestamps)
│   │   └── transcript_formatter.py     ← Format output to structured JSON
│   │
│   ├── speaker/
│   │   └── diarization_interface.py    ← Early placeholder interface (Week 1-2)
│   │
│   └── services/
│       ├── diarization.py              ← pyannote.audio pipeline (singleton)
│       ├── merge.py                    ← Align Whisper segments with speaker turns
│       ├── roles.py                    ← Map SPEAKER_00/01 → Doctor/Patient
│       ├── llm_correction.py           ← Groq LLM role refinement (QA step)
│       └── text_cleanup.py             ← Detect & fix garbled Whisper segments
│
├── static/
│   ├── index.html                      ← Live recording dashboard (UI)
│   └── transcript.html                 ← Chat-style transcript viewer
│
├── output/                             ← Saved session JSON files
├── sample_audio/                       ← Recorded audio files
├── ffmpeg/                             ← Bundled ffmpeg binary (Windows)
│
├── .env                                ← HF_TOKEN, GROQ_API_KEY (not committed)
├── .env.example                        ← Template for environment setup
├── requirements.txt                    ← Python dependencies
└── tests/
    ├── test_merge.py
    ├── test_roles.py
    └── test_swap_endpoint.py
```

### Data Flow (High-Level)

```
Microphone Input
      ↓
[sounddevice] → raw 48kHz stereo PCM audio chunks
      ↓
[AudioLoader] → load from saved WAV file
      ↓
[AudioPreprocessor] → mono, 16kHz, normalized float32
      ↓
[WhisperService] → segments with start/end/text/words[]
      ↓
[detect_garbled_segments] → flag repeated-word hallucinations
      ↓
[cleanup_garbled_segments] → LLM-cleaned text in text_cleaned field
      ↓
[diarize()] → speaker turns: [{start, end, speaker: "SPEAKER_00"}, ...]
      ↓
[merge_transcript_with_speakers()] → each segment gets a speaker label
      ↓
[label_roles()] → SPEAKER_00 → "Doctor", SPEAKER_01 → "Patient"
      ↓
[refine_roles_with_llm()] → Groq LLM validates/corrects role assignments
      ↓
[session JSON] → saved to output/session_YYYYMMDD_HHMMSS.json
      ↓
[Team B API] → /team-b/consultation/{id} delivers structured data
```

---

## 5. Module-by-Module Breakdown

### 5.1 Audio Loading — `app/audio/audio_loader.py`

**Purpose:** Load audio files of any supported format into a NumPy float32 array.

**Supported Formats:** `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`

**How it works:**
- First tries `soundfile` (fast, supports WAV/FLAC/OGG natively).
- Falls back to `pydub` for compressed formats (MP3, M4A, AAC) that require ffmpeg.
- Normalizes the sample values to the `[-1.0, 1.0]` range (float32).
- Handles 16-bit and 32-bit audio automatically.
- Reports duration, sample rate, and channel count on load.

**Key Method:** `load(file_path) → (np.ndarray, int)` returns audio data and sample rate.

**Also provides:** `save(audio_data, output_path, sample_rate)` to write processed audio back to disk.

**Why this matters:** Whisper requires float32 mono 16kHz audio. We must handle whatever format arrives from the microphone or the user's file upload.

---

### 5.2 Audio Preprocessing — `app/audio/preprocessing.py`

**Purpose:** Convert raw audio into the exact format Whisper expects.

**Three-step pipeline:**

1. **Mono Conversion** — Stereo recordings (2 channels) are averaged: `np.mean(audio, axis=1)`. Microphone captures 2-channel audio at 48kHz; Whisper only understands single-channel.

2. **Resampling to 16kHz** — Uses `librosa.resample()`. The microphone captures at 48kHz (default OS audio), but Whisper was trained on 16kHz audio. Feeding 48kHz directly would degrade accuracy.

3. **Amplitude Normalization** — Divides by the max absolute value so the signal fills the `[-1, 1]` range. This compensates for quiet recordings where the speaker was far from the mic.

**Key Class:** `AudioPreprocessor(target_sample_rate=16000)`

**Key Method:** `preprocess(audio_data, original_sample_rate) → np.ndarray`

---

### 5.3 Whisper Transcription — `app/transcription/whisper_service.py`

**Purpose:** Convert preprocessed audio to text with precise timestamps using a local AI model.

**Model Used:** `faster-whisper` (CTranslate2 backend)

**Available Models:**

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| tiny.en | 39 MB | Fastest | Good |
| base.en | 74 MB | Fast | Better ✓ Default |
| small.en | 244 MB | Medium | Best |
| medium.en | 769 MB | Slow | Excellent |

**Key Parameters used in transcription:**
- `beam_size=5` — Searches 5 candidates per step for better accuracy.
- `vad_filter=True` — Voice Activity Detection skips silent parts, reducing hallucinations.
- `vad_parameters=dict(min_silence_duration_ms=500)` — Treats gaps >500ms as silence.
- `word_timestamps=True` — Extracts per-word start/end times, critical for segment splitting in the merge step.
- `compute_type="int8"` — INT8 quantization for faster CPU inference with minimal accuracy loss.

**Output per segment:**
```json
{
  "start": 0.54,
  "end": 3.20,
  "text": "Good morning, how are you feeling today?",
  "words": [
    {"start": 0.54, "end": 0.72, "word": "Good"},
    {"start": 0.72, "end": 1.10, "word": "morning"}
  ]
}
```

**Why word timestamps?** When Whisper produces a 12-second segment that actually contains both the doctor and patient speaking, we need to split it at the right word boundary. Without word timestamps, we'd have to assign the entire segment to one speaker incorrectly.

---

### 5.4 Speaker Diarization — `app/services/diarization.py`

**Purpose:** Identify "who spoke when" in the audio without knowing anything about the speakers in advance.

**Library used:** `pyannote.audio 3.1.1` — state-of-the-art neural speaker diarization.

**Model:** `pyannote/speaker-diarization-3.1` from HuggingFace (requires free account + token + accepting model terms).

**How it works:**
1. The model uses a neural network to segment audio into speech regions.
2. Speaker embeddings (voice fingerprints) are extracted per segment.
3. Clustering groups segments by speaker identity.
4. Output: a sequence of turns with `{start, end, speaker: "SPEAKER_00"}`.

**Speaker Constraints:** We pass `min_speakers=2, max_speakers=2` because a clinical consultation always has exactly two participants. This constrains the clustering algorithm and improves accuracy.

**Singleton Pattern:** The pipeline is loaded once and cached globally (`_pipeline` variable). Loading pyannote takes 3–10 seconds; reloading on every request would be unacceptable.

**Device Detection:** Automatically uses CUDA (GPU) if available, otherwise falls back to CPU.

**Graceful Fallback:** If HF_TOKEN is missing or the model fails to load, all segments are labeled `UNKNOWN` instead of crashing the server.

**Output example:**
```python
[
  {"start": 0.0,  "end": 4.5,  "speaker": "SPEAKER_00"},
  {"start": 4.8,  "end": 9.2,  "speaker": "SPEAKER_01"},
  {"start": 9.5,  "end": 15.0, "speaker": "SPEAKER_00"},
]
```

---

### 5.5 Merge Logic — `app/services/merge.py`

**Purpose:** Align Whisper's text segments (which know the words) with pyannote's speaker turns (which know who spoke), producing segments that have both text AND speaker identity.

**The core problem:** Whisper and pyannote process audio independently. Whisper may create one long segment for text that spans two speakers. Pyannote's turn boundaries don't align with Whisper's segment boundaries. We must reconcile them.

**Algorithm — Maximum Overlap Assignment:**
For each Whisper segment, calculate the temporal overlap with every diarization turn. Assign the speaker whose turn overlaps the most with the segment.

```python
overlap = max(0, min(seg_end, turn_end) - max(seg_start, turn_start))
```

**Long Segment Splitting (>8 seconds):**
If a segment is longer than 8 seconds AND overlaps multiple speakers significantly (>15% of segment duration each), we split it at diarization boundaries using word-level timestamps from Whisper.

**Post-merge sanity check:**
Automatically detects if the merge collapsed multiple speakers into one (a known bug). Prints a warning:  
`⚠️ Merge collapsed 2 speakers into 1 — possible merge bug!`

**Debug logging:** Extensive console output shows every segment's overlap calculation, which turn won, and the final speaker distribution — critical for diagnosing the speaker collapse bug.

---

### 5.6 Role Mapping — `app/services/roles.py`

**Purpose:** Convert abstract speaker labels (`SPEAKER_00`, `SPEAKER_01`) to meaningful clinical roles (`Doctor`, `Patient`).

**Strategy:** In a clinical consultation, the doctor typically speaks first (greets the patient). So we map the **first chronologically occurring speaker** to "Doctor" and the second to "Patient".

**Configurable:** The parameter `first_speaker_is` can be set to `"Patient"` if needed (e.g., if the patient calls the doctor).

**Role Swap feature:** The `/swap-roles/{session_id}` endpoint allows swapping DOCTOR ↔ PATIENT post-hoc if the automatic assignment was wrong. It is idempotent (swapping twice returns to original).

**Output:** Each segment gets a `"role"` field: `"Doctor"` or `"Patient"`.

---

### 5.7 LLM Role Correction — `app/services/llm_correction.py`

**Purpose:** Use a large language model to verify and correct role assignments by understanding the semantic meaning of each utterance.

**Why needed?** Diarization only knows timing patterns — it cannot tell if "What brings you in today?" is more likely the doctor's line. The LLM can.

**How it works:**
1. Build a prompt containing all segments with their current speaker label and text.
2. Instruct the LLM to output a JSON object with a `"roles"` array of corrected labels.
3. Apply the corrections, tracking which segments were changed vs confirmed.

**Models tried (primary → fallback):**
1. `openai/gpt-oss-20b` (primary, fast)
2. `openai/gpt-oss-120b` (fallback, more capable)
3. `groq/compound` (last resort, agentic model)

These run through the Groq API (ultra-fast inference).

**Rules given to the LLM:**
- "What brings you in today?" → Doctor
- "I've been having headaches" → Patient
- "I'll prescribe..." → Doctor
- "Thank you, doctor" → Patient
- Natural turn-taking should be maintained.

**JSON Mode:** Uses `response_format={"type": "json_object"}` to force structured output, reducing parse failures.

**Tolerant Parsing:** Handles markdown code fences (` ```json ``` `), alternative key names (`labels`, `result`, `output`), and validates all roles are "Doctor" or "Patient" before applying.

**Each segment gets a `role_source` field:**
- `"llm_confirmed"` — LLM agreed with diarization
- `"llm_corrected"` — LLM changed the role
- `"diarization_only"` — LLM step failed, using raw diarization result

---

### 5.8 Garbled Text Cleanup — `app/services/text_cleanup.py`

**Purpose:** Detect and fix Whisper ASR hallucinations — most commonly repeated-word patterns like *"Don't. Don't. Don't. Don't."* which Whisper generates in silence or low-SNR audio.

**Step 2.5 in the pipeline** — runs between transcription and diarization.

**Detection heuristics (two rules):**

1. **Repeated Word Pattern:** Any single word repeated 3+ consecutive times.  
   Example: `"the the the the patient"` → flagged.

2. **Low Unique-Word Ratio:** If the ratio of unique words to total words is below 0.4 for segments with 5+ words.  
   Example: `"pain pain pain bad pain"` → 2 unique / 5 total = 0.4 → borderline flagged.

**Cleanup process:**
- Only flagged segments are sent to the LLM (opt-in, targeted).
- Context from the previous and next segments is also sent for better reconstruction.
- LLM returns a `cleaned_text` field.

**Safety guarantee — original text is NEVER overwritten:**
```json
{
  "text": "Don't. Don't. Don't. Don't.",
  "text_cleaned": "I don't understand.",
  "text_cleanup_applied": true,
  "likely_garbled": true,
  "garbled_reason": "Repeated word pattern: 'Don't'"
}
```

**UI representation:** The `transcript.html` viewer shows cleaned text with a green "CLEANED" badge. Hovering shows the original Whisper output as a tooltip.

---

### 5.9 API Routes — `app/api/live_routes.py` & `app/api/routes.py`

**Live Recording Endpoints (`/api/v1/live/`):**

| Endpoint | Method | Purpose |
|---|---|---|
| `/start-recording` | POST | Opens microphone stream via sounddevice |
| `/stop-recording` | POST | Stops stream, saves WAV, runs full 7-step pipeline |
| `/transcript/{session_id}` | GET | Retrieve saved session JSON by ID |
| `/swap-roles/{session_id}` | POST | Swap Doctor/Patient labels in a saved session |
| `/recording-status` | GET | Check if recording is currently active |
| `/team-b/consultation/{id}` | GET | Team B integration endpoint (clean format) |

**File Upload Endpoint (`/api/v1/`):**

| Endpoint | Method | Purpose |
|---|---|---|
| `/transcribe` | POST | Upload an audio file, get transcript JSON |
| `/health` | GET | Health check |
| `/models` | GET | List available Whisper models |

**Recording State Management:**
The live recording uses a global `recording_state` dictionary to hold the `sounddevice.InputStream` object, the audio queue, and the current status. Audio chunks flow from the microphone callback into a `queue.Queue()` and are collected when recording stops.

---

### 5.10 Frontend Dashboard — `static/index.html`

**Purpose:** A browser-based UI for the doctor/nurse to control recording and view the transcript.

**Features:**
- "Start Conversation" button → calls `/api/v1/live/start-recording`
- "End Conversation" button → calls `/api/v1/live/stop-recording`
- Pulsing red status indicator while recording is active
- Loading spinner during transcription processing
- Success popup with session file path and link to chat viewer
- Inline transcript display showing speaker, timestamp, and text per segment

**Transcript Viewer (`static/transcript.html`):**
- Chat-style layout: Doctor on left (blue bubbles), Patient on right (purple bubbles)
- Timestamps per message
- Session metadata header (ID, duration, segment count)
- "Swap Doctor/Patient" button that calls the swap-roles API

**Technology:** Pure HTML/CSS/JavaScript (no framework). The gradient UI uses CSS animations for the recording pulse effect.

---

## 6. Full Processing Pipeline

When a user clicks "End Conversation", this 7-step pipeline runs automatically:

```
STEP 1: Load & Preprocess Audio
  ├── AudioLoader.load(wav_path) → (audio_array, 48000)
  └── AudioPreprocessor.preprocess() → mono, 16kHz, normalized

STEP 2: Whisper Transcription
  ├── WhisperService.load_model() [cached after first call]
  └── WhisperService.transcribe() → segments with word timestamps

STEP 2.5: Garbled Text Detection & Cleanup
  ├── detect_garbled_segments() → flags repeated-word hallucinations
  └── cleanup_garbled_segments() → LLM fixes flagged segments (if any)

STEP 3: Speaker Diarization
  ├── diarize(audio_path, min_speakers=2, max_speakers=2)
  └── Returns: [{start, end, speaker: "SPEAKER_00"}, ...]
  [Fallback: UNKNOWN if HF_TOKEN not set or diarization fails]

STEP 4: Merge Transcript with Speakers
  ├── merge_transcript_with_speakers(whisper_segs, diarization_turns)
  ├── Max-overlap algorithm assigns speaker to each segment
  └── Long segments (>8s) split at word boundaries if possible

STEP 5: Label Roles
  └── label_roles(merged_segs, first_speaker_is="Doctor")
      → SPEAKER_00 → "Doctor", SPEAKER_01 → "Patient"

STEP 6: LLM Role Refinement
  └── refine_roles_with_llm(segments)
      → Groq API validates/corrects role assignments by context
      → Adds role_source: "llm_confirmed" or "llm_corrected"

STEP 7: Save Session JSON
  └── output/session_YYYYMMDD_HHMMSS.json
      (Team B compatible format)
```

---

## 7. API Reference

### POST /api/v1/live/stop-recording

Stops the microphone recording and runs the full pipeline.

**Response:**
```json
{
  "status": "recording_stopped",
  "message": "Recording stopped and transcription complete",
  "audio_file": "sample_audio/live_recording_20260908_160043.wav",
  "duration": 45.3,
  "transcript": { ... full session JSON ... }
}
```

### GET /api/v1/live/team-b/consultation/{consultation_id}

Returns the transcript in the exact format Team B's NLP pipeline expects.

**Response:**
```json
{
  "consultation_id": "20260908_160043",
  "segments": [
    {
      "speaker": "DOCTOR",
      "start_time": 0.54,
      "end_time": 3.20,
      "text": "Good morning. What brings you in today?",
      "confidence": 0.95
    },
    {
      "speaker": "PATIENT",
      "start_time": 3.40,
      "end_time": 7.80,
      "text": "I've been having headaches for the past week.",
      "confidence": 0.95
    }
  ]
}
```

### POST /api/v1/live/swap-roles/{session_id}

Swaps DOCTOR ↔ PATIENT for all segments in the saved session file. Idempotent (calling twice returns to original).

---

## 8. Output JSON Format

Each processed session is saved as a JSON file in the `output/` directory.

```json
{
  "consultation_id": "20260908_160043",
  "created_at": "2026-09-08T16:00:43Z",
  "audio_file": "live_recording_20260908_160043.wav",
  "duration": 45.3,
  "segments": [
    {
      "speaker": "DOCTOR",
      "start_time": 0.54,
      "end_time": 3.20,
      "text": "Good morning. What brings you in today?",
      "confidence": 0.95,
      "_internal": {
        "original_speaker": "SPEAKER_00",
        "role_source": "llm_confirmed",
        "original_text": null,
        "text_cleanup_applied": false
      }
    },
    {
      "speaker": "PATIENT",
      "start_time": 3.40,
      "end_time": 7.80,
      "text": "I've been having headaches for the past week.",
      "confidence": 0.95,
      "_internal": {
        "original_speaker": "SPEAKER_01",
        "role_source": "llm_confirmed",
        "original_text": null,
        "text_cleanup_applied": false
      }
    }
  ],
  "metadata": {
    "total_segments": 17,
    "total_duration": 45.3,
    "diarization_status": "enabled",
    "model": "whisper-local-base.en",
    "format_version": "team_b_compatible_v1"
  }
}
```

**Field descriptions:**
- `speaker` — Role in uppercase: `"DOCTOR"` or `"PATIENT"` (Team B format)
- `start_time` / `end_time` — Seconds from start of recording
- `text` — The Whisper transcript (or LLM-cleaned version if cleanup applied)
- `confidence` — Default 0.95 (real Whisper confidence scores can be integrated)
- `_internal.role_source` — Audit trail: was this role from diarization or LLM-corrected?
- `diarization_status` — `"enabled"` or `"failed"` (for diagnostics)

---

## 9. Configuration & Environment Variables

Create a `.env` file in the project root:

```env
# HuggingFace token for pyannote speaker diarization
# Get from: https://huggingface.co/settings/tokens
# Must accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Groq API key for LLM role correction and text cleanup
# Get from: https://console.groq.com/keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Primary LLM model (fast, good for 15-segment transcripts)
GROQ_MODEL=openai/gpt-oss-20b

# Fallback models tried in order if primary fails
GROQ_FALLBACK_MODELS=openai/gpt-oss-120b,groq/compound
```

**Startup validation:** On server startup, `validate_groq_config()` checks that the configured model exists in the Groq API's available models list and prints a warning if not.

---

## 10. Bug Fixes Implemented

### Bug A — Speaker Collapse (Merge Step)

**Problem:** Diarization correctly detected 2 speakers (28 turns), but the merge assigned all 17 Whisper segments to `SPEAKER_01`. The doctor was completely invisible in the output.

**Root cause:** The overlap calculation was not finding sufficient overlap for short `SPEAKER_00` turns against long Whisper segments. Without debug logging, the bug was invisible.

**Fixes:**
1. Added comprehensive debug logging in `merge.py` — prints every segment's overlap with every diarization turn.
2. Added post-merge sanity check that detects when detected speakers go missing.
3. Sorts diarization turns by start time before processing (ensures consistent iteration).
4. Identifies "unmatched turns" — diarization turns that never contributed to any segment assignment.

**Result:** The warning `⚠️ Merge collapsed N speakers into M` now fires automatically when this happens, pinpointing whether the root cause is a timing offset or a comparison bug.

### Bug B — Groq JSON Failures

**Problem:** `openai/gpt-oss-20b` was returning empty `failed_generation` for large transcripts (17+ segments). The code was silently falling through to the last fallback model.

**Root cause:** Models may hit token limits on larger prompts. The original code had no logging to show what response came back.

**Fixes:**
1. Added `finish_reason` logging for every API response.
2. Added raw response preview (first 150 chars) on every attempt.
3. Tolerant JSON parsing: strips markdown fences, tries alternate keys.
4. Token length estimator warns when prompt exceeds ~3000 tokens.
5. Better validation messages showing received vs expected role count.

---

## 11. Feature Additions

### Real Speaker Diarization (Week 3 upgrade from placeholder)

**Before (Week 1-2):** All segments labeled `"UNKNOWN"`. A `diarization_interface.py` placeholder existed but did nothing.

**After (Week 3+):** Full `pyannote.audio` integration in `services/diarization.py`. The singleton pipeline loads once at startup and reuses on every request. Supports GPU acceleration automatically.

### Chat-Style Transcript Viewer (`transcript.html`)

A separate page (`/transcript?session=<id>`) that renders the conversation as a chat interface:
- Doctor = left-aligned blue bubbles
- Patient = right-aligned purple bubbles
- Role swap button at the top
- Session metadata displayed

### Garbled Text Cleanup (Step 2.5)

New module `text_cleanup.py` that detects and optionally repairs Whisper hallucinations using surrounding context. Completely opt-in — only flagged segments go to the LLM. Original text always preserved.

### Long Segment Splitting

`merge.py` was enhanced with `split_segment_at_boundaries()`. When a Whisper segment is >8 seconds and overlaps multiple speakers, it uses word-level timestamps to split the segment at the diarization turn boundary, creating separate sub-segments per speaker.

### Team B Integration Endpoint

`/team-b/consultation/{consultation_id}` returns the transcript in the exact JSON schema Team B's NLP pipeline expects, with `DOCTOR`/`PATIENT` uppercase labels and `start_time`/`end_time` field names.

---

## 12. Team B Integration

Team B (Clinical NLP) consumes this module's output to generate SOAP notes and prescriptions.

**Integration point:**
```python
import requests

# Get transcript for Team B's NLP pipeline
response = requests.get(
    "http://localhost:8000/api/v1/live/team-b/consultation/20260908_160043"
)
consultation = response.json()

# consultation["segments"] contains DOCTOR/PATIENT labeled utterances
# Team B processes these for SOAP note generation
```

**Format contract:**
```json
{
  "consultation_id": "string",
  "segments": [
    {
      "speaker": "DOCTOR | PATIENT",
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "transcribed utterance",
      "confidence": 0.95
    }
  ]
}
```

The `_internal` metadata field is included for our own auditing but Team B is free to ignore it.

---

## 13. Privacy & Security Design

This is the most critical design requirement of the entire project.

| Concern | Our Approach |
|---|---|
| Audio data leaving the system | Never. Audio is processed locally only. |
| Whisper model calls to cloud | None. `faster-whisper` runs entirely on the local machine. |
| Speaker diarization cloud calls | None. `pyannote.audio` model is downloaded once and runs locally. |
| LLM calls sending patient text | Groq API receives text (not audio), only for role correction and garbled text cleanup. This is an accepted tradeoff. |
| Data storage | Session JSONs saved locally in `output/`. No external database. |
| HIPAA considerations | Local processing satisfies the core HIPAA requirement of keeping PHI on the covered entity's infrastructure. |

**Note on Groq usage:** The LLM correction step does send transcript text to the Groq API. In a fully air-gapped deployment, this step can be disabled by not setting `GROQ_API_KEY`, and the system gracefully falls back to diarization-only role assignment.

---

## 14. Development Timeline

| Phase | Week | What was Built |
|---|---|---|
| **Week 1** | 1 | Project scaffold, audio loading, preprocessing, basic Whisper transcription, `transcript_formatter.py`, file upload API endpoint |
| **Week 2** | 2 | FastAPI full setup, CORS, health/models endpoints, diarization placeholder (`diarization_interface.py`), JSON output, CLI test script |
| **Week 3** | 3 | Real `pyannote.audio` diarization, merge logic, role mapping, LLM correction with Groq, chat UI (`transcript.html`), role swap endpoint, unit tests |
| **Post-Week 3** | — | Bug fix: speaker collapse in merge, Bug fix: Groq JSON failures, new feature: garbled text detection & cleanup (Step 2.5), long segment splitting |

---

## 15. Current Project Status

| Component | Status | Notes |
|---|---|---|
| Audio loading | ✅ Complete | Handles all formats via soundfile + pydub |
| Audio preprocessing | ✅ Complete | Mono, 16kHz, normalized |
| Whisper transcription | ✅ Complete | Word timestamps enabled |
| Speaker diarization | ✅ Complete | pyannote 3.1, requires HF_TOKEN |
| Merge logic | ✅ Complete (debug mode) | Extensive logging for bug diagnosis |
| Role mapping | ✅ Complete | First-speaker = Doctor heuristic |
| LLM role correction | ✅ Complete | Groq with 3 model fallback chain |
| Garbled text cleanup | ✅ Complete | Opt-in, original preserved |
| Long segment splitting | ✅ Complete | Uses word timestamps |
| Live recording API | ✅ Complete | sounddevice stream management |
| File upload API | ✅ Complete | Multipart form upload |
| Chat UI | ✅ Complete | Doctor/Patient bubble layout |
| Recording dashboard | ✅ Complete | Start/Stop/Status/Popup |
| Role swap endpoint | ✅ Complete | Idempotent |
| Team B endpoint | ✅ Complete | Compatible JSON format |
| Unit tests | ✅ Complete | merge, roles, swap endpoint |
| Speaker collapse warning | ✅ Complete | Automatic detection |
| GPU support | ✅ Complete | Auto-detected via torch.cuda |
| Privacy (no cloud audio) | ✅ Complete | Whisper + pyannote run locally |

---

## 16. Q&A — Questions Your Mentor May Ask

### About the Core System

**Q1: Why did you choose faster-whisper over the original OpenAI Whisper?**

`faster-whisper` uses the CTranslate2 inference engine with INT8 quantization. On CPU, this makes it 2–4x faster than the original Whisper library while producing identical transcription output. Since we're running locally without a dedicated GPU in most setups, this speed difference is important for user experience. It also has native support for word-level timestamps, which we need for segment splitting.

**Q2: What is speaker diarization and how does pyannote.audio work internally?**

Speaker diarization answers "who spoke when?" in a recording. `pyannote.audio` works in three stages:
1. **Segmentation** — A neural network identifies speech boundaries and short speaker-homogeneous segments.
2. **Embedding** — A speaker embedding model maps each segment to a vector in "speaker space" (similar to a voice fingerprint).
3. **Clustering** — Agglomerative clustering groups segments with similar embeddings as the same speaker.

The model was pre-trained on thousands of hours of multi-speaker audio and downloaded from HuggingFace.

**Q3: Why do you need both Whisper AND pyannote? Can't you do everything with one?**

No — they solve different problems. Whisper is an ASR (Automatic Speech Recognition) model: it transcribes speech to text with timestamps, but it doesn't know who is speaking. pyannote is a diarization model: it knows who spoke when, but it doesn't know what was said. We merge both outputs to get text + speaker identity together.

**Q4: What is the merge algorithm and what edge cases does it handle?**

The core algorithm is maximum temporal overlap: for each Whisper text segment, find the diarization turn that overlaps it the most in time and assign that speaker. Edge cases handled:
- Zero overlap → assign `UNKNOWN`
- Long segments (>8s) spanning multiple speakers → split at word boundaries
- Post-merge sanity check detects if all segments collapse to one speaker

**Q5: How does the LLM role correction work and why is it necessary?**

Diarization only assigns abstract labels like `SPEAKER_00`/`SPEAKER_01` based on audio patterns — it has no semantic understanding. Role mapping uses the heuristic "first speaker = Doctor," which is wrong if the patient speaks first. The LLM step reads the actual text of every utterance and uses clinical conversation understanding to validate or correct the assignment. For example, "I'll prescribe ibuprofen 400mg twice daily" should always be the Doctor regardless of who spoke first.

**Q6: What happens if the HuggingFace token is not set?**

The `get_diarization_pipeline()` function detects the missing/placeholder token and returns `None`. The `diarize()` function checks for this and returns an empty list. The `merge_transcript_with_speakers()` function handles empty diarization turns by assigning `UNKNOWN` to all segments. The server continues running; no crash occurs. The `diarization_status` field in the output JSON is set to `"failed"` to flag this condition to Team B.

**Q7: How do you prevent Whisper hallucinations in silent audio?**

Two mechanisms:
1. `vad_filter=True` in the Whisper transcription — Voice Activity Detection skips silent regions, preventing hallucination on silence.
2. The `detect_garbled_segments()` function in Step 2.5 catches any hallucinations that slip through, using heuristics like repeated words and low unique-word ratio.

**Q8: What is the output format and why is it structured this way?**

The output uses a `consultation_id` and `segments` array format designed in collaboration with Team B. Each segment has `speaker` (DOCTOR/PATIENT), `start_time`, `end_time`, `text`, and `confidence`. The `_internal` field contains our own audit metadata (original speaker label, role source, cleanup flag) that Team B can ignore but we use for debugging and transparency.

**Q9: What is INT8 quantization and why did you enable it?**

Quantization reduces the precision of model weights from 32-bit floating point to 8-bit integers. This reduces memory usage by ~4x and speeds up inference significantly on CPUs that have optimized INT8 arithmetic (most modern CPUs). The accuracy loss for Whisper with INT8 is negligible — typically less than 0.5% WER (Word Error Rate) difference.

**Q10: How does your system handle multiple concurrent users?**

Currently it doesn't — the `recording_state` is a global dictionary, meaning only one recording session can be active at a time. This is acceptable for a clinical scribe (one doctor at a time in a consultation). For a multi-room clinic, each room would run a separate server instance. This is documented as a known limitation.

**Q11: How do you know the role assignment is correct?**

Three-layer verification:
1. Rule-based heuristic (first speaker = Doctor)
2. LLM semantic verification (context-aware role assignment)
3. Human override (the swap-roles button if the system got it wrong)

The `role_source` field in the output JSON provides full audit trail.

**Q12: What does the `vad_filter` parameter do in Whisper?**

VAD stands for Voice Activity Detection. When enabled, Whisper internally detects which parts of the audio contain speech and skips processing the silent parts. This serves two purposes: it's faster (skips silent regions), and it prevents hallucinations — Whisper is known to "hallucinate" text from silence or background noise if VAD is not used.

---

### About Design Decisions

**Q13: Why FastAPI instead of Flask?**

FastAPI has three key advantages for this project:
1. **Automatic API documentation** (Swagger UI at `/docs`) — useful for Team B integration.
2. **Async support** — Background tasks (like processing after recording stops) don't block the server.
3. **Type validation** — Pydantic models validate request/response data automatically.
4. **Performance** — Comparable to Node.js for async workloads.

**Q14: Why save audio files to disk before processing instead of streaming directly to Whisper?**

Two reasons:
1. pyannote.audio requires a file path on disk — it doesn't support in-memory audio streams.
2. Saving to disk creates an audit trail of the raw recording, useful for debugging transcription issues.
The disk I/O overhead is negligible compared to the 10–30 second processing time.

**Q15: Why use a singleton pattern for the diarization pipeline?**

Loading the pyannote model takes 5–15 seconds and uses ~300MB of RAM. If we loaded it fresh on every request, the user would wait 15 extra seconds per consultation. The singleton loads it once on first use and keeps it in memory for the server's lifetime.

---

## 17. Questions You Can Ask Your Mentor

These are thoughtful questions about potential improvements, limitations, and real-world deployment considerations — asking these shows you've thought deeply about the system.

### Performance Questions

**1. For production deployment in a real hospital, what additional hardware would you recommend to meet real-time processing requirements?**  
*This opens discussion about GPU deployment, dedicated inference servers, and latency SLAs.*

**2. The current Whisper model (base.en) achieves ~85–90% word accuracy on clean speech. How would accuracy degrade in a noisy clinic environment, and what preprocessing improvements would you suggest?**  
*Noise reduction (spectral subtraction, RNNoise), directional microphones, close-talk microphones.*

**3. Is there a way to run Whisper and pyannote.audio in parallel to reduce end-to-end latency?**  
*They both need the same audio file. In theory, you could start diarization immediately and run Whisper concurrently, then merge when both finish.*

**4. The merge step assigns each Whisper segment to one speaker based on maximum overlap. Would a different algorithm (like interpolation or segment-boundary alignment) give better results?**

**5. How would streaming/real-time transcription change the architecture?**  
*Whisper supports streaming via OpenAI's streaming API. pyannote's streaming diarization (pyannote-streaming) exists but is less accurate than offline. There's a fundamental latency/accuracy tradeoff.*

### Accuracy & Quality Questions

**6. What evaluation metrics would you use to measure the quality of the diarization and transcription? How would you build a ground-truth dataset for this?**  
*WER (Word Error Rate) for transcription, DER (Diarization Error Rate) for speaker identification. Ground truth needs manual annotation.*

**7. The LLM role correction uses heuristics like "doctors greet first." Are there clinical scenarios where this assumption breaks down, and how would you make it more robust?**  
*Telehealth (patient calls first), nurse practitioners, multi-party consultations.*

**8. How would you handle a consultation in a non-English language, or a bilingual conversation (code-switching)?**  
*Whisper supports 99 languages — change the `language` parameter. pyannote is language-agnostic. But LLM role correction prompts are in English.*

### Architecture & Scalability Questions

**9. Currently, Team A and Team B are separate modules. In production, should these be separate microservices, or would a monolithic architecture make more sense?**

**10. The `_internal` metadata field is included in Team B's response. Is there a better way to architect the API so internal data doesn't leak into the consumer contract?**  
*Separate internal and external DTOs, use API versioning.*

**11. How would you implement proper session management if multiple doctors used the system simultaneously from different rooms?**  
*Session isolation, database-backed state instead of global dictionary, user authentication.*

### Privacy & Compliance Questions

**12. The Groq LLM step sends transcript text to an external API. In a strict HIPAA environment, how would you replace this?**  
*Run a local LLM (Ollama with Llama 3, Mistral, etc.). Performance would be slower but fully air-gapped.*

**13. Should the raw audio files in `sample_audio/` be automatically deleted after processing, and what are the regulatory implications of retaining vs deleting them?**

**14. What kind of audit logging would you implement in a production system to track who accessed which consultation transcript?**

### Bug & Edge Case Questions

**15. The speaker collapse bug (Bug A) is detected but not automatically fixed. What would an automated recovery strategy look like?**  
*If collapse detected, try with different `min_speakers`/`max_speakers` constraints, or use fallback to sentence-boundary heuristics.*

**16. The garbled text detection uses a fixed threshold of 0.4 unique-word ratio. How would you tune this threshold to minimize both false positives (flagging clean speech) and false negatives (missing real hallucinations)?**

**17. What happens to the system if the Groq API is down or returns a 429 rate limit? Is the degraded behavior acceptable for clinical use?**  
*Currently falls back to diarization-only role assignment. In production, a local fallback LLM would be safer.*

---

*This documentation covers the complete implementation of Team A's Privacy-Preserving Clinical Scribe module as of September 2026. All code is local-first, modular, and designed for integration with Team B's Clinical NLP pipeline.*
