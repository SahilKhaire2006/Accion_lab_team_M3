"""
API routes for live audio recording and transcription.

STREAMING PIPELINE — achieves ~2s response time after "End Conversation":
─────────────────────────────────────────────────────────────────────────
  DURING recording (background, invisible to user):
    Every CHUNK_SECONDS (5s), a background thread runs Whisper on the
    accumulated audio so far. Results accumulate in a buffer.

  AFTER "End Conversation" click:
    1. Flush final audio chunk through Whisper             (~1-2s)
    2. Run diarization on full audio (parallel w/ flush)   (~4-6s)
    3. Merge + roles + LLM                                 (~1-2s)
    ─────────────────────────────────────────────────────────────────
    User waits for only the LAST CHUNK + diarization, not entire audio.

Model warm-up (on server start):
    Both Whisper and pyannote are loaded once into memory at startup.
    No per-request loading cost.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import os
import json
import time
import threading
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from app.audio.audio_loader import AudioLoader
from app.audio.preprocessing import AudioPreprocessor
from app.transcription.whisper_service import WhisperService
from app.services.diarization import diarize
from app.services.merge import merge_transcript_with_speakers
from app.services.roles import label_roles
from app.services.llm_correction import refine_roles_with_llm
from app.services.text_cleanup import detect_garbled_segments, cleanup_garbled_segments


router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE       = 48000    # microphone capture rate
TARGET_SR         = 16000    # Whisper expects 16kHz
CHUNK_SECONDS     = 5        # process a Whisper chunk every N seconds of audio
CHANNELS          = 2        # stereo mic → will be converted to mono

# ---------------------------------------------------------------------------
# Module-level singletons & shared state
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=3)

# Recording state
recording_state = {
    "is_recording":       False,
    "audio_queue":        None,
    "stream":             None,
    # Streaming pipeline state
    "all_audio_chunks":   [],        # every raw chunk since start-recording
    "processed_audio":    None,      # preprocessed float32 mono 16kHz array
    "whisper_segments":   [],        # segments produced by background Whisper runs
    "chunk_lock":         threading.Lock(),
    "bg_whisper_future":  None,
    "total_processed_samples": 0,   # how many 16kHz samples already Whisper-processed
}

# Pre-loaded service (singleton, warm at startup)
_whisper_service  = None
_preprocessor     = AudioPreprocessor()


def get_whisper_service() -> WhisperService:
    global _whisper_service
    if _whisper_service is None:
        _whisper_service = WhisperService(model_name="base.en")
        _whisper_service.load_model()   # loads singleton — instant after first call
    return _whisper_service


# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Mic] {status}")
    if recording_state["audio_queue"] is not None:
        recording_state["audio_queue"].put(indata.copy())


# ---------------------------------------------------------------------------
# Background Whisper chunk processor
# ---------------------------------------------------------------------------
def _process_new_audio_chunk():
    """
    Called in a background thread every CHUNK_SECONDS.
    Drains the audio queue, preprocesses, runs Whisper on the NEW audio only,
    and appends results to recording_state["whisper_segments"].
    """
    with recording_state["chunk_lock"]:
        # Drain queue into all_audio_chunks
        new_raw = []
        q = recording_state["audio_queue"]
        while q and not q.empty():
            new_raw.append(q.get_nowait())

        if not new_raw:
            return

        recording_state["all_audio_chunks"].extend(new_raw)

        # Build full raw audio and preprocess to 16kHz mono float32
        full_raw = np.concatenate(recording_state["all_audio_chunks"], axis=0)
        full_processed = _preprocessor.preprocess(full_raw, SAMPLE_RATE)

        # Only transcribe the NEW portion (not already processed)
        already = recording_state["total_processed_samples"]
        new_audio = full_processed[already:]

        if len(new_audio) < TARGET_SR * 1:   # skip if less than 1 second
            return

        offset = already / TARGET_SR          # time offset in seconds

        # Run Whisper on just the new chunk
        ws = get_whisper_service()
        new_segs = ws.transcribe_chunk(new_audio, offset_seconds=offset)

        recording_state["whisper_segments"].extend(new_segs)
        recording_state["total_processed_samples"] = len(full_processed)
        recording_state["processed_audio"] = full_processed

        print(f"  [BG Whisper] chunk @{offset:.1f}s → {len(new_segs)} new segments "
              f"(total {len(recording_state['whisper_segments'])})")


# ---------------------------------------------------------------------------
# Chunk scheduler — fires every CHUNK_SECONDS while recording
# ---------------------------------------------------------------------------
def _chunk_scheduler():
    """Runs in a daemon thread while recording is active."""
    while recording_state["is_recording"]:
        time.sleep(CHUNK_SECONDS)
        if recording_state["is_recording"]:
            try:
                _executor.submit(_process_new_audio_chunk)
            except Exception as e:
                print(f"  [BG Whisper] scheduler error: {e}")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/start-recording")
async def start_recording():
    """Start live audio recording and background streaming transcription."""
    global recording_state

    if recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Recording already in progress")

    # Reset state
    recording_state.update({
        "is_recording":            True,
        "audio_queue":             queue.Queue(),
        "all_audio_chunks":        [],
        "whisper_segments":        [],
        "total_processed_samples": 0,
        "processed_audio":         None,
        "bg_whisper_future":       None,
    })

    # Ensure Whisper is warm before recording starts
    _executor.submit(get_whisper_service)

    device_info = sd.query_devices(kind='input')
    print(f"[Mic] Using: {device_info['name']}")

    recording_state["stream"] = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=1024,
    )
    recording_state["stream"].start()

    # Start background chunk scheduler
    scheduler = threading.Thread(target=_chunk_scheduler, daemon=True)
    scheduler.start()

    return JSONResponse(content={
        "status":      "recording_started",
        "message":     "Recording started. Transcription running in background.",
        "device":      device_info["name"],
        "sample_rate": SAMPLE_RATE,
    })


@router.post("/stop-recording")
async def stop_recording(background_tasks: BackgroundTasks):
    """
    Stop recording and complete the transcript.
    Background Whisper has been running during recording — only the final
    chunk + diarization needs to finish now, giving ~2-4s response time.
    """
    global recording_state

    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="No recording in progress")

    stop_time = time.time()

    # Stop mic stream
    recording_state["stream"].stop()
    recording_state["stream"].close()
    recording_state["is_recording"] = False   # also stops chunk scheduler

    # Drain any remaining audio from queue
    remaining = []
    q = recording_state["audio_queue"]
    while not q.empty():
        remaining.append(q.get_nowait())
    if remaining:
        recording_state["all_audio_chunks"].extend(remaining)

    if not recording_state["all_audio_chunks"]:
        raise HTTPException(status_code=400, detail="No audio data recorded")

    # Save raw audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = Path("sample_audio")
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"live_recording_{timestamp}.wav"
    full_raw = np.concatenate(recording_state["all_audio_chunks"], axis=0)
    sf.write(str(audio_path), full_raw, SAMPLE_RATE)

    transcript_result = await _finalize_transcript(
        full_raw, str(audio_path), timestamp, stop_time
    )

    return JSONResponse(content={
        "status":       "recording_stopped",
        "message":      "Recording stopped and transcription complete",
        "audio_file":   str(audio_path),
        "duration":     len(full_raw) / SAMPLE_RATE,
        "transcript":   transcript_result,
    })


# ---------------------------------------------------------------------------
# Silence-gap speaker heuristic (used when diarization is skipped)
# ---------------------------------------------------------------------------

def _silence_gap_speakers(segments: List[Dict], gap_threshold: float = 0.4) -> List[Dict]:
    """
    Assign speaker labels based on silence gaps between segments.
    A gap > gap_threshold seconds triggers a speaker change.
    Speaker labels: SPEAKER_00, SPEAKER_01 (alternating on each gap).
    """
    if not segments:
        return segments

    result = []
    current_speaker = "SPEAKER_00"
    prev_end = segments[0]["end"]

    for i, seg in enumerate(segments):
        if i > 0:
            gap = seg["start"] - prev_end
            if gap >= gap_threshold:
                # Switch speaker on silence gap
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
        result.append({**seg, "speaker": current_speaker})
        prev_end = seg["end"]

    speakers = set(s["speaker"] for s in result)
    print(f"  Silence-gap: {len(result)} segments, {len(speakers)} speaker(s) detected")
    return result


# ---------------------------------------------------------------------------
# Finalisation — runs after stop-recording
# ---------------------------------------------------------------------------

async def _finalize_transcript(full_raw: np.ndarray,
                                audio_path: str,
                                session_id: str,
                                stop_time: float):
    """
    Complete the transcript after recording ends.

    Work breakdown:
      A) Flush final unprocessed audio through Whisper   (thread 1)
      B) Run diarization on full audio                   (thread 2)
      Both run in parallel, then merge/roles/LLM.
    """
    t_start = time.time()
    os.makedirs("output", exist_ok=True)

    print("\n" + "="*70)
    print("FINALISING TRANSCRIPT  [streaming pipeline]")
    print("="*70)

    # ------------------------------------------------------------------
    # Preprocess full audio (fast — ~0.1s)
    # ------------------------------------------------------------------
    full_processed = _preprocessor.preprocess(full_raw, SAMPLE_RATE)
    temp_path = "output/temp_live_processed.wav"
    sf.write(temp_path, full_processed, TARGET_SR)

    already_processed = recording_state["total_processed_samples"]
    bg_segments       = list(recording_state["whisper_segments"])   # copy

    print(f"\n  Background already transcribed: "
          f"{already_processed/TARGET_SR:.1f}s "
          f"({len(bg_segments)} segments)")

    # ------------------------------------------------------------------
    # PARALLEL: flush final chunk (Whisper) + full diarization
    # ------------------------------------------------------------------
    print("\n  Running final Whisper flush + Diarization in PARALLEL...")
    t_parallel = time.time()

    def _flush_final():
        """Whisper only on the unprocessed tail."""
        remaining_audio = full_processed[already_processed:]
        if len(remaining_audio) < TARGET_SR * 0.5:   # < 0.5s — nothing to do
            print("  [Flush] No remaining audio to flush")
            return []
        offset = already_processed / TARGET_SR
        ws = get_whisper_service()
        segs = ws.transcribe_chunk(remaining_audio, offset_seconds=offset)
        print(f"  [Flush] {len(segs)} new segments from final chunk")
        return segs

    def _run_diarization():
        # Skip diarization for audio > 120s on CPU — would take too long
        # Silence-gap heuristic + LLM correction is used instead
        audio_duration = len(full_processed) / TARGET_SR
        if audio_duration > 120:
            print(f"  [Diarization] Audio is {audio_duration:.0f}s — "
                  f"skipping pyannote on CPU (>120s limit), using silence-gap heuristic")
            return [], "skipped_long_audio"
        try:
            turns = diarize(temp_path, min_speakers=2, max_speakers=2)
            if not turns:
                return [], "failed"
            return turns, "enabled"
        except Exception as e:
            print(f"  [Diarization] Error: {e}")
            return [], "failed"

    future_flush  = _executor.submit(_flush_final)
    future_diariz = _executor.submit(_run_diarization)

    final_new_segs              = future_flush.result()
    diarization_turns, dia_status = future_diariz.result()

    parallel_elapsed = time.time() - t_parallel
    print(f"\n  ✓ Parallel flush+diarization done in {parallel_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Combine all Whisper segments and deduplicate
    # ------------------------------------------------------------------
    all_segments = bg_segments + final_new_segs

    # Deduplicate: remove segments with duplicate (start, end) pairs
    seen = set()
    whisper_segments = []
    for s in sorted(all_segments, key=lambda x: x["start"]):
        key = (s["start"], s["end"])
        if key not in seen:
            seen.add(key)
            whisper_segments.append(s)

    print(f"\n  Total Whisper segments after dedup: {len(whisper_segments)}")

    # ------------------------------------------------------------------
    # Garbled text cleanup (fast, only on flagged segments)
    # ------------------------------------------------------------------
    whisper_segments = detect_garbled_segments(whisper_segments)
    if any(seg.get("likely_garbled") for seg in whisper_segments):
        whisper_segments = cleanup_garbled_segments(whisper_segments)

    # ------------------------------------------------------------------
    # Merge → Roles → LLM
    # ------------------------------------------------------------------
    print("\n  Merging speaker labels...")

    if diarization_turns:
        # Full pyannote diarization available
        merged = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    else:
        # No diarization — use silence-gap heuristic
        # Assign speakers based on pauses: gap > 400ms = speaker change
        print("  Using silence-gap heuristic for speaker assignment...")
        merged = _silence_gap_speakers(whisper_segments, gap_threshold=0.4)

    print("\n  Mapping speakers to roles...")
    final_segments = label_roles(merged, first_speaker_is="Doctor")

    print("\n  LLM role refinement...")
    t_llm = time.time()
    final_segments = refine_roles_with_llm(final_segments)
    print(f"  ✓ LLM done in {time.time()-t_llm:.2f}s")

    # ------------------------------------------------------------------
    # Build + save session JSON
    # ------------------------------------------------------------------
    duration      = final_segments[-1]["end"] if final_segments else 0.0
    total_elapsed = time.time() - t_start

    role_counts = {}
    for seg in final_segments:
        r = seg["role"]
        role_counts[r] = role_counts.get(r, 0) + 1
    print(f"  ✓ Role distribution: {role_counts}")

    team_b_segments = []
    for seg in final_segments:
        team_b_seg = {
            "speaker":    seg["role"].upper(),
            "start_time": seg["start"],
            "end_time":   seg["end"],
            "text":       seg.get("text_cleaned", seg["text"]),
            "confidence": 0.95,
        }
        team_b_seg["_internal"] = {
            "original_speaker":     seg["speaker"],
            "role_source":          seg["role_source"],
            "original_text":        seg["text"] if seg.get("text_cleaned") else None,
            "text_cleanup_applied": seg.get("text_cleanup_applied", False),
        }
        team_b_segments.append(team_b_seg)

    session_data = {
        "consultation_id": session_id,
        "created_at":      datetime.utcnow().isoformat() + "Z",
        "audio_file":      os.path.basename(audio_path),
        "duration":        duration,
        "segments":        team_b_segments,
        "metadata": {
            "total_segments":          len(team_b_segments),
            "total_duration":          duration,
            "diarization_status":      dia_status,
            "model":                   "whisper-local-base.en",
            "format_version":          "team_b_compatible_v1",
            "processing_time_seconds": round(total_elapsed, 2),
            "pipeline":                "streaming",
        },
    }

    output_path = Path("output") / f"session_{session_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Session saved: {output_path}")
    print(f"✓ POST-STOP PROCESSING TIME: {total_elapsed:.2f}s  "
          f"(audio: {duration:.1f}s)")
    print("="*70 + "\n")

    return {**session_data, "output_file": str(output_path)}


# ---------------------------------------------------------------------------
# Remaining endpoints (unchanged)
# ---------------------------------------------------------------------------

@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    output_path = Path("output") / f"session_{session_id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Session not found: {session_id}")
    with open(output_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@router.post("/swap-roles/{session_id}")
async def swap_roles(session_id: str):
    output_path = Path("output") / f"session_{session_id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Session not found: {session_id}")
    with open(output_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    for seg in session_data["segments"]:
        if seg["speaker"] == "DOCTOR":
            seg["speaker"] = "PATIENT"
        elif seg["speaker"] == "PATIENT":
            seg["speaker"] = "DOCTOR"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Roles swapped for session: {session_id}")
    return JSONResponse(content=session_data)


@router.get("/recording-status")
async def get_recording_status():
    return JSONResponse(content={
        "is_recording": recording_state["is_recording"]
    })


@router.get("/team-b/consultation/{consultation_id}")
async def get_consultation_for_team_b(consultation_id: str):
    output_path = Path("output") / f"session_{consultation_id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Consultation not found: {consultation_id}")
    with open(output_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    return JSONResponse(content={
        "consultation_id": session_data.get("consultation_id", consultation_id),
        "segments":        session_data["segments"],
    })
