"""
API routes for live audio recording and transcription.

PIPELINE OVERVIEW:
─────────────────────────────────────────────────────────────────────────
  DURING recording:
    Every 5s → background Whisper chunk processing (text only, no speaker)

  ON "End Conversation":
    1. Flush final Whisper chunk                (~1-3s)
    2. Apply silence-gap speaker assignment     (~0.05s)
    3. LLM role assignment                      (~2-5s)
    4. Return response to user                  ← user sees result fast

  IN BACKGROUND (after response sent):
    5. Run full pyannote diarization            (~60-120s on CPU)
    6. Re-merge + re-run LLM with better labels
    7. Update saved JSON file silently
    8. /diarization-status/{id} endpoint tells UI when ready
─────────────────────────────────────────────────────────────────────────
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
SAMPLE_RATE   = 48000
TARGET_SR     = 16000
CHUNK_SECONDS = 5
CHANNELS      = 2

# ---------------------------------------------------------------------------
# Singletons & shared state
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)

recording_state = {
    "is_recording":            False,
    "audio_queue":             None,
    "stream":                  None,
    "all_audio_chunks":        [],
    "whisper_segments":        [],
    "chunk_lock":              threading.Lock(),
    "total_processed_samples": 0,
}

# Background diarization status per session
# {session_id: "pending" | "running" | "done" | "failed"}
_diarization_status: Dict[str, str] = {}

_whisper_service = None
_preprocessor    = AudioPreprocessor()


def get_whisper_service() -> WhisperService:
    global _whisper_service
    if _whisper_service is None:
        _whisper_service = WhisperService(model_name="base.en")
        _whisper_service.load_model()
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
# Background Whisper chunk processor (runs every CHUNK_SECONDS)
# ---------------------------------------------------------------------------
def _process_new_audio_chunk():
    with recording_state["chunk_lock"]:
        new_raw = []
        q = recording_state["audio_queue"]
        while q and not q.empty():
            new_raw.append(q.get_nowait())
        if not new_raw:
            return

        recording_state["all_audio_chunks"].extend(new_raw)
        full_raw = np.concatenate(recording_state["all_audio_chunks"], axis=0)
        full_processed = _preprocessor.preprocess(full_raw, SAMPLE_RATE)

        already = recording_state["total_processed_samples"]
        new_audio = full_processed[already:]
        if len(new_audio) < TARGET_SR * 1:
            return

        offset = already / TARGET_SR
        ws = get_whisper_service()
        new_segs = ws.transcribe_chunk(new_audio, offset_seconds=offset)

        recording_state["whisper_segments"].extend(new_segs)
        recording_state["total_processed_samples"] = len(full_processed)
        print(f"  [BG Whisper] chunk @{offset:.1f}s → "
              f"{len(new_segs)} new segs (total {len(recording_state['whisper_segments'])})")


def _chunk_scheduler():
    while recording_state["is_recording"]:
        time.sleep(CHUNK_SECONDS)
        if recording_state["is_recording"]:
            try:
                _executor.submit(_process_new_audio_chunk)
            except Exception as e:
                print(f"  [BG Whisper] scheduler error: {e}")


# ---------------------------------------------------------------------------
# Speaker heuristics
# ---------------------------------------------------------------------------
def _silence_gap_speakers(segments: List[Dict], gap_threshold: float = 0.35) -> List[Dict]:
    """
    Assign speaker labels by detecting pauses between segments.
    A gap >= gap_threshold seconds → speaker changed.
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
                current_speaker = ("SPEAKER_01"
                                   if current_speaker == "SPEAKER_00"
                                   else "SPEAKER_00")
        result.append({**seg, "speaker": current_speaker})
        prev_end = seg["end"]

    speakers = set(s["speaker"] for s in result)
    print(f"  Silence-gap: {len(result)} segments, "
          f"{len(speakers)} speaker(s) detected")
    return result


# ---------------------------------------------------------------------------
# Background diarization updater
# ---------------------------------------------------------------------------
def _background_diarize_and_update(session_id: str,
                                    temp_path: str,
                                    whisper_segments: List[Dict]):
    """
    Runs AFTER the HTTP response is already sent.
    Re-runs pyannote diarization, re-merges, re-runs LLM, overwrites JSON.
    Updates _diarization_status[session_id] throughout.
    """
    output_path = Path("output") / f"session_{session_id}.json"
    _diarization_status[session_id] = "running"
    print(f"\n  [BG Diarization] Starting for session {session_id}...")

    try:
        t0 = time.time()

        # Run full pyannote diarization (no time limit here — runs in bg)
        diarization_turns = diarize(temp_path, min_speakers=2, max_speakers=2)

        if not diarization_turns:
            print(f"  [BG Diarization] No turns returned, keeping silence-gap result")
            _diarization_status[session_id] = "failed"
            return

        elapsed_dia = time.time() - t0
        print(f"  [BG Diarization] Diarization done in {elapsed_dia:.1f}s")

        # Merge with proper diarization
        merged = merge_transcript_with_speakers(whisper_segments, diarization_turns)
        final_segments = label_roles(merged, first_speaker_is="Doctor")

        # Re-run LLM with properly diarized segments
        print("  [BG Diarization] Running LLM role refinement...")
        final_segments = refine_roles_with_llm(final_segments)

        # Load existing JSON, replace only segments and metadata
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        else:
            print(f"  [BG Diarization] Session file not found: {output_path}")
            _diarization_status[session_id] = "failed"
            return

        # Rebuild segments in Team B format
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

        session_data["segments"] = team_b_segments
        session_data["metadata"]["diarization_status"] = "enabled"
        session_data["metadata"]["total_segments"] = len(team_b_segments)
        session_data["metadata"]["bg_diarization_time"] = round(time.time() - t0, 1)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        total = time.time() - t0
        print(f"  [BG Diarization] ✓ Session updated in {total:.1f}s — "
              f"{len(team_b_segments)} segments")

        _diarization_status[session_id] = "done"

    except Exception as e:
        print(f"  [BG Diarization] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        _diarization_status[session_id] = "failed"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/start-recording")
async def start_recording():
    global recording_state

    if recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Recording already in progress")

    recording_state.update({
        "is_recording":            True,
        "audio_queue":             queue.Queue(),
        "all_audio_chunks":        [],
        "whisper_segments":        [],
        "total_processed_samples": 0,
    })

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

    threading.Thread(target=_chunk_scheduler, daemon=True).start()

    return JSONResponse(content={
        "status":      "recording_started",
        "message":     "Recording started. Transcription running in background.",
        "device":      device_info["name"],
        "sample_rate": SAMPLE_RATE,
    })


@router.post("/stop-recording")
async def stop_recording(background_tasks: BackgroundTasks):
    global recording_state

    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="No recording in progress")

    recording_state["stream"].stop()
    recording_state["stream"].close()
    recording_state["is_recording"] = False

    remaining = []
    q = recording_state["audio_queue"]
    while not q.empty():
        remaining.append(q.get_nowait())
    if remaining:
        recording_state["all_audio_chunks"].extend(remaining)

    if not recording_state["all_audio_chunks"]:
        raise HTTPException(status_code=400, detail="No audio data recorded")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = Path("sample_audio")
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"live_recording_{timestamp}.wav"
    full_raw = np.concatenate(recording_state["all_audio_chunks"], axis=0)
    sf.write(str(audio_path), full_raw, SAMPLE_RATE)

    transcript_result = await _finalize_transcript(
        full_raw, str(audio_path), timestamp, background_tasks
    )

    return JSONResponse(content={
        "status":     "recording_stopped",
        "message":    "Recording stopped and transcription complete",
        "audio_file": str(audio_path),
        "duration":   len(full_raw) / SAMPLE_RATE,
        "transcript": transcript_result,
    })


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

async def _finalize_transcript(full_raw: np.ndarray,
                                audio_path: str,
                                session_id: str,
                                background_tasks: BackgroundTasks):
    t_start = time.time()
    os.makedirs("output", exist_ok=True)

    print("\n" + "="*70)
    print("FINALISING TRANSCRIPT  [streaming + bg-diarization pipeline]")
    print("="*70)

    # Preprocess
    full_processed = _preprocessor.preprocess(full_raw, SAMPLE_RATE)
    temp_path = "output/temp_live_processed.wav"
    sf.write(temp_path, full_processed, TARGET_SR)

    already_processed = recording_state["total_processed_samples"]
    bg_segments       = list(recording_state["whisper_segments"])

    print(f"\n  Background transcribed: {already_processed/TARGET_SR:.1f}s "
          f"({len(bg_segments)} segments)")

    # ------------------------------------------------------------------
    # Flush final Whisper chunk
    # ------------------------------------------------------------------
    print("\n  Flushing final audio chunk through Whisper...")
    t_flush = time.time()
    remaining_audio = full_processed[already_processed:]
    final_new_segs = []
    if len(remaining_audio) >= TARGET_SR * 0.5:
        offset = already_processed / TARGET_SR
        ws = get_whisper_service()
        final_new_segs = ws.transcribe_chunk(remaining_audio, offset_seconds=offset)
        print(f"  [Flush] {len(final_new_segs)} new segments in "
              f"{time.time()-t_flush:.2f}s")
    else:
        print("  [Flush] No remaining audio")

    # ------------------------------------------------------------------
    # Deduplicate all Whisper segments
    # ------------------------------------------------------------------
    all_segs = bg_segments + final_new_segs
    seen = set()
    whisper_segments = []
    for s in sorted(all_segs, key=lambda x: x["start"]):
        key = (s["start"], s["end"])
        if key not in seen:
            seen.add(key)
            whisper_segments.append(s)
    print(f"\n  Total Whisper segments: {len(whisper_segments)}")

    # Garbled text cleanup
    whisper_segments = detect_garbled_segments(whisper_segments)
    if any(seg.get("likely_garbled") for seg in whisper_segments):
        whisper_segments = cleanup_garbled_segments(whisper_segments)

    # ------------------------------------------------------------------
    # Quick speaker assignment (silence-gap) for immediate response
    # ------------------------------------------------------------------
    print("\n  Applying silence-gap speaker assignment...")
    merged = _silence_gap_speakers(whisper_segments, gap_threshold=0.35)
    final_segments = label_roles(merged, first_speaker_is="Doctor")

    # ------------------------------------------------------------------
    # LLM role refinement
    # ------------------------------------------------------------------
    print("\n  LLM role refinement...")
    t_llm = time.time()
    final_segments = refine_roles_with_llm(final_segments)
    print(f"  ✓ LLM done in {time.time()-t_llm:.2f}s")

    role_counts = {}
    for seg in final_segments:
        role_counts[seg["role"]] = role_counts.get(seg["role"], 0) + 1
    print(f"  ✓ Role distribution: {role_counts}")

    # ------------------------------------------------------------------
    # Build + save initial session JSON
    # ------------------------------------------------------------------
    duration      = final_segments[-1]["end"] if final_segments else 0.0
    total_elapsed = time.time() - t_start

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
            "diarization_status":      "pending_background",
            "model":                   "whisper-local-base.en",
            "format_version":          "team_b_compatible_v1",
            "processing_time_seconds": round(total_elapsed, 2),
            "pipeline":                "streaming+bg-diarization",
        },
    }

    output_path = Path("output") / f"session_{session_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Initial session saved: {output_path}")
    print(f"✓ RESPONSE TIME: {total_elapsed:.2f}s  (audio: {duration:.1f}s)")
    print("  → Launching background diarization to improve accuracy...")
    print("="*70 + "\n")

    # ------------------------------------------------------------------
    # Schedule background diarization (runs AFTER response is returned)
    # ------------------------------------------------------------------
    _diarization_status[session_id] = "pending"

    # Copy whisper_segments for background thread (immutable snapshot)
    ws_snapshot = [dict(s) for s in whisper_segments]
    temp_path_copy = temp_path  # already saved to disk

    background_tasks.add_task(
        _background_diarize_and_update,
        session_id,
        temp_path_copy,
        ws_snapshot,
    )

    return {**session_data, "output_file": str(output_path)}


# ---------------------------------------------------------------------------
# Remaining endpoints
# ---------------------------------------------------------------------------

@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    output_path = Path("output") / f"session_{session_id}.json"
    if not output_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Session not found: {session_id}")
    with open(output_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@router.get("/diarization-status/{session_id}")
async def get_diarization_status(session_id: str):
    """
    Poll this endpoint to know when background diarization is complete.
    Returns: pending | running | done | failed | unknown
    Frontend can auto-refresh the transcript when status == 'done'.
    """
    status = _diarization_status.get(session_id, "unknown")
    return JSONResponse(content={
        "session_id": session_id,
        "status":     status,
        "message": {
            "pending":  "Diarization queued, will start shortly",
            "running":  "Diarization in progress, transcript will auto-update",
            "done":     "Diarization complete, reload transcript for accurate speakers",
            "failed":   "Diarization failed, silence-gap labels remain",
            "unknown":  "Session not found in diarization queue",
        }.get(status, status)
    })


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
    return JSONResponse(content={"is_recording": recording_state["is_recording"]})


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
