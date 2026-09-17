"""
API routes for live audio recording and transcription.

Parallel processing pipeline (latency optimisation):
  - Step 1  : Preprocess audio              (sequential — both models need this file)
  - Step 2  : Whisper transcription  ─┐
                                       ├─ run simultaneously via ThreadPoolExecutor
  - Step 3  : Speaker diarization   ─┘
  - Step 2.5: Garbled text cleanup          (runs inside Whisper thread, no extra cost)
  - Step 4+ : Merge → Roles → LLM → Save  (sequential — each depends on previous)
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
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.audio.audio_loader import AudioLoader
from app.audio.preprocessing import AudioPreprocessor
from app.transcription.whisper_service import WhisperService
from app.transcription.transcript_formatter import TranscriptFormatter
from app.services.diarization import diarize
from app.services.merge import merge_transcript_with_speakers
from app.services.roles import label_roles
from app.services.llm_correction import refine_roles_with_llm
from app.services.text_cleanup import detect_garbled_segments, cleanup_garbled_segments


router = APIRouter()

# Global recording state
recording_state = {
    "is_recording": False,
    "audio_queue": None,
    "sample_rate": 48000,
    "recording_thread": None,
    "audio_data": []
}

# Shared thread pool — reused across requests (avoids thread creation overhead)
_executor = ThreadPoolExecutor(max_workers=2)


def audio_callback(indata, frames, time_info, status):
    """Callback function for audio recording."""
    if status:
        print(f"Audio callback status: {status}")
    if recording_state["audio_queue"] is not None:
        recording_state["audio_queue"].put(indata.copy())


@router.post("/start-recording")
async def start_recording():
    """Start live audio recording."""
    global recording_state

    if recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Recording already in progress")

    try:
        recording_state["is_recording"] = True
        recording_state["audio_queue"] = queue.Queue()
        recording_state["audio_data"] = []

        device_info = sd.query_devices(kind='input')
        print(f"Using microphone: {device_info['name']}")

        recording_state["stream"] = sd.InputStream(
            samplerate=recording_state["sample_rate"],
            channels=2,
            callback=audio_callback,
            blocksize=1024
        )
        recording_state["stream"].start()

        return JSONResponse(content={
            "status": "recording_started",
            "message": "Recording started successfully",
            "device": device_info['name'],
            "sample_rate": recording_state["sample_rate"]
        })

    except Exception as e:
        recording_state["is_recording"] = False
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")


@router.post("/stop-recording")
async def stop_recording(background_tasks: BackgroundTasks):
    """Stop recording and transcribe audio using parallel Whisper + Diarization."""
    global recording_state

    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="No recording in progress")

    try:
        recording_state["stream"].stop()
        recording_state["stream"].close()
        recording_state["is_recording"] = False

        audio_chunks = []
        while not recording_state["audio_queue"].empty():
            audio_chunks.append(recording_state["audio_queue"].get())

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio data recorded")

        audio_data = np.concatenate(audio_chunks, axis=0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_dir = Path("sample_audio")
        audio_dir.mkdir(exist_ok=True)
        audio_filename = f"live_recording_{timestamp}.wav"
        audio_path = audio_dir / audio_filename

        sf.write(str(audio_path), audio_data, recording_state["sample_rate"])

        transcript_result = await transcribe_audio(str(audio_path), timestamp)

        return JSONResponse(content={
            "status": "recording_stopped",
            "message": "Recording stopped and transcription complete",
            "audio_file": str(audio_path),
            "duration": len(audio_data) / recording_state["sample_rate"],
            "transcript": transcript_result
        })

    except Exception as e:
        recording_state["is_recording"] = False
        raise HTTPException(status_code=500, detail=f"Error stopping recording: {str(e)}")


# ---------------------------------------------------------------------------
# Worker functions — run inside ThreadPoolExecutor threads
# ---------------------------------------------------------------------------

def _run_whisper(temp_path: str):
    """
    Thread worker: Whisper transcription + garbled text cleanup.
    Returns whisper_segments (with text_cleaned where applicable).
    """
    t0 = time.time()
    print("\n  [WHISPER THREAD] Starting transcription...")

    whisper_service = WhisperService(model_name="base.en")
    whisper_service.load_model()
    segments = whisper_service.transcribe(temp_path)

    # Step 2.5 runs inside the Whisper thread — no extra wall-clock cost
    print("\n  [WHISPER THREAD] Checking for garbled segments...")
    segments = detect_garbled_segments(segments)
    if any(seg.get("likely_garbled") for seg in segments):
        segments = cleanup_garbled_segments(segments)
    else:
        print("  [WHISPER THREAD] No garbled segments detected")

    elapsed = time.time() - t0
    print(f"\n  [WHISPER THREAD] Done in {elapsed:.2f}s ({len(segments)} segments)")
    return segments


def _run_diarization(temp_path: str):
    """
    Thread worker: pyannote.audio speaker diarization.
    Returns (diarization_turns, diarization_status).
    """
    t0 = time.time()
    print("\n  [DIARIZATION THREAD] Starting diarization...")

    try:
        turns = diarize(temp_path, min_speakers=2, max_speakers=2)
        elapsed = time.time() - t0

        if not turns:
            print(f"  [DIARIZATION THREAD] No results, fallback to UNKNOWN ({elapsed:.2f}s)")
            return [], "failed"

        print(f"  [DIARIZATION THREAD] Done in {elapsed:.2f}s ({len(turns)} turns)")
        return turns, "enabled"

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [DIARIZATION THREAD] Error after {elapsed:.2f}s: {e}")
        print("  [DIARIZATION THREAD] Falling back to UNKNOWN speaker labels")
        return [], "failed"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def transcribe_audio(audio_path: str, session_id: str = None):
    """
    Full processing pipeline with parallel Whisper + Diarization.

    Timeline:
        ┌─ Preprocess (sequential) ──────────────────────────────────┐
        │                                                             │
        │   ┌─ Whisper + garbled cleanup (thread 1) ─┐               │
        │   │                                         ├─ merge → ... │
        │   └─ Diarization             (thread 2) ─┘               │
        └─────────────────────────────────────────────────────────────┘
    """
    pipeline_start = time.time()

    try:
        audio_loader = AudioLoader()
        preprocessor = AudioPreprocessor()

        print("\n" + "="*70)
        print("PROCESSING RECORDING  [parallel Whisper + Diarization]")
        print("="*70)

        # ------------------------------------------------------------------
        # STEP 1 — Preprocess (sequential, both threads need this file)
        # ------------------------------------------------------------------
        t1 = time.time()
        print("\nSTEP 1: Loading and preprocessing audio...")
        audio_data, sample_rate = audio_loader.load(audio_path)
        processed_audio = preprocessor.preprocess(audio_data, sample_rate)

        os.makedirs("output", exist_ok=True)
        temp_path = "output/temp_live_processed.wav"
        audio_loader.save(processed_audio, temp_path, preprocessor.target_sample_rate)
        print(f"  ✓ Preprocessing done in {time.time() - t1:.2f}s")

        # ------------------------------------------------------------------
        # STEP 2 + STEP 3 — Whisper & Diarization in PARALLEL
        # ------------------------------------------------------------------
        print("\nSTEP 2+3: Running Whisper and Diarization in PARALLEL...")
        t_parallel = time.time()

        future_whisper = _executor.submit(_run_whisper, temp_path)
        future_diarize = _executor.submit(_run_diarization, temp_path)

        # Collect results — both must finish before we can merge
        whisper_segments = future_whisper.result()      # blocks until Whisper done
        diarization_turns, diarization_status = future_diarize.result()  # blocks until diarization done

        parallel_elapsed = time.time() - t_parallel
        print(f"\n  ✓ Parallel step done in {parallel_elapsed:.2f}s "
              f"(vs ~{parallel_elapsed * 1.8:.1f}s sequential estimate)")

        # ------------------------------------------------------------------
        # STEP 4 — Merge
        # ------------------------------------------------------------------
        print("\nSTEP 4: Merging speaker labels with transcript...")
        t4 = time.time()
        merged_segments = merge_transcript_with_speakers(whisper_segments, diarization_turns)
        print(f"  ✓ Merged {len(merged_segments)} segments in {time.time() - t4:.2f}s")

        # ------------------------------------------------------------------
        # STEP 5 — Label roles
        # ------------------------------------------------------------------
        print("\nSTEP 5: Mapping speakers to roles...")
        final_segments = label_roles(merged_segments, first_speaker_is="Doctor")

        # ------------------------------------------------------------------
        # STEP 6 — LLM role refinement
        # ------------------------------------------------------------------
        print("\nSTEP 6: LLM role refinement...")
        t6 = time.time()
        final_segments = refine_roles_with_llm(final_segments)
        print(f"  ✓ LLM refinement done in {time.time() - t6:.2f}s")

        role_counts = {}
        for seg in final_segments:
            role = seg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        print(f"  ✓ Role distribution: {role_counts}")

        # ------------------------------------------------------------------
        # STEP 7 — Build and save session JSON
        # ------------------------------------------------------------------
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        duration = final_segments[-1]["end"] if final_segments else 0.0
        total_elapsed = time.time() - pipeline_start

        team_b_segments = []
        for seg in final_segments:
            team_b_seg = {
                "speaker": seg["role"].upper(),
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg.get("text_cleaned", seg["text"]),
                "confidence": 0.95
            }
            team_b_seg["_internal"] = {
                "original_speaker": seg["speaker"],
                "role_source": seg["role_source"],
                "original_text": seg["text"] if seg.get("text_cleaned") else None,
                "text_cleanup_applied": seg.get("text_cleanup_applied", False)
            }
            team_b_segments.append(team_b_seg)

        session_data = {
            "consultation_id": session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "audio_file": os.path.basename(audio_path),
            "duration": duration,
            "segments": team_b_segments,
            "metadata": {
                "total_segments": len(team_b_segments),
                "total_duration": duration,
                "diarization_status": diarization_status,
                "model": "whisper-local-base.en",
                "format_version": "team_b_compatible_v1",
                "processing_time_seconds": round(total_elapsed, 2)
            }
        }

        output_path = Path("output") / f"session_{session_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Session saved to: {output_path}")
        print(f"✓ TOTAL PROCESSING TIME: {total_elapsed:.2f}s  "
              f"(audio duration: {duration:.1f}s)")
        print("="*70 + "\n")

        return {
            **session_data,
            "output_file": str(output_path)
        }

    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ---------------------------------------------------------------------------
# Remaining endpoints — unchanged
# ---------------------------------------------------------------------------

@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    """Retrieve a saved transcript by session ID."""
    output_path = Path("output") / f"session_{session_id}.json"

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}."
        )

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        return JSONResponse(content=session_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading session: {str(e)}")


@router.post("/swap-roles/{session_id}")
async def swap_roles(session_id: str):
    """Swap Doctor and Patient role assignments (idempotent)."""
    output_path = Path("output") / f"session_{session_id}.json"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        for segment in session_data["segments"]:
            if segment["speaker"] == "DOCTOR":
                segment["speaker"] = "PATIENT"
            elif segment["speaker"] == "PATIENT":
                segment["speaker"] = "DOCTOR"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Roles swapped for session: {session_id}")
        return JSONResponse(content=session_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error swapping roles: {str(e)}")


@router.get("/recording-status")
async def get_recording_status():
    """Get current recording status."""
    return JSONResponse(content={"is_recording": recording_state["is_recording"]})


@router.get("/team-b/consultation/{consultation_id}")
async def get_consultation_for_team_b(consultation_id: str):
    """
    Team B NLP Integration Endpoint.
    Returns transcript in Team B's required format.
    """
    output_path = Path("output") / f"session_{consultation_id}.json"

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Consultation not found: {consultation_id}"
        )

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        team_b_response = {
            "consultation_id": session_data.get("consultation_id", consultation_id),
            "segments": session_data["segments"]
        }
        return JSONResponse(content=team_b_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading consultation: {str(e)}")
