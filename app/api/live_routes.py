"""
API routes for live audio recording and transcription.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
import threading
import queue

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


def audio_callback(indata, frames, time, status):
    """Callback function for audio recording."""
    if status:
        print(f"Audio callback status: {status}")
    
    # Add audio data to queue
    if recording_state["audio_queue"] is not None:
        recording_state["audio_queue"].put(indata.copy())


@router.post("/start-recording")
async def start_recording():
    """Start live audio recording."""
    global recording_state
    
    if recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="Recording already in progress")
    
    try:
        # Initialize recording state
        recording_state["is_recording"] = True
        recording_state["audio_queue"] = queue.Queue()
        recording_state["audio_data"] = []
        
        # Get default input device
        device_info = sd.query_devices(kind='input')
        print(f"Using microphone: {device_info['name']}")
        
        # Start recording stream
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
    """Stop recording and transcribe audio with speaker diarization."""
    global recording_state
    
    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="No recording in progress")
    
    try:
        # Stop recording
        recording_state["stream"].stop()
        recording_state["stream"].close()
        recording_state["is_recording"] = False
        
        # Collect all audio data from queue
        audio_chunks = []
        while not recording_state["audio_queue"].empty():
            audio_chunks.append(recording_state["audio_queue"].get())
        
        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio data recorded")
        
        # Concatenate audio chunks
        audio_data = np.concatenate(audio_chunks, axis=0)
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_dir = Path("sample_audio")
        audio_dir.mkdir(exist_ok=True)
        audio_filename = f"live_recording_{timestamp}.wav"
        audio_path = audio_dir / audio_filename
        
        sf.write(str(audio_path), audio_data, recording_state["sample_rate"])
        
        # Transcribe audio with new diarization pipeline
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


async def transcribe_audio(audio_path: str, session_id: str = None):
    """
    Transcribe recorded audio with speaker diarization.
    
    Pipeline: transcribe → diarize → merge → label_roles → save
    """
    try:
        # Initialize components
        audio_loader = AudioLoader()
        preprocessor = AudioPreprocessor()
        whisper_service = WhisperService(model_name="base.en")
        
        print("\n" + "="*70)
        print("PROCESSING RECORDING")
        print("="*70)
        
        # Step 1: Load and preprocess audio
        print("\nSTEP 1: Loading and preprocessing audio...")
        audio_data, sample_rate = audio_loader.load(audio_path)
        processed_audio = preprocessor.preprocess(audio_data, sample_rate)
        
        # Save preprocessed audio
        temp_path = "output/temp_live_processed.wav"
        os.makedirs("output", exist_ok=True)
        audio_loader.save(processed_audio, temp_path, preprocessor.target_sample_rate)
        
        # Step 2: Transcribe with Whisper
        print("\nSTEP 2: Transcribing with Whisper...")
        whisper_service.load_model()
        whisper_segments = whisper_service.transcribe(temp_path)
        
        # Step 2.5: Detect and cleanup garbled segments (opt-in, targeted)
        print("\nSTEP 2.5: Checking for garbled segments...")
        whisper_segments = detect_garbled_segments(whisper_segments)
        if any(seg.get("likely_garbled") for seg in whisper_segments):
            whisper_segments = cleanup_garbled_segments(whisper_segments)
        else:
            print("  No garbled segments detected")
        
        # Step 3: Diarization (with graceful fallback)
        print("\nSTEP 3: Running speaker diarization...")
        diarization_status = "enabled"
        try:
            # Use speaker count constraints for doctor-patient conversation
            diarization_turns = diarize(temp_path, min_speakers=2, max_speakers=2)
            if not diarization_turns:
                print("  Diarization returned no results, falling back to UNKNOWN")
                diarization_status = "failed"
        except Exception as e:
            print(f"  ⚠️  Diarization error: {e}")
            print("  Falling back to UNKNOWN speaker labels")
            diarization_turns = []
            diarization_status = "failed"
        
        # Step 4: Merge diarization with transcript
        print("\nSTEP 4: Merging speaker labels with transcript...")
        merged_segments = merge_transcript_with_speakers(whisper_segments, diarization_turns)
        print(f"  ✓ Merged {len(merged_segments)} segments")
        
        # Step 5: Label roles (Doctor/Patient)
        print("\nSTEP 5: Mapping speakers to roles...")
        final_segments = label_roles(merged_segments, first_speaker_is="Doctor")
        
        # Step 6: LLM Refinement (mandatory quality assurance)
        print("\nSTEP 6: LLM role refinement...")
        final_segments = refine_roles_with_llm(final_segments)
        
        # Count roles
        role_counts = {}
        for seg in final_segments:
            role = seg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        print(f"  ✓ Role distribution: {role_counts}")
        
        # Step 7: Build session JSON (compatible with Team B NLP format)
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        duration = final_segments[-1]["end"] if final_segments else 0.0
        
        # Transform segments to Team B compatible format
        team_b_segments = []
        for seg in final_segments:
            team_b_seg = {
                "speaker": seg["role"].upper(),  # "Doctor" -> "DOCTOR", "Patient" -> "PATIENT"
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg.get("text_cleaned", seg["text"]),  # Use cleaned text if available
                "confidence": 0.95  # Default confidence (can be improved with actual Whisper confidence)
            }
            
            # Add internal metadata (for our own use)
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
                "format_version": "team_b_compatible_v1"
            }
        }
        
        # Save session JSON
        output_path = Path("output") / f"session_{session_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Session saved to: {output_path}")
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


@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    """
    Retrieve a saved transcript by session ID.
    
    Args:
        session_id: Session identifier (timestamp format: YYYYMMDD_HHMMSS)
        
    Returns:
        Complete session JSON with segments
    """
    output_path = Path("output") / f"session_{session_id}.json"
    
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Check that the session ID is correct."
        )
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        return JSONResponse(content=session_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading session: {str(e)}"
        )


@router.post("/swap-roles/{session_id}")
async def swap_roles(session_id: str):
    """
    Swap Doctor and Patient role assignments (idempotent).
    
    Args:
        session_id: Session identifier
        
    Returns:
        Updated session JSON with swapped roles
    """
    output_path = Path("output") / f"session_{session_id}.json"
    
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Load session
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Swap DOCTOR ↔ PATIENT in all segments (Team B format)
        for segment in session_data["segments"]:
            if segment["speaker"] == "DOCTOR":
                segment["speaker"] = "PATIENT"
            elif segment["speaker"] == "PATIENT":
                segment["speaker"] = "DOCTOR"
            # Leave other speakers unchanged
        
        # Save updated session
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Roles swapped for session: {session_id}")
        
        return JSONResponse(content=session_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error swapping roles: {str(e)}"
        )


@router.get("/recording-status")
async def get_recording_status():
    """Get current recording status."""
    return JSONResponse(content={
        "is_recording": recording_state["is_recording"]
    })


@router.get("/team-b/consultation/{consultation_id}")
async def get_consultation_for_team_b(consultation_id: str):
    """
    Team B NLP Integration Endpoint
    
    Returns transcript in the format required by Team B's NLP pipeline.
    
    Format:
    {
        "consultation_id": "consult_001",
        "segments": [
            {
                "speaker": "DOCTOR",
                "start_time": 0.0,
                "end_time": 3.5,
                "text": "Good morning. What brings you in today?",
                "confidence": 0.97
            }
        ]
    }
    
    Args:
        consultation_id: Session/consultation identifier
        
    Returns:
        Consultation transcript in Team B format
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
        
        # Format is already Team B compatible from transcription
        # Return only the essential fields they need
        team_b_response = {
            "consultation_id": session_data.get("consultation_id", consultation_id),
            "segments": session_data["segments"]
        }
        
        return JSONResponse(content=team_b_response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading consultation: {str(e)}"
        )
