"""
FastAPI routes for transcription service.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
from typing import Optional

from app.audio.audio_loader import AudioLoader
from app.audio.preprocessing import AudioPreprocessor
from app.transcription.whisper_service import WhisperService
from app.transcription.transcript_formatter import TranscriptFormatter
from app.speaker.diarization_interface import create_diarization_service


router = APIRouter()

# Initialize services
audio_loader = AudioLoader()
preprocessor = AudioPreprocessor()
transcript_formatter = TranscriptFormatter()

# Global whisper service (lazy loaded)
_whisper_service = None


def get_whisper_service(model_name: str = "base.en") -> WhisperService:
    """Get or create Whisper service instance."""
    global _whisper_service
    if _whisper_service is None:
        _whisper_service = WhisperService(model_name=model_name)
        _whisper_service.load_model()
    return _whisper_service


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    model: Optional[str] = Form("base.en"),
    save_output: Optional[bool] = Form(True)
):
    """
    Transcribe an audio file using local Whisper model.
    
    Args:
        file: Audio file (wav, mp3, m4a, flac, ogg)
        session_id: Optional session identifier
        model: Whisper model to use (tiny.en, base.en, small.en, etc.)
        save_output: Whether to save transcript to output directory
        
    Returns:
        JSON transcript with segments and timestamps
    """
    temp_dir = None
    
    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        temp_input_path = Path(temp_dir) / file.filename
        temp_processed_path = Path(temp_dir) / "processed.wav"
        
        # Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n{'='*60}")
        print(f"Processing: {file.filename}")
        print(f"{'='*60}")
        
        # Step 1: Load audio
        audio_data, sample_rate = audio_loader.load(str(temp_input_path))
        
        # Step 2: Preprocess audio
        processed_audio = preprocessor.preprocess(audio_data, sample_rate)
        
        # Save preprocessed audio
        audio_loader.save(processed_audio, str(temp_processed_path), preprocessor.target_sample_rate)
        
        # Step 3: Transcribe using Whisper
        whisper_service = get_whisper_service(model_name=model)
        segments = whisper_service.transcribe(str(temp_processed_path))
        
        # Step 4: Speaker diarization (placeholder)
        diarization_service = create_diarization_service("placeholder")
        speaker_labels = diarization_service.assign_speakers(processed_audio, segments)
        
        # Step 5: Format transcript
        transcript = transcript_formatter.format(
            segments=segments,
            session_id=session_id,
            speaker_labels=speaker_labels
        )
        
        # Step 6: Save output if requested
        if save_output:
            output_path = transcript_formatter.save(transcript)
            transcript["saved_to"] = output_path
        
        print(f"{'='*60}")
        print("✓ Processing complete!")
        print(f"{'='*60}\n")
        
        return JSONResponse(content=transcript)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Clinical Scribe Transcription Service",
        "version": "1.0.0"
    }


@router.get("/models")
async def list_models():
    """List available Whisper models."""
    return {
        "available_models": WhisperService.AVAILABLE_MODELS,
        "default_model": "base.en",
        "recommended": {
            "fastest": "tiny.en",
            "balanced": "base.en",
            "accurate": "small.en"
        }
    }
