"""
Real speaker diarization using pyannote.audio.
"""
import os
import time
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Global pipeline cache
_pipeline = None
_pipeline_load_time = None


def get_diarization_pipeline():
    """
    Get or create the diarization pipeline (singleton pattern).
    
    Returns:
        Diarization pipeline or None if loading fails
    """
    global _pipeline, _pipeline_load_time
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        from pyannote.audio import Pipeline
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token or hf_token == "your_huggingface_token_here":
            print("⚠️  WARNING: HF_TOKEN not set or is placeholder value")
            print("   Diarization will fall back to UNKNOWN speaker labels")
            print("   To enable diarization:")
            print("   1. Get token from https://huggingface.co/settings/tokens")
            print("   2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   3. Set HF_TOKEN in .env file")
            return None
        
        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading speaker diarization pipeline...")
        print(f"  Device: {device}")
        
        start_time = time.time()
        
        # Load pre-trained pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        # Move to appropriate device
        _pipeline.to(device)
        
        _pipeline_load_time = time.time() - start_time
        
        print(f"✓ Diarization pipeline loaded in {_pipeline_load_time:.2f}s")
        
        return _pipeline
        
    except Exception as e:
        print(f"⚠️  ERROR loading diarization pipeline: {e}")
        print("   Falling back to UNKNOWN speaker labels")
        return None


def diarize(audio_path: str, min_speakers: int = None, max_speakers: int = None) -> List[Dict]:
    """
    Perform speaker diarization on audio file.
    
    Args:
        audio_path: Path to audio file (WAV format preferred)
        min_speakers: Minimum number of speakers (optional, improves accuracy)
        max_speakers: Maximum number of speakers (optional, improves accuracy)
        
    Returns:
        List of diarization turns: [{"start": float, "end": float, "speaker": str}, ...]
        Sorted by start time. Returns empty list if diarization fails.
    """
    pipeline = get_diarization_pipeline()
    
    if pipeline is None:
        print("Diarization pipeline not available, returning empty result")
        return []
    
    try:
        print(f"Running speaker diarization on: {Path(audio_path).name}")
        if min_speakers is not None and max_speakers is not None:
            print(f"  Constraining to {min_speakers}-{max_speakers} speakers")
        start_time = time.time()
        
        # Run diarization with optional speaker constraints
        kwargs = {}
        if min_speakers is not None:
            kwargs['min_speakers'] = min_speakers
        if max_speakers is not None:
            kwargs['max_speakers'] = max_speakers
        
        output = pipeline(audio_path, **kwargs)
        
        # Extract speaker turns - support both old and new pyannote.audio API
        turns = []
        
        # NEW API (pyannote.audio >= 3.0): output has .speaker_diarization attribute
        if hasattr(output, 'speaker_diarization'):
            for turn, speaker in output.speaker_diarization:
                turns.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker
                })
        # OLD API (pyannote.audio < 3.0): output has .itertracks() method
        else:
            for turn, _, speaker in output.itertracks(yield_label=True):
                turns.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker
                })
        
        # Sort by start time
        turns.sort(key=lambda x: x["start"])
        
        # Count unique speakers
        unique_speakers = set(turn["speaker"] for turn in turns)
        
        elapsed = time.time() - start_time
        print(f"✓ Diarization complete in {elapsed:.2f}s")
        print(f"  Detected {len(unique_speakers)} speaker(s): {', '.join(sorted(unique_speakers))}")
        print(f"  Generated {len(turns)} speaker turn(s)")
        
        if len(unique_speakers) > 2:
            print(f"  ⚠️  Note: {len(unique_speakers)} speakers detected (more than 2)")
        
        return turns
        
    except Exception as e:
        print(f"⚠️  ERROR during diarization: {e}")
        print("   Returning empty result (will fall back to UNKNOWN)")
        return []
