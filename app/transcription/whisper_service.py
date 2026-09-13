"""
Whisper-based transcription service for local speech-to-text processing.
"""
from faster_whisper import WhisperModel
from typing import List, Dict
import numpy as np
from pathlib import Path


class WhisperService:
    """Handles local Whisper transcription."""
    
    AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v2"]
    
    def __init__(self, model_name: str = "base.en", device: str = "cpu"):
        """
        Initialize Whisper service.
        
        Args:
            model_name: Whisper model to use (tiny.en, base.en, etc.)
            device: Device to run on ("cpu" or "cuda")
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not recognized. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None
        
        print(f"Initializing Whisper model: {model_name}")
        print(f"Device: {device}")
    
    def load_model(self):
        """Load the Whisper model into memory."""
        if self.model is None:
            print(f"Loading model '{self.model_name}'...")
            self.model = WhisperModel(
                self.model_name, 
                device=self.device,
                compute_type="int8"  # Optimized for CPU
            )
            print("Model loaded successfully!")
    
    def transcribe(
        self, 
        audio_path: str,
        language: str = "en"
    ) -> List[Dict]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")
            
        Returns:
            List of segments with timestamps and text
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        print(f"\nTranscribing audio: {Path(audio_path).name}")
        print("This may take a moment...")
        
        # Run transcription with word-level timestamps
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice Activity Detection
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True  # Enable word-level timestamps for segment splitting
        )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Extract segments with word-level timestamps
        transcript_segments = []
        for segment in segments:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            }
            
            # Include word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                segment_data["words"] = [
                    {
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "word": word.word
                    }
                    for word in segment.words
                ]
            
            transcript_segments.append(segment_data)
            print(f"  [{segment_data['start']:.2f}s - {segment_data['end']:.2f}s] {segment_data['text']}")
        
        print(f"\nTranscription complete! ({len(transcript_segments)} segments)")
        
        return transcript_segments
    
    def transcribe_numpy(
        self, 
        audio_data: np.ndarray,
        language: str = "en"
    ) -> List[Dict]:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_data: Audio data as numpy array
            language: Language code (default: "en")
            
        Returns:
            List of segments with timestamps and text
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        print(f"\nTranscribing audio from numpy array...")
        print("This may take a moment...")
        
        # Run transcription
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Extract segments
        transcript_segments = []
        for segment in segments:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            }
            transcript_segments.append(segment_data)
        
        print(f"\nTranscription complete! ({len(transcript_segments)} segments)")
        
        return transcript_segments
