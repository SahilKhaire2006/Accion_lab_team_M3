"""
Audio loader module for handling various audio file formats.
"""
import os
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np


class AudioLoader:
    """Handles loading audio files from various formats."""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    def __init__(self):
        self.sample_rate = 16000  # Whisper expects 16kHz
    
    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load an audio file and return audio data with sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Load audio file
        audio_data, sample_rate = sf.read(file_path)
        
        print(f"Loaded audio: {path.name}")
        print(f"  Original sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"  Channels: {audio_data.ndim}")
        
        return audio_data, sample_rate
    
    def save(self, audio_data: np.ndarray, output_path: str, sample_rate: int = 16000):
        """
        Save audio data to a file.
        
        Args:
            audio_data: Audio data array
            output_path: Path to save the audio file
            sample_rate: Sample rate of the audio
        """
        sf.write(output_path, audio_data, sample_rate)
        print(f"Saved audio to: {output_path}")
