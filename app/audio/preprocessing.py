"""
Audio preprocessing module for normalizing and preparing audio for transcription.
"""
import numpy as np
import librosa
from typing import Optional


class AudioPreprocessor:
    """Preprocesses audio files for optimal Whisper transcription."""
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize preprocessor.
        
        Args:
            target_sample_rate: Target sample rate (Whisper uses 16kHz)
        """
        self.target_sample_rate = target_sample_rate
    
    def preprocess(
        self, 
        audio_data: np.ndarray, 
        original_sample_rate: int
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Steps:
        1. Convert to mono
        2. Resample to 16kHz
        3. Normalize volume
        
        Args:
            audio_data: Input audio data
            original_sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio data
        """
        print("Preprocessing audio...")
        
        # Step 1: Convert to mono
        audio_mono = self._to_mono(audio_data)
        print(f"  ✓ Converted to mono")
        
        # Step 2: Resample to target rate
        if original_sample_rate != self.target_sample_rate:
            audio_resampled = self._resample(
                audio_mono, 
                original_sample_rate, 
                self.target_sample_rate
            )
            print(f"  ✓ Resampled to {self.target_sample_rate} Hz")
        else:
            audio_resampled = audio_mono
            print(f"  ✓ Already at {self.target_sample_rate} Hz")
        
        # Step 3: Normalize volume
        audio_normalized = self._normalize(audio_resampled)
        print(f"  ✓ Normalized volume")
        
        return audio_normalized
    
    def _to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono.
        
        Args:
            audio_data: Input audio (can be mono or stereo)
            
        Returns:
            Mono audio data
        """
        if audio_data.ndim == 1:
            # Already mono
            return audio_data
        elif audio_data.ndim == 2:
            # Convert stereo to mono by averaging channels
            return np.mean(audio_data, axis=1)
        else:
            raise ValueError(f"Unsupported audio shape: {audio_data.shape}")
    
    def _resample(
        self, 
        audio_data: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        return librosa.resample(
            audio_data, 
            orig_sr=orig_sr, 
            target_sr=target_sr
        )
    
    def _normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio volume.
        
        Args:
            audio_data: Input audio
            
        Returns:
            Normalized audio
        """
        # Normalize to [-1, 1] range
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            return audio_data / max_val
        return audio_data
