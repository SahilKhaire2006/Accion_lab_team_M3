"""
Audio loader module for handling various audio file formats.
"""
import os
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np
from pydub import AudioSegment


class AudioLoader:
    """Handles loading audio files from various formats."""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    
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
        
        # Try soundfile first (faster for WAV, FLAC, OGG)
        try:
            audio_data, sample_rate = sf.read(file_path)
            print(f"Loaded audio using soundfile: {path.name}")
        except Exception as e:
            # Use pydub for formats soundfile doesn't support (MP3, M4A, AAC)
            print(f"Soundfile failed, using pydub for {path.suffix} format...")
            audio_data, sample_rate = self._load_with_pydub(file_path)
            print(f"Loaded audio using pydub: {path.name}")
        
        print(f"  Original sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
        
        # Determine channels
        if audio_data.ndim == 1:
            channels = 1
        elif audio_data.ndim == 2:
            channels = audio_data.shape[1]
        else:
            channels = 1
        
        print(f"  Channels: {channels}")
        
        return audio_data, sample_rate
    
    def _load_with_pydub(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio using pydub (supports MP3, M4A, etc. without ffmpeg in some cases).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        path = Path(file_path)
        
        # Determine format from extension
        format_map = {
            '.mp3': 'mp3',
            '.m4a': 'm4a',
            '.aac': 'aac',
            '.ogg': 'ogg',
            '.flac': 'flac',
            '.wav': 'wav'
        }
        
        audio_format = format_map.get(path.suffix.lower(), path.suffix[1:])
        
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(file_path, format=audio_format)
            
            # Get sample rate
            sample_rate = audio.frame_rate
            
            # Convert to numpy array
            # pydub stores audio as interleaved samples
            samples = np.array(audio.get_array_of_samples())
            
            # Convert to float32 in range [-1, 1]
            if audio.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0
            else:  # 8-bit or other
                samples = samples.astype(np.float32) / 128.0
            
            # Reshape for stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            return samples, sample_rate
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file {file_path}. "
                f"Error: {str(e)}. "
                f"Note: MP3/M4A support requires ffmpeg to be installed. "
                f"For now, please convert your file to WAV format or install ffmpeg."
            )
    
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
