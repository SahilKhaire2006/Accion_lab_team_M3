"""
Speaker diarization interface (placeholder for future implementation).

This module provides the architecture for integrating speaker diarization
in Week 3 of the project. Currently returns UNKNOWN for all speakers.
"""
from typing import List, Dict, Optional
import numpy as np
from abc import ABC, abstractmethod


class DiarizationInterface(ABC):
    """Abstract interface for speaker diarization systems."""
    
    @abstractmethod
    def load_model(self):
        """Load the diarization model."""
        pass
    
    @abstractmethod
    def assign_speakers(
        self, 
        audio_data: np.ndarray, 
        segments: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign speaker labels to transcript segments.
        
        Args:
            audio_data: Audio data as numpy array
            segments: List of transcript segments
            
        Returns:
            Dictionary mapping segment index to speaker ID
        """
        pass
    
    @abstractmethod
    def merge_with_transcript(
        self, 
        segments: List[Dict], 
        speaker_labels: Dict[int, str]
    ) -> List[Dict]:
        """
        Merge speaker labels with transcript segments.
        
        Args:
            segments: List of transcript segments
            speaker_labels: Dictionary mapping segment index to speaker ID
            
        Returns:
            Updated segments with speaker labels
        """
        pass


class PlaceholderDiarization(DiarizationInterface):
    """
    Placeholder implementation for speaker diarization.
    
    Returns "UNKNOWN" for all speakers until actual diarization is implemented.
    
    TODO (Week 3):
    - Integrate pyannote.audio or similar diarization library
    - Implement speaker clustering
    - Align speaker segments with transcript segments
    - Add speaker identification/verification
    - Implement speaker embedding extraction
    """
    
    def __init__(self):
        self.model = None
        print("Warning: Using placeholder diarization (all speakers = UNKNOWN)")
    
    def load_model(self):
        """
        Load the diarization model.
        
        TODO: Implement model loading
        Example:
            from pyannote.audio import Pipeline
            self.model = Pipeline.from_pretrained("pyannote/speaker-diarization")
        """
        print("TODO: Load diarization model")
        self.model = None  # Placeholder
    
    def assign_speakers(
        self, 
        audio_data: np.ndarray, 
        segments: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign speaker labels to transcript segments.
        
        Currently returns UNKNOWN for all segments.
        
        TODO: Implement actual speaker assignment
        Steps:
        1. Run diarization on audio_data
        2. Extract speaker segments with timestamps
        3. Match speaker segments with transcript segments
        4. Assign speaker labels based on temporal overlap
        
        Args:
            audio_data: Audio data as numpy array
            segments: List of transcript segments
            
        Returns:
            Dictionary mapping segment index to speaker ID
        """
        # Placeholder: all segments assigned to UNKNOWN
        speaker_labels = {}
        for idx in range(len(segments)):
            speaker_labels[idx] = "UNKNOWN"
        
        print(f"Assigned speakers to {len(segments)} segments (placeholder mode)")
        
        return speaker_labels
    
    def merge_with_transcript(
        self, 
        segments: List[Dict], 
        speaker_labels: Dict[int, str]
    ) -> List[Dict]:
        """
        Merge speaker labels with transcript segments.
        
        Args:
            segments: List of transcript segments
            speaker_labels: Dictionary mapping segment index to speaker ID
            
        Returns:
            Updated segments with speaker labels
        """
        merged_segments = []
        for idx, segment in enumerate(segments):
            merged_segment = segment.copy()
            merged_segment["speaker"] = speaker_labels.get(idx, "UNKNOWN")
            merged_segments.append(merged_segment)
        
        return merged_segments


# Factory function for future extensibility
def create_diarization_service(service_type: str = "placeholder") -> DiarizationInterface:
    """
    Factory function to create diarization service.
    
    Args:
        service_type: Type of diarization service to create
        
    Returns:
        DiarizationInterface implementation
    """
    if service_type == "placeholder":
        return PlaceholderDiarization()
    else:
        raise ValueError(f"Unknown diarization service: {service_type}")
