"""
Transcript formatter for generating structured JSON output.
"""
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path


class TranscriptFormatter:
    """Formats transcription segments into structured JSON."""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def format(
        self, 
        segments: List[Dict],
        session_id: str = None,
        speaker_labels: Dict[int, str] = None
    ) -> Dict:
        """
        Format transcript segments into structured JSON.
        
        Args:
            segments: List of transcript segments with start, end, text
            session_id: Optional session identifier
            speaker_labels: Optional mapping of segment index to speaker ID
            
        Returns:
            Structured transcript dictionary
        """
        if session_id is None:
            session_id = self._generate_session_id()
        
        # Add speaker labels to segments
        formatted_segments = []
        for idx, segment in enumerate(segments):
            formatted_segment = {
                "speaker": speaker_labels.get(idx, "UNKNOWN") if speaker_labels else "UNKNOWN",
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            formatted_segments.append(formatted_segment)
        
        # Create structured output
        transcript = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": "whisper-local",
            "segments": formatted_segments,
            "metadata": {
                "total_segments": len(formatted_segments),
                "total_duration": segments[-1]["end"] if segments else 0.0,
                "speaker_diarization": "not_implemented"
            }
        }
        
        return transcript
    
    def save(self, transcript: Dict, output_path: str = None) -> str:
        """
        Save transcript to JSON file.
        
        Args:
            transcript: Formatted transcript dictionary
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"session_{timestamp}.json"
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Transcript saved to: {output_path}")
        
        return str(output_path)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def to_json_string(self, transcript: Dict, pretty: bool = True) -> str:
        """
        Convert transcript to JSON string.
        
        Args:
            transcript: Formatted transcript dictionary
            pretty: Whether to format with indentation
            
        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(transcript, indent=2, ensure_ascii=False)
        return json.dumps(transcript, ensure_ascii=False)
