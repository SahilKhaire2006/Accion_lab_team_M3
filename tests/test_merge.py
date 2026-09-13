"""
Tests for transcript-diarization merging logic.
"""
import pytest
from app.services.merge import merge_transcript_with_speakers, calculate_overlap


def test_calculate_overlap_full_overlap():
    """Test overlap calculation when segments fully overlap."""
    overlap = calculate_overlap(0.0, 5.0, 0.0, 5.0)
    assert overlap == 5.0


def test_calculate_overlap_partial():
    """Test overlap calculation with partial overlap."""
    overlap = calculate_overlap(0.0, 5.0, 3.0, 8.0)
    assert overlap == 2.0


def test_calculate_overlap_no_overlap():
    """Test overlap calculation when segments don't overlap."""
    overlap = calculate_overlap(0.0, 2.0, 5.0, 8.0)
    assert overlap == 0.0


def test_merge_with_perfect_alignment():
    """Test merging when transcript and diarization align perfectly."""
    whisper_segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello doctor"},
        {"start": 3.0, "end": 6.0, "text": "Hello patient"}
    ]
    
    diarization_turns = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"}
    ]
    
    result = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Hello doctor"
    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["text"] == "Hello patient"


def test_merge_with_overlapping_turns():
    """Test merging when one transcript segment overlaps multiple speaker turns."""
    whisper_segments = [
        {"start": 1.0, "end": 4.0, "text": "Long sentence"}
    ]
    
    diarization_turns = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},  # 1.5s overlap
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"}   # 1.5s overlap
    ]
    
    result = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    
    assert len(result) == 1
    # Both have equal overlap, should pick first one (SPEAKER_00)
    assert result[0]["speaker"] in ["SPEAKER_00", "SPEAKER_01"]


def test_merge_with_no_overlap():
    """Test merging when transcript segment has no overlapping speaker turn."""
    whisper_segments = [
        {"start": 10.0, "end": 12.0, "text": "Late segment"}
    ]
    
    diarization_turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}
    ]
    
    result = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    
    assert len(result) == 1
    assert result[0]["speaker"] == "UNKNOWN"


def test_merge_with_empty_diarization():
    """Test merging when diarization returns no turns."""
    whisper_segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello"}
    ]
    
    diarization_turns = []
    
    result = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    
    assert len(result) == 1
    assert result[0]["speaker"] == "UNKNOWN"


def test_merge_preserves_whisper_data():
    """Test that merge preserves all Whisper segment fields."""
    whisper_segments = [
        {"start": 1.5, "end": 3.7, "text": "Test message"}
    ]
    
    diarization_turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}
    ]
    
    result = merge_transcript_with_speakers(whisper_segments, diarization_turns)
    
    assert result[0]["start"] == 1.5
    assert result[0]["end"] == 3.7
    assert result[0]["text"] == "Test message"
    assert result[0]["speaker"] == "SPEAKER_00"
