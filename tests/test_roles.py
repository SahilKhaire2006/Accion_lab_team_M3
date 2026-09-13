"""
Tests for role mapping logic.
"""
import pytest
from app.services.roles import label_roles


def test_label_roles_two_speakers_doctor_first():
    """Test role mapping with two speakers, Doctor first."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "text": "Hi", "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 9.0, "text": "How are you?", "speaker": "SPEAKER_00"}
    ]
    
    result = label_roles(segments, first_speaker_is="Doctor")
    
    assert result[0]["role"] == "Doctor"
    assert result[1]["role"] == "Patient"
    assert result[2]["role"] == "Doctor"


def test_label_roles_two_speakers_patient_first():
    """Test role mapping with two speakers, Patient first."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "text": "Hi", "speaker": "SPEAKER_01"}
    ]
    
    result = label_roles(segments, first_speaker_is="Patient")
    
    assert result[0]["role"] == "Patient"
    assert result[1]["role"] == "Doctor"


def test_label_roles_three_speakers():
    """Test role mapping with three speakers (third stays as raw label)."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "text": "Hi", "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 9.0, "text": "Also here", "speaker": "SPEAKER_02"}
    ]
    
    result = label_roles(segments, first_speaker_is="Doctor")
    
    assert result[0]["role"] == "Doctor"
    assert result[1]["role"] == "Patient"
    assert result[2]["role"] == "SPEAKER_02"  # Passed through


def test_label_roles_with_unknown():
    """Test role mapping when some segments are UNKNOWN."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "UNKNOWN"},
        {"start": 3.0, "end": 6.0, "text": "Hi", "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 9.0, "text": "Hey", "speaker": "SPEAKER_01"}
    ]
    
    result = label_roles(segments, first_speaker_is="Doctor")
    
    assert result[0]["role"] == "UNKNOWN"
    assert result[1]["role"] == "Doctor"  # First non-UNKNOWN speaker
    assert result[2]["role"] == "Patient"  # Second non-UNKNOWN speaker


def test_label_roles_single_speaker():
    """Test role mapping with only one speaker."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "text": "World", "speaker": "SPEAKER_00"}
    ]
    
    result = label_roles(segments, first_speaker_is="Doctor")
    
    assert result[0]["role"] == "Doctor"
    assert result[1]["role"] == "Doctor"


def test_label_roles_preserves_fields():
    """Test that label_roles preserves all original fields."""
    segments = [
        {"start": 1.5, "end": 3.7, "text": "Test", "speaker": "SPEAKER_00"}
    ]
    
    result = label_roles(segments, first_speaker_is="Doctor")
    
    assert result[0]["start"] == 1.5
    assert result[0]["end"] == 3.7
    assert result[0]["text"] == "Test"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["role"] == "Doctor"
