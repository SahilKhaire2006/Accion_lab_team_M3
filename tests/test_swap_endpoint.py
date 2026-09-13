"""
Tests for role swap endpoint.
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from starlette.testclient import TestClient
from app.main import app


def get_client():
    """Get a test client instance."""
    return TestClient(app)


def create_test_session(session_id: str, segments: list) -> Path:
    """Helper to create a test session file."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    session_data = {
        "session_id": session_id,
        "created_at": "2026-09-08T12:00:00Z",
        "audio_file": "test.wav",
        "duration": 10.0,
        "segments": segments,
        "metadata": {
            "total_segments": len(segments),
            "total_duration": 10.0,
            "diarization_status": "enabled"
        }
    }
    
    file_path = output_dir / f"session_{session_id}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2)
    
    return file_path


def test_swap_roles_basic():
    """Test basic role swapping (Doctor ↔ Patient)."""
    client = get_client()
    session_id = "test_swap_basic"
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello", "speaker": "SPEAKER_00", "role": "Doctor"},
        {"start": 3.0, "end": 6.0, "text": "Hi", "speaker": "SPEAKER_01", "role": "Patient"}
    ]
    
    file_path = create_test_session(session_id, segments)
    
    try:
        # First swap
        response = client.post(f"/api/v1/live/swap-roles/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["segments"][0]["role"] == "Patient"
        assert data["segments"][1]["role"] == "Doctor"
        
        # Second swap (idempotence test)
        response = client.post(f"/api/v1/live/swap-roles/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["segments"][0]["role"] == "Doctor"  # Back to original
        assert data["segments"][1]["role"] == "Patient"
        
    finally:
        if file_path.exists():
            file_path.unlink()


def test_swap_roles_preserves_other_speakers():
    """Test that swap preserves other speakers (SPEAKER_02+)."""
    client = get_client()
    session_id = "test_swap_other"
    segments = [
        {"start": 0.0, "end": 3.0, "text": "One", "speaker": "SPEAKER_00", "role": "Doctor"},
        {"start": 3.0, "end": 6.0, "text": "Two", "speaker": "SPEAKER_01", "role": "Patient"},
        {"start": 6.0, "end": 9.0, "text": "Three", "speaker": "SPEAKER_02", "role": "SPEAKER_02"}
    ]
    
    file_path = create_test_session(session_id, segments)
    
    try:
        response = client.post(f"/api/v1/live/swap-roles/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["segments"][0]["role"] == "Patient"
        assert data["segments"][1]["role"] == "Doctor"
        assert data["segments"][2]["role"] == "SPEAKER_02"  # Unchanged
        
    finally:
        if file_path.exists():
            file_path.unlink()


def test_swap_roles_preserves_unknown():
    """Test that swap preserves UNKNOWN roles."""
    client = get_client()
    session_id = "test_swap_unknown"
    segments = [
        {"start": 0.0, "end": 3.0, "text": "One", "speaker": "UNKNOWN", "role": "UNKNOWN"},
        {"start": 3.0, "end": 6.0, "text": "Two", "speaker": "SPEAKER_00", "role": "Doctor"}
    ]
    
    file_path = create_test_session(session_id, segments)
    
    try:
        response = client.post(f"/api/v1/live/swap-roles/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["segments"][0]["role"] == "UNKNOWN"  # Unchanged
        assert data["segments"][1]["role"] == "Patient"  # Swapped
        
    finally:
        if file_path.exists():
            file_path.unlink()


def test_swap_roles_nonexistent_session():
    """Test swap with non-existent session returns 404."""
    client = get_client()
    response = client.post("/api/v1/live/swap-roles/nonexistent_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_transcript_success():
    """Test retrieving an existing transcript."""
    client = get_client()
    session_id = "test_get_transcript"
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Test", "speaker": "SPEAKER_00", "role": "Doctor"}
    ]
    
    file_path = create_test_session(session_id, segments)
    
    try:
        response = client.get(f"/api/v1/live/transcript/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session_id
        assert len(data["segments"]) == 1
        
    finally:
        if file_path.exists():
            file_path.unlink()


def test_get_transcript_not_found():
    """Test retrieving non-existent transcript returns 404."""
    client = get_client()
    response = client.get("/api/v1/live/transcript/nonexistent_99999")
    assert response.status_code == 404
