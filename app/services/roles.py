"""
Role mapping for clinical conversations (Doctor/Patient).
"""
from typing import List, Dict


def label_roles(
    merged_segments: List[Dict],
    first_speaker_is: str = "Doctor"
) -> List[Dict]:
    """
    Map raw speaker labels to clinical roles.
    
    The first distinct speaker gets mapped to first_speaker_is (default: "Doctor"),
    the second distinct speaker gets the other role ("Patient" if first is Doctor),
    and any additional speakers (SPEAKER_02+) are passed through unchanged.
    
    Args:
        merged_segments: List of segments with "speaker" field
        first_speaker_is: Role to assign to first speaker ("Doctor" or "Patient")
        
    Returns:
        List of segments with added "role" field
    """
    # Identify first two distinct speakers in chronological order
    seen_speakers = []
    for seg in merged_segments:
        speaker = seg["speaker"]
        if speaker not in seen_speakers and speaker != "UNKNOWN":
            seen_speakers.append(speaker)
            if len(seen_speakers) >= 2:
                break
    
    # Create role mapping
    role_map = {}
    
    if len(seen_speakers) >= 1:
        if first_speaker_is == "Doctor":
            role_map[seen_speakers[0]] = "Doctor"
            if len(seen_speakers) >= 2:
                role_map[seen_speakers[1]] = "Patient"
        elif first_speaker_is == "Patient":
            role_map[seen_speakers[0]] = "Patient"
            if len(seen_speakers) >= 2:
                role_map[seen_speakers[1]] = "Doctor"
        else:
            # Invalid first_speaker_is, use default
            role_map[seen_speakers[0]] = "Doctor"
            if len(seen_speakers) >= 2:
                role_map[seen_speakers[1]] = "Patient"
    
    # Map all other speakers to themselves (e.g., SPEAKER_02 stays as SPEAKER_02)
    # and UNKNOWN stays as UNKNOWN
    
    # Apply role mapping
    result = []
    for seg in merged_segments:
        speaker = seg["speaker"]
        role = role_map.get(speaker, speaker)  # Default to speaker label if not mapped
        
        result.append({
            **seg,
            "role": role
        })
    
    return result
