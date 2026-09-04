"""
Simple CLI test script for transcription pipeline.
"""
import sys
from pathlib import Path

from app.audio.audio_loader import AudioLoader
from app.audio.preprocessing import AudioPreprocessor
from app.transcription.whisper_service import WhisperService
from app.transcription.transcript_formatter import TranscriptFormatter
from app.speaker.diarization_interface import create_diarization_service


def test_transcription(audio_file: str, model: str = "base.en"):
    """
    Test the complete transcription pipeline.
    
    Args:
        audio_file: Path to audio file
        model: Whisper model to use
    """
    print("\n" + "="*70)
    print("PRIVACY-PRESERVING CLINICAL SCRIBE - TRANSCRIPTION TEST")
    print("="*70 + "\n")
    
    # Initialize components
    audio_loader = AudioLoader()
    preprocessor = AudioPreprocessor()
    whisper_service = WhisperService(model_name=model)
    transcript_formatter = TranscriptFormatter()
    diarization_service = create_diarization_service("placeholder")
    
    try:
        # Step 1: Load audio
        print("STEP 1: Loading audio file...")
        audio_data, sample_rate = audio_loader.load(audio_file)
        
        # Step 2: Preprocess
        print("\nSTEP 2: Preprocessing audio...")
        processed_audio = preprocessor.preprocess(audio_data, sample_rate)
        
        # Save preprocessed audio for inspection
        temp_path = "output/temp_processed.wav"
        audio_loader.save(processed_audio, temp_path, preprocessor.target_sample_rate)
        
        # Step 3: Transcribe
        print("\nSTEP 3: Transcribing with Whisper...")
        whisper_service.load_model()
        segments = whisper_service.transcribe(temp_path)
        
        # Step 4: Speaker diarization (placeholder)
        print("\nSTEP 4: Assigning speakers...")
        speaker_labels = diarization_service.assign_speakers(processed_audio, segments)
        
        # Step 5: Format and save
        print("\nSTEP 5: Formatting transcript...")
        transcript = transcript_formatter.format(
            segments=segments,
            speaker_labels=speaker_labels
        )
        
        output_path = transcript_formatter.save(transcript)
        
        # Display results
        print("\n" + "="*70)
        print("TRANSCRIPTION COMPLETE")
        print("="*70)
        print(f"\nSession ID: {transcript['session_id']}")
        print(f"Total segments: {len(transcript['segments'])}")
        print(f"Duration: {transcript['metadata']['total_duration']:.2f} seconds")
        print(f"Output saved to: {output_path}")
        print("\nSample segments:")
        for i, seg in enumerate(transcript['segments'][:3]):
            print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")
        if len(transcript['segments']) > 3:
            print(f"  ... and {len(transcript['segments']) - 3} more segments")
        
        print("\n" + "="*70 + "\n")
        
        return transcript
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_transcription.py <audio_file> [model]")
        print("Example: python test_transcription.py sample_audio/doctor_patient.wav base.en")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base.en"
    
    test_transcription(audio_file, model)
