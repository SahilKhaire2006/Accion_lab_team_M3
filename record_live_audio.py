"""
Live audio recording and transcription for clinical scribe.
Records audio from microphone and automatically transcribes it.
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import sys
import os

# Import application modules
from app.audio.audio_loader import AudioLoader
from app.audio.preprocessing import AudioPreprocessor
from app.transcription.whisper_service import WhisperService
from app.transcription.transcript_formatter import TranscriptFormatter
from app.speaker.diarization_interface import create_diarization_service


def list_audio_devices():
    """List all available audio input devices."""
    print("\n" + "="*70)
    print("AVAILABLE AUDIO DEVICES")
    print("="*70)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (Input channels: {device['max_input_channels']})")
    print("="*70 + "\n")


def record_audio(duration=30, sample_rate=48000, device=None):
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate (48000 Hz default)
        device: Device ID (None = default microphone)
        
    Returns:
        audio_data: Recorded audio as numpy array
    """
    print(f"\n🎙️  Recording will start in 3 seconds...")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("\n3...")
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("\n🔴 RECORDING NOW! Speak into your microphone...\n")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,  # Stereo
        dtype='float32',
        device=device
    )
    
    sd.wait()  # Wait until recording is finished
    
    print("\n✓ Recording complete!")
    
    return audio_data, sample_rate


def save_recording(audio_data, sample_rate, output_dir="sample_audio"):
    """
    Save recorded audio to file.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_recording_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    sf.write(filepath, audio_data, sample_rate)
    
    print(f"✓ Saved to: {filepath}")
    
    return filepath


def transcribe_live_recording(audio_path, model="base.en", auto_transcribe=True):
    """
    Transcribe the recorded audio using the clinical scribe pipeline.
    
    Args:
        audio_path: Path to recorded audio file
        model: Whisper model to use
        auto_transcribe: If True, automatically transcribe after recording
        
    Returns:
        transcript: Formatted transcript dictionary
    """
    print("\n" + "="*70)
    print("TRANSCRIBING LIVE RECORDING")
    print("="*70 + "\n")
    
    # Initialize components
    audio_loader = AudioLoader()
    preprocessor = AudioPreprocessor()
    whisper_service = WhisperService(model_name=model)
    transcript_formatter = TranscriptFormatter()
    diarization_service = create_diarization_service("placeholder")
    
    try:
        # Step 1: Load audio
        print("STEP 1: Loading audio...")
        audio_data, sample_rate = audio_loader.load(audio_path)
        
        # Step 2: Preprocess
        print("\nSTEP 2: Preprocessing audio...")
        processed_audio = preprocessor.preprocess(audio_data, sample_rate)
        
        # Save preprocessed audio
        temp_path = "output/temp_processed.wav"
        os.makedirs("output", exist_ok=True)
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
        print("\nTranscript preview:")
        print("-" * 70)
        for i, seg in enumerate(transcript['segments'][:5]):
            print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")
        if len(transcript['segments']) > 5:
            print(f"... and {len(transcript['segments']) - 5} more segments")
        print("="*70 + "\n")
        
        return transcript
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main recording and transcription workflow."""
    print("\n" + "="*70)
    print("LIVE AUDIO RECORDER & TRANSCRIBER - Clinical Scribe")
    print("="*70)
    
    # List available devices
    list_audio_devices()
    
    # Get recording parameters
    try:
        duration_input = input("Enter recording duration in seconds (default: 30): ").strip()
        duration = int(duration_input) if duration_input else 30
        
        device_input = input("Enter device ID (press Enter for default microphone): ").strip()
        device = int(device_input) if device_input else None
        
        model_input = input("Enter Whisper model (tiny.en/base.en/small.en, default: base.en): ").strip()
        model = model_input if model_input else "base.en"
        
        auto_transcribe = input("Auto-transcribe after recording? (Y/n): ").strip().lower() != 'n'
        
    except ValueError:
        print("Invalid input. Using default values.")
        duration = 30
        device = None
        model = "base.en"
        auto_transcribe = True
    
    try:
        # Record audio
        audio_data, sample_rate = record_audio(
            duration=duration,
            sample_rate=48000,
            device=device
        )
        
        # Save recording
        filepath = save_recording(audio_data, sample_rate)
        
        # Auto-transcribe if requested
        if auto_transcribe:
            transcript = transcribe_live_recording(filepath, model=model)
        else:
            # Instructions for manual transcription
            print("\n" + "="*70)
            print("RECORDING SAVED")
            print("="*70)
            print(f"\nTo transcribe this recording later, run:")
            print(f"\n  python test_transcription.py {filepath} {model}")
            print(f"\nOr use the API:")
            print(f"  1. Start server: python -m uvicorn app.main:app --reload")
            print(f"  2. Open: http://localhost:8000/docs")
            print(f"  3. Upload: {filepath}")
            print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure your microphone is connected and enabled")
        print("  - Try selecting a different device ID")
        print("  - Check Windows sound settings")
        print("  - Ensure ffmpeg is in your PATH")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
