"""
Create a simple test audio file for testing transcription.
This uses text-to-speech to generate a sample doctor-patient conversation.
"""
import numpy as np
from scipy.io import wavfile
import os

def generate_tone(frequency, duration, sample_rate=16000):
    """Generate a simple sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    tone = np.sin(2 * np.pi * frequency * t)
    return tone

def create_sample_wav():
    """Create a simple audio file for testing."""
    sample_rate = 16000
    duration = 5  # 5 seconds
    
    # Generate a simple tone (since we don't have TTS)
    audio = generate_tone(440, duration, sample_rate)  # A4 note
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    # Save
    output_dir = "sample_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_sample.wav")
    
    wavfile.write(output_path, sample_rate, audio)
    print(f"✓ Created sample audio file: {output_path}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print("\nNOTE: This is a simple tone for testing the pipeline.")
    print("For real transcription testing, please use actual speech audio.")
    print("\nYou can:")
    print("  1. Record your own audio")
    print("  2. Download sample audio from: https://www.voiptroubleshooter.com/open_speech/american.html")
    print("  3. Use any WAV/MP3 file with speech")

if __name__ == "__main__":
    try:
        create_sample_wav()
    except ImportError:
        print("scipy not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        print("Please run this script again.")
