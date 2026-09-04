"""
Quick test to verify all imports work correctly.
Run this after installation to verify everything is set up.
"""
import sys

def test_imports():
    """Test all critical imports."""
    print("Testing imports...\n")
    
    tests = [
        ("FastAPI", "fastapi", "FastAPI"),
        ("Uvicorn", "uvicorn", "__version__"),
        ("Faster-Whisper", "faster_whisper", "WhisperModel"),
        ("NumPy", "numpy", "__version__"),
        ("Librosa", "librosa", "__version__"),
        ("SoundFile", "soundfile", "__version__"),
        ("Pydub", "pydub", "AudioSegment"),
        ("Python-dotenv", "dotenv", "load_dotenv"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module, attr in tests:
        try:
            mod = __import__(module, fromlist=[attr])
            if hasattr(mod, attr):
                print(f"✓ {name:20s} - OK")
                passed += 1
            else:
                print(f"✗ {name:20s} - Import OK but {attr} not found")
                failed += 1
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")
    
    if failed == 0:
        print("✓ All dependencies installed correctly!")
        print("\nNext steps:")
        print("  1. Place an audio file in sample_audio/ folder")
        print("  2. Run: python test_transcription.py sample_audio/your_file.wav")
        print("  3. Or start server: python -m uvicorn app.main:app --reload")
        return True
    else:
        print("✗ Some dependencies failed to import.")
        print("\nTry running: pip install -r requirements-simple.txt")
        return False

def test_app_imports():
    """Test application module imports."""
    print("\nTesting application modules...\n")
    
    try:
        from app.audio.audio_loader import AudioLoader
        print("✓ AudioLoader")
        
        from app.audio.preprocessing import AudioPreprocessor
        print("✓ AudioPreprocessor")
        
        from app.transcription.whisper_service import WhisperService
        print("✓ WhisperService")
        
        from app.transcription.transcript_formatter import TranscriptFormatter
        print("✓ TranscriptFormatter")
        
        from app.speaker.diarization_interface import create_diarization_service
        print("✓ DiarizationInterface")
        
        from app.api.routes import router
        print("✓ API Routes")
        
        from app.main import app
        print("✓ FastAPI App")
        
        print("\n✓ All application modules loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Application module error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEPENDENCY CHECK")
    print("="*50 + "\n")
    
    deps_ok = test_imports()
    
    if deps_ok:
        app_ok = test_app_imports()
        
        if app_ok:
            print("\n" + "="*50)
            print("✓ SYSTEM READY!")
            print("="*50 + "\n")
            sys.exit(0)
    
    print("\n" + "="*50)
    print("✗ SETUP INCOMPLETE")
    print("="*50 + "\n")
    sys.exit(1)
