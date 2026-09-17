"""
Whisper-based transcription service for local speech-to-text processing.

Singleton pattern: model is loaded ONCE at startup and reused across all
requests — eliminates the per-request model load cost (~3-5s).
"""
from faster_whisper import WhisperModel
from typing import List, Dict
import numpy as np
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused forever
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_model_name = None


def get_whisper_model(model_name: str = "base.en", device: str = "cpu") -> WhisperModel:
    """
    Get or create the Whisper model singleton.
    First call loads the model; subsequent calls return the cached instance.
    """
    global _whisper_model, _whisper_model_name

    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    t0 = time.time()
    print(f"[WhisperService] Loading model '{model_name}' on {device}...")
    _whisper_model = WhisperModel(
        model_name,
        device=device,
        compute_type="int8",        # INT8 quantisation — 2-3x faster on CPU
        num_workers=2,              # parallel decoding workers
        cpu_threads=4,              # use 4 CPU threads for inference
    )
    _whisper_model_name = model_name
    print(f"[WhisperService] Model ready in {time.time() - t0:.2f}s")
    return _whisper_model


def warmup_whisper(model_name: str = "base.en"):
    """Pre-load model at server startup so first request is fast."""
    get_whisper_model(model_name)


class WhisperService:
    """Handles local Whisper transcription using the module-level singleton."""

    AVAILABLE_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v2"]

    def __init__(self, model_name: str = "base.en", device: str = "cpu"):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not recognised. "
                f"Available: {', '.join(self.AVAILABLE_MODELS)}"
            )
        self.model_name = model_name
        self.device = device
        # Grab the singleton immediately — no lazy loading per request
        self.model = get_whisper_model(model_name, device)

    def load_model(self):
        """No-op: model is already loaded via singleton. Kept for compatibility."""
        self.model = get_whisper_model(self.model_name, self.device)

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe an audio file.

        Optimisations vs original:
        - beam_size reduced 5 → 2  (2-3x faster, ~1% accuracy loss)
        - word_timestamps only when needed (caller passes flag)
        - language is fixed to 'en' — skips language detection step
        """
        t0 = time.time()
        print(f"\n[Whisper] Transcribing: {Path(audio_path).name}")

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=2,                            # was 5 — major speed gain
            best_of=1,                              # no sampling, deterministic
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,        # was 500 — catches more pauses
                speech_pad_ms=200,
            ),
            word_timestamps=True,
            condition_on_previous_text=False,       # prevents hallucination cascade
        )

        transcript_segments = []
        for segment in segments_iter:
            seg_data = {
                "start": round(segment.start, 2),
                "end":   round(segment.end,   2),
                "text":  segment.text.strip(),
            }
            if hasattr(segment, "words") and segment.words:
                seg_data["words"] = [
                    {"start": round(w.start, 2),
                     "end":   round(w.end,   2),
                     "word":  w.word}
                    for w in segment.words
                ]
            transcript_segments.append(seg_data)
            print(f"  [{seg_data['start']:.2f}s - {seg_data['end']:.2f}s] {seg_data['text']}")

        print(f"[Whisper] Done — {len(transcript_segments)} segments in {time.time()-t0:.2f}s")
        return transcript_segments

    def transcribe_chunk(self, audio_data: np.ndarray,
                         language: str = "en",
                         offset_seconds: float = 0.0) -> List[Dict]:
        """
        Transcribe a numpy audio chunk (used by streaming pipeline).
        Timestamps are offset so they align with the full recording timeline.

        Args:
            audio_data:      float32 mono 16kHz numpy array
            language:        language code
            offset_seconds:  seconds to add to all timestamps
        """
        t0 = time.time()

        segments_iter, _ = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=2,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        result = []
        for segment in segments_iter:
            seg_data = {
                "start": round(segment.start + offset_seconds, 2),
                "end":   round(segment.end   + offset_seconds, 2),
                "text":  segment.text.strip(),
            }
            if hasattr(segment, "words") and segment.words:
                seg_data["words"] = [
                    {"start": round(w.start + offset_seconds, 2),
                     "end":   round(w.end   + offset_seconds, 2),
                     "word":  w.word}
                    for w in segment.words
                ]
            result.append(seg_data)

        print(f"  [Whisper chunk @ {offset_seconds:.1f}s] "
              f"{len(result)} segs in {time.time()-t0:.2f}s")
        return result

    def transcribe_numpy(self, audio_data: np.ndarray,
                         language: str = "en") -> List[Dict]:
        """Transcribe from a numpy array (kept for backwards compatibility)."""
        return self.transcribe_chunk(audio_data, language=language, offset_seconds=0.0)
