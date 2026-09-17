"""
Main FastAPI application for Privacy-Preserving Clinical Scribe.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.api.routes import router
from app.api.live_routes import router as live_router
from app.services.llm_correction import validate_groq_config
from app.transcription.whisper_service import warmup_whisper
from app.services.diarization import warmup_diarization

# Create FastAPI app
app = FastAPI(
    title="Clinical Scribe Transcription Service",
    description="Privacy-preserving on-device audio transcription for clinical encounters",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["transcription"])
app.include_router(live_router, prefix="/api/v1/live", tags=["live-recording"])

# Serve static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """
    Pre-warm all models at startup so the first request is fast.
    Both models load in background threads — server is ready immediately
    but models will be hot by the time the first recording starts.
    """
    import threading

    groq_warning = validate_groq_config()
    if groq_warning:
        print(groq_warning)

    # Pre-load Whisper in background (takes 3-5s, non-blocking)
    def _warm_whisper():
        print("[Startup] Pre-loading Whisper model...")
        warmup_whisper("base.en")
        print("[Startup] ✓ Whisper model ready")

    # Pre-load pyannote in background (takes 5-15s, non-blocking)
    def _warm_diarization():
        print("[Startup] Pre-loading Diarization pipeline...")
        warmup_diarization()
        print("[Startup] ✓ Diarization pipeline ready")

    threading.Thread(target=_warm_whisper,     daemon=True).start()
    threading.Thread(target=_warm_diarization, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the live recording dashboard."""
    html_file = Path("static/index.html")
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    else:
        return """
        <html>
            <body>
                <h1>Clinical Scribe Dashboard</h1>
                <p>Dashboard not found. Please ensure static/index.html exists.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """


@app.get("/transcript", response_class=HTMLResponse)
async def transcript_viewer():
    """Serve the transcript viewer page."""
    html_file = Path("static/transcript.html")
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    else:
        return """
        <html>
            <body>
                <h1>Transcript Viewer</h1>
                <p>Transcript viewer not found.</p>
                <p><a href="/">Return to Dashboard</a></p>
            </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
