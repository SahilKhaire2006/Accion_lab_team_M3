"""
Main FastAPI application for Privacy-Preserving Clinical Scribe.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Clinical Scribe Transcription Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
