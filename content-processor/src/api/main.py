"""FastAPI main application - Knowledge Graph Builder API.

This API provides endpoints for building knowledge graphs from PDF documents.

Features:
- PDF upload and processing
- Concept extraction using LLM
- Knowledge graph construction
- ChromaDB storage for semantic search
- DAG optimization and transitive reduction
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import HealthResponse
from api.dependencies import verify_components
from api.routes import router
from api.config import api_settings
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, status

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Startup:
    - Verify all components (LLM, embeddings, ChromaDB)
    - Create necessary directories

    Shutdown:
    - Cleanup resources
    """
    # Startup
    print("=" * 60)
    print("🚀 Starting Knowledge Graph Builder API")
    print("=" * 60)

    # Create directories
    Path(api_settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(api_settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # Verify components
    print("\n📦 Initializing components...")
    components = verify_components()

    for name, status_msg in components.items():
        symbol = "✅" if "OK" in status_msg else "❌"
        print(f"{symbol} {name}: {status_msg}")

    print("\n" + "=" * 60)
    print(f"✨ API ready at http://{api_settings.host}:{api_settings.port}")
    print(f"📚 Docs at http://{api_settings.host}:{api_settings.port}/docs")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\n🛑 Shutting down Knowledge Graph Builder API")


# Create FastAPI app
app = FastAPI(
    title=api_settings.api_title,
    version=api_settings.api_version,
    description=api_settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check",
    description="Check if the API and all components are healthy"
)
async def health_check():
    """Health check endpoint."""
    components = verify_components()

    # Check if all components are OK
    all_healthy = all("OK" in status_msg for status_msg in components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=api_settings.api_version,
        components=components
    )


# Root endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="API root",
    description="Get API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": api_settings.api_title,
        "version": api_settings.api_version,
        "description": api_settings.api_description,
        "docs_url": "/docs",
        "health_url": "/health"
    }


# Include API routes
app.include_router(router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=api_settings.reload
    )
