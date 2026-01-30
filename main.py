# main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

from src.routes import router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('evaluation_service.log')
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    
    # Startup
    logger.info("=" * 80)
    logger.info("STARTING AI ANSWER SHEET EVALUATOR")
    logger.info("Version: 2.0.0")
    logger.info("=" * 80)
    
    # Verify environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        logger.error("Please set GEMINI_API_KEY before starting the service")
    else:
        logger.info("✓ GEMINI_API_KEY configured")
    
    logger.info("Service architecture:")
    logger.info("  Phase 1: OCR Text Extraction (Gemini Vision)")
    logger.info("  Phase 2: Page-by-Page Evaluation (Gemini LLM)")
    logger.info("  Phase 3: PDF Annotation (PyMuPDF)")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  POST /evaluate - Main evaluation endpoint")
    logger.info("  GET  /health   - Health check")
    logger.info("  GET  /info     - Service information")
    logger.info("  GET  /docs     - API documentation")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("SHUTTING DOWN AI ANSWER SHEET EVALUATOR")
    logger.info("=" * 80)


# Create FastAPI application
app = FastAPI(
    title="AI Answer Sheet Evaluator",
    description="""
    Automated evaluation system for subjective answer sheets using AI.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "AI Answer Sheet Evaluator",
        "url": "https://github.com/your-repo"
    },
    license_info={
        "name": "MIT"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, tags=["Evaluation"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "error_type": type(exc).__name__,
            "path": str(request.url)
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )