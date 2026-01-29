from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.routes import router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting Answer Sheet Evaluation API")
    logger.info("Initializing services...")
    
    # TODO: Add any startup tasks (e.g., checking API keys, warming up models)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Answer Sheet Evaluation API")
    # TODO: Add any cleanup tasks


# Create FastAPI application
app = FastAPI(
    title="Answer Sheet Evaluation API",
    description="""
    Automated subjective answer sheet evaluation system using OCR, Computer Vision, and LLM.
    
    ## Features
    
    - **PDF Processing**: Accepts PDF answer sheets
    - **OCR**: Extracts handwritten/printed text using Tesseract
    - **Computer Vision**: Analyzes page metadata (writing density, diagrams, crossed-out regions)
    - **AI Evaluation**: Uses LLM (Gemini/OpenAI) for subject-aware evaluation
    - **PDF Annotation**: Overlays marks and remarks on original PDF
    
    ## Workflow
    
    1. Upload PDF answer sheet
    2. Provide subject and marking scheme
    3. System processes each page:
       - Converts to image
       - Applies OCR
       - Extracts vision metadata
       - Evaluates with LLM
    4. Returns annotated PDF with marks and remarks
    
    ## API Keys Required
    
    - **Gemini**: Set `GEMINI_API_KEY` environment variable
    - **OpenAI**: Set `OPENAI_API_KEY` environment variable
    
    ## Limitations
    
    - Maximum PDF size: 50MB
    - Supports English language by default
    - OCR accuracy depends on handwriting quality
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
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


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "detail": "An internal error occurred. Please try again later.",
        "error_type": type(exc).__name__
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )