# src/routes.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json
import logging
from typing import Optional

from src.schemas import (
    EvaluationRequest,
    EvaluationSummary,
    AnnotationConfig,
    EvaluationRequest,
    AnnotationConfig
)
from src.services.llm_service import LLMService, LLMProvider
from src.services.evaluation_service import GeminiEvaluationService
from src.file_utils import TempFileManager, validate_pdf, get_file_size_mb

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/evaluate-pdf-ocr", response_class=FileResponse)
async def evaluate_pdf_gemini_only(
    file: UploadFile = File(..., description="PDF answer sheet to evaluate"),
    subject: Optional[str] = Form(default="General", description="Subject name"),
    marking_scheme: Optional[str] = Form(default=None, description="Question paper and marking scheme"),
    max_marks_per_page: Optional[float] = Form(default=10.0, description="Maximum marks per page"),
    annotation_font_size: int = Form(default=10, description="Font size for annotations")
) -> FileResponse:
    """
    Evaluate PDF using Gemini for BOTH OCR and evaluation (no Tesseract, no OpenCV preprocessing)
    
    This endpoint uses Google Gemini's vision capabilities for text extraction and evaluation.
    It's simpler and faster than the standard pipeline but requires a Gemini API key.
    
    ## Advantages over standard pipeline:
    - No dependency on Tesseract OCR
    - No complex image preprocessing
    - Better handling of handwritten text
    - Faster processing
    - Single API for both OCR and evaluation
    
    ## Requirements:
    - GEMINI_API_KEY environment variable must be set
    
    Args:
        file: PDF file upload (REQUIRED)
        subject: Subject name for evaluation (optional, default: "General")
        marking_scheme: Text or JSON containing questions and marking scheme (optional)
        max_marks_per_page: Maximum marks per page (optional, default: 10.0)
        annotation_font_size: Font size for PDF annotations (optional, default: 10)
        
    Returns:
        Annotated PDF file with marks and remarks
    """
    logger.info("=" * 80)
    logger.info("NEW GEMINI ONLY EVALUATION REQUEST")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Max marks per page: {max_marks_per_page}")
    logger.info("Using Gemini for both OCR and evaluation")
    logger.info("=" * 80)
    
    temp_manager = TempFileManager()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        logger.info("File type validation passed")
        
        # Save uploaded file
        logger.info("Saving uploaded file...")
        input_pdf_path = await temp_manager.save_upload(file, suffix=".pdf")
        logger.info(f"File saved to: {input_pdf_path}")
        
        # Validate PDF
        logger.info("Validating PDF structure...")
        if not validate_pdf(input_pdf_path):
            logger.error("PDF validation failed")
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )
        
        logger.info("PDF validation passed")
        
        # Check file size (limit to 50MB)
        file_size = get_file_size_mb(input_pdf_path)
        logger.info(f"File size: {file_size:.2f} MB")
        
        if file_size > 50:
            logger.error(f"File too large: {file_size:.2f} MB")
            raise HTTPException(
                status_code=400,
                detail=f"PDF file too large ({file_size:.1f}MB). Maximum size is 50MB"
            )
        
        # Use default marking scheme if not provided
        if marking_scheme is None:
            marking_scheme = f"Evaluate the student's answers for {subject}. Award marks based on correctness, completeness, and clarity."
            logger.info("Using default marking scheme")
        else:
            logger.info(f"Using custom marking scheme (length: {len(marking_scheme)} chars)")
        
        # Create evaluation request
        logger.info("Creating evaluation request for Gemini")
        eval_request = EvaluationRequest(
            subject=subject,
            marking_scheme=marking_scheme,
            max_marks_per_page=max_marks_per_page
        )
        
        # Create annotation config
        annotation_config = AnnotationConfig(
            font_size=annotation_font_size
        )
        logger.info(f"Annotation config: font_size={annotation_font_size}")
        
        # Initialize Gemini only services
        logger.info("Initializing Gemini LLM service")
        llm_service = LLMService(provider=LLMProvider.GEMINI)
        
        logger.info("Initializing GeminiEvaluationService")
        evaluation_service = GeminiEvaluationService(
            llm_service=llm_service,
            annotation_config=annotation_config
        )
        
        # Create output path
        output_pdf_path = temp_manager.create_temp_path(suffix="_gemini_annotated.pdf")
        logger.info(f"Output path: {output_pdf_path}")
        
        # Evaluate PDF using Gemini
        logger.info("Starting Gemini PDF evaluation...")
        annotated_path, summary = evaluation_service.evaluate_pdf(
            input_pdf_path,
            output_pdf_path,
            eval_request
        )
        
        logger.info("Gemini PDF evaluation completed successfully")
        logger.info(f"Results: {summary.total_marks_awarded}/{summary.total_max_marks} ({summary.percentage}%)")
        logger.info(f"Processing time: {summary.processing_time_seconds}s")
        
        # Prepare response
        response = FileResponse(
            path=str(annotated_path),
            media_type="application/pdf",
            filename=f"gemini_evaluated_{file.filename}",
            headers={
                "X-Total-Marks": str(summary.total_marks_awarded),
                "X-Max-Marks": str(summary.total_max_marks),
                "X-Percentage": str(summary.percentage),
                "X-Processing-Time": str(summary.processing_time_seconds),
                "X-OCR-Method": "Gemini Vision",
                "X-Evaluation-Method": "Gemini"
            }
        )
        
        logger.info(f"Returning Gemini annotated PDF: gemini_evaluated_{file.filename}")
        logger.info("=" * 80)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        temp_manager.cleanup_all()
        logger.error("HTTP exception occurred")
        raise
        
    except Exception as e:
        # Clean up and raise internal error
        temp_manager.cleanup_all()
        logger.error(f"Internal error during Gemini evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Gemini evaluation failed: {str(e)}"
        )


@router.get("/info")
async def gemini_info():
    """Information about the Gemini only evaluation endpoint"""
    logger.info("Info endpoint accessed")
    return {
        "endpoint": "/gemini/evaluate-pdf",
        "method": "POST",
        "description": "Evaluate PDF using Gemini for both OCR and evaluation",
        "ocr_method": "Gemini Vision API",
        "evaluation_method": "Gemini LLM",
        "advantages": [
            "No Tesseract dependency",
            "No OpenCV preprocessing",
            "Better handwriting recognition",
            "Faster processing",
            "Single API integration"
        ],
        "requirements": [
            "GEMINI_API_KEY environment variable"
        ],
        "model": {
            "ocr": 'gemini-3-flash-preview',
            "evaluation": 'gemini-3-flash-preview'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check endpoint accessed")
    return {
        "status": "healthy",
        "service": "Answer Sheet Evaluator",
        "version": "1.0.0"
    }


@router.get("/")
async def root():
    """Root endpoint with API information"""
    logger.info("Root endpoint accessed")
    return {
        "service": "Answer Sheet Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "/evaluate-pdf": "POST - Evaluate PDF answer sheet",
            "/evaluate-pdf-ocr": "POST - Evaluate PDF using Gemini OCR",
            "/health": "GET - Health check",
            "/info": "GET - Gemini endpoint information",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - API documentation (ReDoc)"
        },
        "llm_provider": "Google Gemini",
        "note": "Set GEMINI_API_KEY environment variable before use"
    }