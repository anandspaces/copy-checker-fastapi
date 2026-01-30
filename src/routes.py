# src/routes.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging
from typing import Optional

from src.schemas import EvaluationRequest, AnnotationConfig, LLMProvider
from src.services.service_factory import ServiceFactory
from src.file_utils import TempFileManager, validate_pdf, get_file_size_mb

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/evaluate", response_class=FileResponse)
async def evaluate_answer_sheet(
    file: UploadFile = File(..., description="PDF answer sheet"),
    subject: str = Form(..., description="Subject name"),
    marking_scheme: str = Form(..., description="Marking scheme or rubric"),
    max_marks_per_page: float = Form(..., gt=0, description="Maximum marks per page"),
    strict_marking: bool = Form(default=False, description="Apply strict grading"),
    include_partial_credit: bool = Form(default=True, description="Award partial marks"),
    annotation_font_size: int = Form(default=10, ge=6, le=14, description="Font size for annotations"),
    show_remarks: bool = Form(default=True, description="Show remarks on pages"),
    show_marks: bool = Form(default=True, description="Show marks on pages"),
    show_summary: bool = Form(default=True, description="Show summary on last page")
) -> FileResponse:
    """
    Evaluate answer sheet PDF using AI
    
    **Pipeline:**
    1. OCR: Extract text from all pages using Gemini Vision
    2. Evaluation: Evaluate each page using Gemini LLM
    3. Annotation: Add marks and feedback to PDF
    
    **Requirements:**
    - GEMINI_API_KEY environment variable must be set
    - PDF file size must be under 50MB
    - Valid PDF file format
    
    Args:
        file: PDF answer sheet to evaluate
        subject: Subject name (e.g., "Mathematics", "Physics")
        marking_scheme: Detailed marking criteria and rubric
        max_marks_per_page: Maximum marks allocated per page
        strict_marking: If true, apply stricter grading standards
        include_partial_credit: If true, award partial marks for partially correct answers
        annotation_font_size: Font size for PDF annotations (6-14)
        show_remarks: Show detailed feedback on pages
        show_marks: Show marks on pages
        show_summary: Show overall summary on last page
        
    Returns:
        Annotated PDF file with marks and feedback
        
    Raises:
        400: Invalid file format or size
        500: Processing error
    """
    logger.info("=" * 80)
    logger.info("NEW EVALUATION REQUEST")
    logger.info(f"File: {file.filename}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Max marks per page: {max_marks_per_page}")
    logger.info(f"Strict marking: {strict_marking}")
    logger.info(f"Partial credit: {include_partial_credit}")
    logger.info("=" * 80)
    
    temp_manager = TempFileManager()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file
        logger.info("Saving uploaded file...")
        input_pdf_path = await temp_manager.save_upload(file, suffix=".pdf")
        
        # Validate PDF
        if not validate_pdf(input_pdf_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted PDF file"
            )
        
        # Check file size (50MB limit)
        file_size = get_file_size_mb(input_pdf_path)
        logger.info(f"File size: {file_size:.2f} MB")
        
        if file_size > 50:
            raise HTTPException(
                status_code=400,
                detail=f"PDF file too large ({file_size:.1f}MB). Maximum size is 50MB"
            )
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            subject=subject,
            marking_scheme=marking_scheme,
            max_marks_per_page=max_marks_per_page,
            llm_provider=LLMProvider.GEMINI,
            strict_marking=strict_marking,
            include_partial_credit=include_partial_credit
        )
        
        # Create annotation config
        annotation_config = AnnotationConfig(
            font_size=annotation_font_size,
            show_remarks=show_remarks,
            show_marks=show_marks,
            show_summary=show_summary
        )
        
        # Get orchestrator
        logger.info("Initializing evaluation pipeline...")
        orchestrator = ServiceFactory.get_orchestrator(
            ocr_provider="gemini",
            llm_provider=LLMProvider.GEMINI,
            annotation_config=annotation_config
        )
        
        # Create output path
        output_pdf_path = temp_manager.create_temp_path(suffix="_evaluated.pdf")
        
        # Run complete pipeline
        logger.info("Starting evaluation pipeline...")
        annotated_path, summary, ocr_result = orchestrator.evaluate_answer_sheet(
            input_pdf_path,
            output_pdf_path,
            eval_request
        )
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Final score: {summary.total_marks_awarded}/{summary.total_max_marks} "
                   f"({summary.percentage}%) - Grade: {summary.grade}")
        
        # Prepare response with metadata headers
        response = FileResponse(
            path=str(annotated_path),
            media_type="application/pdf",
            filename=f"evaluated_{file.filename}",
            headers={
                "X-Total-Marks": str(summary.total_marks_awarded),
                "X-Max-Marks": str(summary.total_max_marks),
                "X-Percentage": str(summary.percentage),
                "X-Grade": summary.grade,
                "X-Total-Pages": str(summary.total_pages),
                "X-Processing-Time-Seconds": str(summary.processing_time_seconds),
                "X-OCR-Provider": ocr_result.ocr_provider,
                "X-OCR-Time-Ms": str(ocr_result.total_processing_time_ms)
            }
        )
        
        logger.info(f"Returning annotated PDF: evaluated_{file.filename}")
        logger.info("=" * 80)
        
        return response
    except Exception as e:
        temp_manager.cleanup_all()
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Answer Sheet Evaluator",
        "version": "2.0.0",
        "architecture": "OCR → Evaluation → Annotation"
    }


@router.get("/info")
async def service_info():
    """Service information and capabilities"""
    return {
        "service": "AI Answer Sheet Evaluator",
        "version": "2.0.0",
        "description": "Automated evaluation of subjective answer sheets using AI",
        "architecture": {
            "phase_1": "OCR Text Extraction (Gemini Vision)",
            "phase_2": "Page-by-Page Evaluation (Gemini LLM)",
            "phase_3": "PDF Annotation (PyMuPDF)"
        },
        "features": [
            "High-accuracy handwriting recognition",
            "Intelligent marking with partial credit",
            "Detailed feedback and remarks",
            "Automated grading with letter grades",
            "PDF annotation with marks and comments"
        ],
        "supported_providers": {
            "ocr": ["gemini"],
            "evaluation": ["gemini"]
        },
        "requirements": [
            "GEMINI_API_KEY environment variable",
            "PDF files under 50MB"
        ],
        "endpoints": {
            "/evaluate": "POST - Main evaluation endpoint",
            "/health": "GET - Health check",
            "/info": "GET - Service information",
            "/docs": "GET - Interactive API documentation"
        }
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Answer Sheet Evaluator API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "service_info": "/info"
    }