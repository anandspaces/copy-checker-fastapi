from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json
from typing import Optional

from src.schemas import (
    EvaluationRequest,
    EvaluationSummary,
    LLMProvider,
    AnnotationConfig
)
from src.services.llm_service import LLMService
from src.services.evaluation_service import EvaluationService
from src.utils.file_utils import TempFileManager, validate_pdf, get_file_size_mb


router = APIRouter()


@router.post("/evaluate-pdf", response_class=FileResponse)
async def evaluate_pdf(
    file: UploadFile = File(..., description="PDF answer sheet to evaluate"),
    subject: str = Form(..., description="Subject name"),
    marking_scheme: str = Form(..., description="Question paper and marking scheme"),
    llm_provider: str = Form(default="gemini", description="LLM provider: gemini or openai"),
    max_marks_per_page: Optional[float] = Form(default=None, description="Maximum marks per page"),
    annotation_font_size: int = Form(default=10, description="Font size for annotations")
) -> FileResponse:
    """
    Evaluate a PDF answer sheet and return annotated PDF
    
    Args:
        file: PDF file upload
        subject: Subject name for evaluation
        marking_scheme: Text or JSON containing questions and marking scheme
        llm_provider: LLM provider to use (gemini or openai)
        max_marks_per_page: Optional override for max marks per page
        annotation_font_size: Font size for PDF annotations
        
    Returns:
        Annotated PDF file with marks and remarks
    """
    temp_manager = TempFileManager()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file
        input_pdf_path = await temp_manager.save_upload(file, suffix=".pdf")
        
        # Validate PDF
        if not validate_pdf(input_pdf_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )
        
        # Check file size (limit to 50MB)
        file_size = get_file_size_mb(input_pdf_path)
        if file_size > 50:
            raise HTTPException(
                status_code=400,
                detail=f"PDF file too large ({file_size:.1f}MB). Maximum size is 50MB"
            )
        
        # Parse LLM provider
        try:
            provider = LLMProvider(llm_provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid LLM provider. Must be 'gemini' or 'openai'"
            )
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            subject=subject,
            marking_scheme=marking_scheme,
            llm_provider=provider,
            max_marks_per_page=max_marks_per_page
        )
        
        # Create annotation config
        annotation_config = AnnotationConfig(
            font_size=annotation_font_size
        )
        
        # Initialize services
        # TODO: Add API key management (environment variables or secure storage)
        llm_service = LLMService(provider=provider)
        evaluation_service = EvaluationService(
            llm_service=llm_service,
            annotation_config=annotation_config
        )
        
        # Create output path
        output_pdf_path = temp_manager.create_temp_path(suffix="_annotated.pdf")
        
        # Evaluate PDF
        annotated_path, summary = evaluation_service.evaluate_pdf(
            input_pdf_path,
            output_pdf_path,
            eval_request
        )
        
        # Prepare response
        response = FileResponse(
            path=str(annotated_path),
            media_type="application/pdf",
            filename=f"evaluated_{file.filename}",
            headers={
                "X-Total-Marks": str(summary.total_marks_awarded),
                "X-Max-Marks": str(summary.total_max_marks),
                "X-Percentage": str(summary.percentage),
                "X-Processing-Time": str(summary.processing_time_seconds)
            }
        )
        
        # Note: Cleanup happens when response is sent
        # Temporary files will be cleaned up by the OS eventually
        # For production, implement proper cleanup with background tasks
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        temp_manager.cleanup_all()
        raise
        
    except Exception as e:
        # Clean up and raise internal error
        temp_manager.cleanup_all()
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Answer Sheet Evaluator",
        "version": "1.0.0"
    }


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Answer Sheet Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "/evaluate-pdf": "POST - Evaluate PDF answer sheet",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - API documentation (ReDoc)"
        },
        "supported_llm_providers": ["gemini", "openai"]
    }