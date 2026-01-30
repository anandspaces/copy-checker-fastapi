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
    # REQUIRED: PDF file
    file: UploadFile = File(..., description="PDF answer sheet (REQUIRED)"),
    
    # OPTIONAL: Subject and marks (will be auto-extracted if not provided)
    subject: Optional[str] = Form(None, description="Subject name (auto-extracted if not provided)"),
    total_marks: Optional[float] = Form(None, gt=0, description="Total marks (auto-extracted if not provided)"),
    
    # OPTIONAL: Marking scheme
    marking_scheme: Optional[str] = Form(None, description="Marking scheme or rubric (generic if not provided)"),
    
    # OPTIONAL: Evaluation settings
    strict_marking: bool = Form(default=False, description="Apply strict grading"),
    include_partial_credit: bool = Form(default=True, description="Award partial marks"),
    
    # OPTIONAL: Annotation settings
    annotation_font_size: int = Form(default=10, ge=6, le=14, description="Font size for annotations"),
    show_remarks: bool = Form(default=True, description="Show remarks on pages"),
    show_marks: bool = Form(default=True, description="Show marks on pages"),
    show_summary: bool = Form(default=True, description="Show summary on last page")
) -> FileResponse:
    """
    **AI-Powered Answer Sheet Evaluator - Human-like Copy Checking**
    
    Upload a PDF answer sheet and get it evaluated question-by-question with marks and feedback.
    
    ## What This API Does:
    
    1. **📄 Extracts Metadata** from first page (subject, total marks, student info)
    2. **🔍 Identifies Questions** and student answers from all pages
    3. **🤖 Evaluates Each Question** using AI with detailed feedback
    4. **✍️ Annotates PDF** with marks and remarks for each question
    
    ## Pipeline:
    
    **Phase 1: OCR Extraction**
    - Extracts metadata from first page (subject, total marks, etc.)
    - Identifies all questions across pages
    - Extracts student's answers for each question
    - Detects marks allocated per question
    
    **Phase 2: Question-wise Evaluation**
    - Evaluates each question individually using Gemini AI
    - Awards marks based on correctness, completeness, clarity
    - Provides strengths and improvement suggestions
    - Applies partial credit if enabled
    
    **Phase 3: PDF Annotation**
    - Adds marks next to each question on the PDF
    - Color-coded feedback (green=correct, orange=partial, red=incorrect)
    - Adds detailed remarks for each question
    - Summary page with overall grade
    
    ## Required:
    - **file**: PDF answer sheet (max 50MB)
    
    ## Optional (Auto-extracted if not provided):
    - **subject**: Subject name (e.g., "Physics", "Mathematics")
    - **total_marks**: Total marks for the paper
    - **marking_scheme**: Detailed marking criteria
    
    ## Returns:
    - Annotated PDF with question-wise marks and feedback
    - Metadata headers with total score, grade, and processing info
    
    ## Example Usage:
    
    ```bash
    curl -X POST "http://localhost:8000/evaluate" \\
      -F "file=@answer_sheet.pdf" \\
      -F "subject=Physics" \\
      -F "marking_scheme=Detailed explanations: 5 marks, Correct formulas: 3 marks" \\
      -o evaluated.pdf
    ```
    
    Or simply upload just the PDF - subject and marks will be auto-extracted!
    
    ```bash
    curl -X POST "http://localhost:8000/evaluate" \\
      -F "file=@answer_sheet.pdf" \\
      -o evaluated.pdf
    ```
    """
    logger.info("=" * 80)
    logger.info("NEW QUESTION-WISE EVALUATION REQUEST")
    logger.info(f"File: {file.filename}")
    logger.info(f"Subject: {subject or 'AUTO-EXTRACT'}")
    logger.info(f"Total Marks: {total_marks or 'AUTO-EXTRACT'}")
    logger.info(f"Marking Scheme: {'Provided' if marking_scheme else 'Generic'}")
    logger.info(f"Strict marking: {strict_marking}, Partial credit: {include_partial_credit}")
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
            total_marks=total_marks,
            marking_scheme=marking_scheme,
            llm_provider=LLMProvider.GEMINI,
            strict_marking=strict_marking,
            include_partial_credit=include_partial_credit,
            auto_extract_metadata=True
        )
        
        # Create annotation config
        annotation_config = AnnotationConfig(
            font_size=annotation_font_size,
            show_remarks=show_remarks,
            show_marks=show_marks,
            show_summary=show_summary
        )
        
        # Get orchestrator
        logger.info("Initializing question-wise evaluation pipeline...")
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
        logger.info(f"Questions evaluated: {summary.total_questions}")
        logger.info(f"Final score: {summary.total_marks_awarded}/{summary.total_max_marks} "
                   f"({summary.percentage}%) - Grade: {summary.grade}")
        
        # Prepare response with metadata headers
        response = FileResponse(
            path=str(annotated_path),
            media_type="application/pdf",
            filename=f"evaluated_{file.filename}",
            headers={
                "X-Total-Questions": str(summary.total_questions),
                "X-Total-Marks": str(summary.total_marks_awarded),
                "X-Max-Marks": str(summary.total_max_marks),
                "X-Percentage": str(summary.percentage),
                "X-Grade": summary.grade,
                "X-Subject": ocr_result.metadata.subject or "Unknown",
                "X-Processing-Time-Seconds": str(summary.processing_time_seconds),
                "X-OCR-Provider": ocr_result.ocr_provider,
                "X-OCR-Time-Ms": str(ocr_result.processing_time_ms),
                "X-Metadata-Extracted": "true" if ocr_result.metadata.subject else "false"
            }
        )
        
        logger.info(f"Returning annotated PDF: evaluated_{file.filename}")
        logger.info("=" * 80)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )
    finally:
        # Cleanup happens automatically via TempFileManager context
        pass


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Answer Sheet Evaluator",
        "version": "3.0.0 - Question-wise",
        "mode": "Human-like Copy Checker"
    }


@router.get("/info")
async def service_info():
    """Service information and capabilities"""
    return {
        "service": "AI Answer Sheet Evaluator",
        "version": "3.0.0",
        "mode": "Question-wise Evaluation (Human-like Copy Checker)",
        "description": "Automated evaluation of subjective answer sheets with question-wise marking",
        
        "architecture": {
            "phase_1": "OCR: Extract metadata + questions from all pages",
            "phase_2": "Evaluation: Grade each question individually with AI",
            "phase_3": "Annotation: Add question-wise marks and remarks to PDF"
        },
        
        "features": [
            "✅ Auto-extracts subject and total marks from first page",
            "✅ Identifies all questions across pages",
            "✅ Question-wise evaluation with detailed feedback",
            "✅ Color-coded annotations (green/orange/red)",
            "✅ Partial credit support",
            "✅ Strengths and improvements for each question",
            "✅ Letter grade assignment",
            "✅ Summary page with overall performance"
        ],
        
        "how_it_works": {
            "step_1": "Upload PDF - subject/marks are auto-extracted from first page",
            "step_2": "AI identifies all questions and student answers",
            "step_3": "Each question is evaluated individually",
            "step_4": "PDF is annotated with marks and feedback per question",
            "step_5": "Download evaluated PDF with color-coded results"
        },
        
        "api_usage": {
            "required_params": ["file (PDF)"],
            "optional_params": [
                "subject (auto-extracted if not provided)",
                "total_marks (auto-extracted if not provided)",
                "marking_scheme (generic if not provided)",
                "strict_marking (default: false)",
                "include_partial_credit (default: true)"
            ]
        },
        
        "supported_providers": {
            "ocr": ["Gemini Vision"],
            "evaluation": ["Gemini LLM"]
        },
        
        "requirements": [
            "GEMINI_API_KEY environment variable",
            "PDF files under 50MB"
        ],
        
        "example_curl": """
curl -X POST "http://localhost:8000/evaluate" \\
  -F "file=@answer_sheet.pdf" \\
  -F "subject=Physics" \\
  -F "marking_scheme=Clear explanation: 5 marks, Formula: 3 marks" \\
  -o evaluated.pdf
        """,
        
        "endpoints": {
            "/evaluate": "POST - Main evaluation endpoint (question-wise)",
            "/health": "GET - Health check",
            "/info": "GET - Service information",
            "/docs": "GET - Interactive API documentation"
        }
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Answer Sheet Evaluator API - Question-wise Mode",
        "version": "3.0.0",
        "mode": "Human-like Copy Checker",
        "tagline": "Upload PDF → Auto-extract → Question-wise Evaluation → Annotated Results",
        "documentation": "/docs",
        "health_check": "/health",
        "service_info": "/info",
        "quick_start": {
            "1": "Upload your answer sheet PDF (required)",
            "2": "Optionally provide subject, marks, marking scheme",
            "3": "API auto-extracts metadata from first page if not provided",
            "4": "Get back annotated PDF with question-wise marks and feedback"
        }
    }