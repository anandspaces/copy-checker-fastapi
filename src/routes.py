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
    show_remarks: bool = Form(default=True, description="Show remarks on pages"),
    show_marks: bool = Form(default=True, description="Show marks on pages"),
    show_summary: bool = Form(default=True, description="Show summary on last page")
) -> FileResponse:
    """
    **🎓 AI-Powered Human-Like Answer Sheet Checker**
    
    Upload a PDF answer sheet and get it checked like a real teacher would - 
    with handwritten-style marks and remarks in red pen!
    
    ## ✨ What Makes This Human-Like:
    
    1. **🔍 Smart Coordinate Detection** - Uses Gemini Vision to find blank spaces
    2. **✍️ Handwriting Style** - Red pen annotations that look natural
    3. **📍 Intelligent Placement** - Marks appear near answers, not in boxes
    4. **🎯 Context-Aware** - Understands page layout and multi-page questions
    
    ## 🔄 How It Works:
    
    **Phase 1: OCR Extraction**
    - Scans first page for subject, total marks, student info
    - Identifies all questions across all pages
    - Extracts student's handwritten answers
    - Detects marks allocated per question
    
    **Phase 2: AI Evaluation**
    - Gemini AI evaluates each question individually
    - Awards marks based on correctness, completeness, clarity
    - Provides constructive feedback and improvement suggestions
    - Applies partial credit intelligently
    
    **Phase 3: Human-Like Annotation** ⭐ NEW!
    - Gemini Vision finds blank spaces near answers
    - Writes marks in red pen (e.g., "Q1: 8/10 (80%)")
    - Adds remarks naturally below marks
    - Uses checkmarks (✓) and crosses (✗)
    - Summary page with overall grade
    
    ## 📋 Required:
    - **file**: PDF answer sheet (max 50MB)
    
    ## 📋 Optional (Auto-extracted if not provided):
    - **subject**: Subject name (e.g., "Physics", "Mathematics")
    - **total_marks**: Total marks for the paper
    - **marking_scheme**: Detailed marking criteria
    
    ## 📤 Returns:
    - Annotated PDF with human-like handwritten marks and feedback
    - Metadata headers with scores, grade, and processing info
    
    ## 💡 Example Usage:
    
    **Simple (auto-extract everything):**
    ```bash
    curl -X POST "http://localhost:8000/evaluate" \\
      -F "file=@answer_sheet.pdf" \\
      -o checked_copy.pdf
    ```
    
    **With custom marking scheme:**
    ```bash
    curl -X POST "http://localhost:8000/evaluate" \\
      -F "file=@answer_sheet.pdf" \\
      -F "subject=Physics" \\
      -F "marking_scheme=Explanation: 5 marks, Formula: 3 marks, Calculation: 2 marks" \\
      -o checked_copy.pdf
    ```
    
    ## 🎨 Key Features:
    - ✅ Red pen annotations (like a real teacher)
    - ✅ Marks written near answers (not in margins)
    - ✅ Natural handwriting-style fonts
    - ✅ Checkmarks for correct answers
    - ✅ Crosses for incorrect answers
    - ✅ Multi-line remarks that wrap naturally
    - ✅ Oval/underline emphasis (teacher style)
    - ✅ Smart blank space detection
    - ✅ Handles multi-page questions
    - ✅ Professional summary page
    """
    logger.info("=" * 80)
    logger.info("NEW HUMAN-LIKE COPY CHECKING REQUEST")
    logger.info(f"File: {file.filename}")
    logger.info(f"Subject: {subject or 'AUTO-EXTRACT'}")
    logger.info(f"Total Marks: {total_marks or 'AUTO-EXTRACT'}")
    logger.info(f"Marking Scheme: {'Provided' if marking_scheme else 'Generic'}")
    logger.info(f"Strict marking: {strict_marking}, Partial credit: {include_partial_credit}")
    logger.info("Annotation Mode: HUMAN-LIKE with Gemini Vision coordinate detection")
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
            font_size=10,  # Used as base, but will vary
            show_remarks=show_remarks,
            show_marks=show_marks,
            show_summary=show_summary
        )
        
        # Get orchestrator with annotation
        logger.info("Initializing HUMAN-LIKE evaluation pipeline...")
        orchestrator = ServiceFactory.get_orchestrator(
            ocr_provider="gemini",
            llm_provider=LLMProvider.GEMINI,
            annotation_config=annotation_config
        )
        
        # Create output path
        output_pdf_path = temp_manager.create_temp_path(suffix="_checked.pdf")
        
        # Run complete pipeline
        logger.info("Starting human-like copy checking pipeline...")
        annotated_path, summary, ocr_result = orchestrator.evaluate_answer_sheet(
            input_pdf_path,
            output_pdf_path,
            eval_request
        )
        
        logger.info("✓ Copy checking completed successfully!")
        logger.info(f"Questions checked: {summary.total_questions}")
        logger.info(f"Final score: {summary.total_marks_awarded}/{summary.total_max_marks} "
                   f"({summary.percentage}%) - Grade: {summary.grade}")
        
        # Prepare response with metadata headers
        response = FileResponse(
            path=str(annotated_path),
            media_type="application/pdf",
            filename=f"checked_{file.filename}",
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
                "X-Annotation-Mode": "Human-Like with Vision AI",
                "X-Metadata-Extracted": "true" if ocr_result.metadata.subject else "false"
            }
        )
        
        logger.info(f"Returning checked PDF: checked_{file.filename}")
        logger.info("=" * 80)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Copy checking failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Copy checking failed: {str(e)}"
        )
    finally:
        # Cleanup happens automatically
        pass


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Answer Sheet Checker (Human-Like)",
        "version": "4.0.0 - Human-Like Annotations",
        "mode": "Teacher-Style Copy Checker with Vision AI"
    }


@router.get("/info")
async def service_info():
    """Service information and capabilities"""
    return {
        "service": "AI Answer Sheet Checker",
        "version": "4.0.0",
        "mode": "Human-Like Copy Checking with Vision AI",
        "tagline": "Red pen annotations that look and feel like a real teacher checked your copy",
        "description": "Automated evaluation with intelligent handwritten-style annotations",
        
        "architecture": {
            "phase_1": "OCR: Extract metadata + questions from all pages",
            "phase_2": "Evaluation: AI grades each question with detailed feedback",
            "phase_3": "Annotation: Gemini Vision finds blank spaces and writes naturally"
        },
        
        "key_innovations": [
            "🎯 Vision AI detects optimal annotation positions",
            "✍️ Handwriting-style fonts and natural placement",
            "🔴 Red pen effect for authentic teacher feel",
            "📍 Marks appear near answers, not in boxes",
            "✓/✗ Visual indicators for correct/incorrect",
            "📝 Multi-line remarks that wrap naturally",
            "🎨 Ovals and underlines for emphasis",
            "📄 Handles multi-page questions intelligently"
        ],
        
        "features": [
            "✅ Auto-extracts subject and total marks",
            "✅ Identifies all questions across pages",
            "✅ Question-wise evaluation with AI",
            "✅ Human-like handwritten annotations",
            "✅ Intelligent coordinate detection",
            "✅ Partial credit support",
            "✅ Strengths and improvements",
            "✅ Letter grade assignment",
            "✅ Professional summary page"
        ],
        
        "how_it_works": {
            "step_1": "Upload PDF - AI extracts questions and metadata",
            "step_2": "AI evaluates each question individually",
            "step_3": "Vision AI finds blank spaces near answers",
            "step_4": "Writes marks and remarks naturally in red pen",
            "step_5": "Download checked copy with human-like annotations"
        },
        
        "annotation_details": {
            "coordinate_detection": "Gemini Vision API",
            "font_style": "Handwriting-like (Helvetica with variations)",
            "color": "Red pen (RGB: 0.8, 0.1, 0.1)",
            "placement": "Smart - near answers in blank spaces",
            "visual_elements": ["Checkmarks", "Crosses", "Ovals", "Underlines"],
            "fallback": "Right margin if no blank space found"
        },
        
        "edge_cases_handled": [
            "✓ Multi-page questions",
            "✓ Very full pages (uses margins)",
            "✓ Multi-column layouts",
            "✓ Dense handwriting (finds gaps)",
            "✓ Coordinate validation and bounds checking",
            "✓ Graceful fallback for failed detection"
        ],
        
        "api_usage": {
            "required": ["file (PDF)"],
            "optional": [
                "subject (auto-extracted)",
                "total_marks (auto-extracted)",
                "marking_scheme (generic default)",
                "strict_marking (default: false)",
                "include_partial_credit (default: true)"
            ]
        },
        
        "supported_providers": {
            "ocr": ["Gemini Vision"],
            "evaluation": ["Gemini LLM"],
            "coordinate_detection": ["Gemini Vision 2.0"]
        },
        
        "requirements": [
            "GEMINI_API_KEY environment variable",
            "PDF files under 50MB",
            "Clear handwritten answers"
        ],
        
        "example_curl": """
curl -X POST "http://localhost:8000/evaluate" \\
  -F "file=@student_answer.pdf" \\
  -F "subject=Physics" \\
  -F "marking_scheme=Theory: 6 marks, Diagram: 2 marks, Calculation: 2 marks" \\
  -o checked_copy.pdf
        """,
        
        "endpoints": {
            "/evaluate": "POST - Main evaluation endpoint (human-like mode)",
            "/health": "GET - Health check",
            "/info": "GET - Service information",
            "/docs": "GET - Interactive API documentation"
        },
        
        "comparison": {
            "old_system": "Box annotations in margins (robotic)",
            "new_system": "Natural handwritten marks near answers (human-like)"
        }
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🎓 AI Answer Sheet Checker - Human-Like Mode",
        "version": "4.0.0",
        "mode": "Teacher-Style Copy Checker with Vision AI",
        "tagline": "Your digital teacher that checks copies with a red pen ✍️",
        "innovation": "First AI checker with Vision-guided handwritten annotations",
        "documentation": "/docs",
        "health_check": "/health",
        "service_info": "/info",
        "quick_start": {
            "1": "Upload your answer sheet PDF",
            "2": "AI extracts questions and evaluates each one",
            "3": "Vision AI finds blank spaces on pages",
            "4": "Get back checked copy with natural red pen marks",
            "result": "Looks like a real teacher checked it! ✓"
        },
        "try_it": "POST /evaluate with your PDF to see the magic!"
    }