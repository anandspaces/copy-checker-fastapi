# src/services/orchestrator_service.py

import logging
from pathlib import Path
from typing import Tuple

from src.schemas import (
    EvaluationRequest,
    EvaluationSummary,
    OCRResult,
    AnnotationConfig
)
from src.services.ocr_service import OCRService
from src.services.evaluation_service import EvaluationService
from src.services.pdf_annotation_service import PDFAnnotationService

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates the complete evaluation pipeline:
    1. OCR Extraction (metadata + questions from all pages)
    2. Question-by-question Evaluation
    3. PDF Annotation (question-wise marks)
    """
    
    def __init__(
        self,
        ocr_service: OCRService,
        evaluation_service: EvaluationService,
        annotation_service: PDFAnnotationService
    ):
        """
        Initialize orchestrator with required services
        
        Args:
            ocr_service: Service for text extraction
            evaluation_service: Service for answer evaluation
            annotation_service: Service for PDF annotation
        """
        logger.info("Initializing EvaluationOrchestrator (Question-wise)")
        self.ocr_service = ocr_service
        self.evaluation_service = evaluation_service
        self.annotation_service = annotation_service
        logger.info("EvaluationOrchestrator initialized successfully")
    
    def evaluate_answer_sheet(
        self,
        pdf_path: Path,
        output_path: Path,
        request: EvaluationRequest
    ) -> Tuple[Path, EvaluationSummary, OCRResult]:
        """
        Complete evaluation pipeline
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for annotated output PDF
            request: Evaluation request with parameters
            
        Returns:
            Tuple of (annotated_pdf_path, evaluation_summary, ocr_result)
        """
        logger.info("=" * 80)
        logger.info("STARTING QUESTION-WISE EVALUATION PIPELINE")
        logger.info(f"Input PDF: {pdf_path}")
        logger.info(f"Subject: {request.subject or 'Auto-extract'}")
        logger.info(f"Total Marks: {request.total_marks or 'Auto-extract'}")
        logger.info("=" * 80)
        
        try:
            # PHASE 1: OCR Extraction (Metadata + Questions)
            logger.info("PHASE 1: OCR TEXT EXTRACTION")
            logger.info("-" * 80)
            ocr_result = self._run_ocr_phase(pdf_path)
            
            logger.info(f"OCR Phase Complete:")
            logger.info(f"  - Metadata extracted: Subject={ocr_result.metadata.subject}, "
                       f"Total Marks={ocr_result.metadata.total_marks}")
            logger.info(f"  - Questions extracted: {len(ocr_result.questions)}")
            logger.info(f"  - Total OCR time: {ocr_result.processing_time_ms:.0f}ms")
            
            # Use extracted metadata if not provided
            subject = request.subject or ocr_result.metadata.subject or "Unknown Subject"
            total_marks = request.total_marks or ocr_result.metadata.total_marks
            
            # PHASE 2: Question-wise Evaluation
            logger.info("")
            logger.info("PHASE 2: QUESTION-BY-QUESTION EVALUATION")
            logger.info("-" * 80)
            
            evaluation_summary = self._run_evaluation_phase(
                ocr_result.questions,
                subject,
                request.marking_scheme,
                total_marks,
                request.strict_marking,
                request.include_partial_credit
            )
            
            logger.info(f"Evaluation Phase Complete:")
            logger.info(f"  - Questions evaluated: {evaluation_summary.total_questions}")
            logger.info(f"  - Final Score: {evaluation_summary.total_marks_awarded}/"
                       f"{evaluation_summary.total_max_marks} "
                       f"({evaluation_summary.percentage}%)")
            logger.info(f"  - Grade: {evaluation_summary.grade}")
            
            # PHASE 3: PDF Annotation (Question-wise)
            logger.info("")
            logger.info("PHASE 3: PDF ANNOTATION (QUESTION-WISE)")
            logger.info("-" * 80)
            
            annotated_path = self._run_annotation_phase(
                pdf_path,
                output_path,
                evaluation_summary
            )
            
            logger.info(f"Annotation Phase Complete: {annotated_path}")
            
            logger.info("=" * 80)
            logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total processing time: {evaluation_summary.processing_time_seconds:.2f}s")
            logger.info("=" * 80)
            
            return annotated_path, evaluation_summary, ocr_result
            
        except Exception as e:
            logger.error("EVALUATION PIPELINE FAILED", exc_info=True)
            raise Exception(f"Evaluation pipeline failed: {str(e)}")
    
    def _run_ocr_phase(self, pdf_path: Path) -> OCRResult:
        """
        Phase 1: Extract metadata and questions from PDF
        
        Args:
            pdf_path: Path to input PDF
            
        Returns:
            OCRResult with metadata and questions
        """
        logger.info("Starting OCR extraction (metadata + questions)...")
        
        try:
            ocr_result = self.ocr_service.extract_from_pdf(pdf_path)
            
            # Log OCR statistics
            logger.info(f"OCR Statistics:")
            logger.info(f"  - Total pages: {ocr_result.total_pages}")
            logger.info(f"  - Questions found: {len(ocr_result.questions)}")
            logger.info(f"  - Subject: {ocr_result.metadata.subject}")
            logger.info(f"  - Total marks: {ocr_result.metadata.total_marks}")
            logger.info(f"  - Provider: {ocr_result.ocr_provider}")
            
            # Log question details
            for q in ocr_result.questions[:5]:  # First 5
                logger.debug(f"  Q{q.question_number}: {q.allocated_marks or 'N/A'} marks, "
                           f"pages {q.page_numbers}")
            
            if len(ocr_result.questions) > 5:
                logger.debug(f"  ... and {len(ocr_result.questions) - 5} more questions")
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"OCR phase failed: {str(e)}", exc_info=True)
            raise
    
    def _run_evaluation_phase(
        self,
        questions,
        subject,
        marking_scheme,
        total_marks,
        strict_marking,
        include_partial_credit
    ) -> EvaluationSummary:
        """
        Phase 2: Evaluate each question
        
        Args:
            questions: List of QuestionData
            subject: Subject name
            marking_scheme: Marking criteria
            total_marks: Total marks
            strict_marking: Strict mode
            include_partial_credit: Partial credit
            
        Returns:
            EvaluationSummary with results
        """
        logger.info("Starting question-by-question evaluation...")
        
        try:
            evaluation_summary = self.evaluation_service.evaluate_questions(
                questions=questions,
                subject=subject,
                marking_scheme=marking_scheme,
                total_marks=total_marks,
                strict_marking=strict_marking,
                include_partial_credit=include_partial_credit
            )
            
            # Log evaluation statistics
            logger.info(f"Evaluation Statistics:")
            logger.info(f"  - Questions evaluated: {evaluation_summary.total_questions}")
            logger.info(f"  - Total marks awarded: {evaluation_summary.total_marks_awarded}")
            logger.info(f"  - Total max marks: {evaluation_summary.total_max_marks}")
            logger.info(f"  - Percentage: {evaluation_summary.percentage}%")
            logger.info(f"  - Grade: {evaluation_summary.grade}")
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"Evaluation phase failed: {str(e)}", exc_info=True)
            raise
    
    def _run_annotation_phase(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        evaluation_summary: EvaluationSummary
    ) -> Path:
        """
        Phase 3: Annotate PDF with question-wise results
        
        Args:
            input_pdf_path: Original PDF path
            output_pdf_path: Output path for annotated PDF
            evaluation_summary: Evaluation results
            
        Returns:
            Path to annotated PDF
        """
        logger.info("Starting PDF annotation (question-wise)...")
        
        try:
            annotated_path = self.annotation_service.annotate_pdf(
                input_pdf_path,
                output_pdf_path,
                evaluation_summary.questions_evaluated,
                evaluation_summary
            )
            
            logger.info(f"PDF annotated successfully: {annotated_path}")
            
            return annotated_path
            
        except Exception as e:
            logger.error(f"Annotation phase failed: {str(e)}", exc_info=True)
            raise