# src/services/orchestrator_service.py

import logging
from pathlib import Path
from typing import Tuple

from src.schemas import (
    EvaluationRequest,
    EvaluationSummary,
    DocumentOCRResult,
    AnnotationConfig
)
from src.services.ocr_service import OCRService
from src.services.evaluation_service import EvaluationService
from src.services.pdf_annotation_service import PDFAnnotationService

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates the complete evaluation pipeline:
    1. OCR Extraction (all pages)
    2. Page-by-page Evaluation
    3. PDF Annotation
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
        logger.info("Initializing EvaluationOrchestrator")
        self.ocr_service = ocr_service
        self.evaluation_service = evaluation_service
        self.annotation_service = annotation_service
        logger.info("EvaluationOrchestrator initialized successfully")
    
    def evaluate_answer_sheet(
        self,
        pdf_path: Path,
        output_path: Path,
        request: EvaluationRequest
    ) -> Tuple[Path, EvaluationSummary, DocumentOCRResult]:
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
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info(f"Input PDF: {pdf_path}")
        logger.info(f"Subject: {request.subject}")
        logger.info(f"Max marks per page: {request.max_marks_per_page}")
        logger.info("=" * 80)
        
        try:
            # PHASE 1: OCR Extraction
            logger.info("PHASE 1: OCR TEXT EXTRACTION")
            logger.info("-" * 80)
            ocr_result = self._run_ocr_phase(pdf_path)
            logger.info(f"OCR Phase Complete: {ocr_result.total_pages} pages processed")
            logger.info(f"Total OCR time: {ocr_result.total_processing_time_ms:.0f}ms")
            
            # PHASE 2: Evaluation
            logger.info("")
            logger.info("PHASE 2: PAGE-BY-PAGE EVALUATION")
            logger.info("-" * 80)
            evaluation_summary = self._run_evaluation_phase(ocr_result, request)
            logger.info(f"Evaluation Phase Complete")
            logger.info(f"Final Score: {evaluation_summary.total_marks_awarded}/"
                       f"{evaluation_summary.total_max_marks} "
                       f"({evaluation_summary.percentage}%) - Grade: {evaluation_summary.grade}")
            
            # PHASE 3: Annotation
            logger.info("")
            logger.info("PHASE 3: PDF ANNOTATION")
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
    
    def _run_ocr_phase(self, pdf_path: Path) -> DocumentOCRResult:
        """
        Phase 1: Extract text from all pages
        
        Args:
            pdf_path: Path to input PDF
            
        Returns:
            DocumentOCRResult with extracted text
        """
        logger.info("Starting OCR extraction...")
        
        try:
            ocr_result = self.ocr_service.extract_text_from_pdf(pdf_path)
            
            # Log OCR statistics
            total_chars = sum(len(page.raw_text) for page in ocr_result.pages)
            avg_confidence = sum(page.confidence for page in ocr_result.pages) / len(ocr_result.pages)
            
            logger.info(f"OCR Statistics:")
            logger.info(f"  - Pages processed: {ocr_result.total_pages}")
            logger.info(f"  - Total characters extracted: {total_chars}")
            logger.info(f"  - Average confidence: {avg_confidence*100:.1f}%")
            logger.info(f"  - Provider: {ocr_result.ocr_provider}")
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"OCR phase failed: {str(e)}", exc_info=True)
            raise
    
    def _run_evaluation_phase(
        self,
        ocr_result: DocumentOCRResult,
        request: EvaluationRequest
    ) -> EvaluationSummary:
        """
        Phase 2: Evaluate each page
        
        Args:
            ocr_result: OCR results from phase 1
            request: Evaluation request
            
        Returns:
            EvaluationSummary with marks and feedback
        """
        logger.info("Starting page-by-page evaluation...")
        
        try:
            # Extract texts from OCR results
            extracted_texts = [page.raw_text for page in ocr_result.pages]
            
            # Run evaluation
            evaluation_summary = self.evaluation_service.evaluate_document(
                extracted_texts=extracted_texts,
                subject=request.subject,
                marking_scheme=request.marking_scheme,
                max_marks_per_page=request.max_marks_per_page,
                strict_marking=request.strict_marking,
                include_partial_credit=request.include_partial_credit
            )
            
            # Log evaluation statistics
            logger.info(f"Evaluation Statistics:")
            logger.info(f"  - Pages evaluated: {evaluation_summary.total_pages}")
            logger.info(f"  - Total marks: {evaluation_summary.total_marks_awarded}/"
                       f"{evaluation_summary.total_max_marks}")
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
        Phase 3: Annotate PDF with results
        
        Args:
            input_pdf_path: Original PDF path
            output_pdf_path: Output path for annotated PDF
            evaluation_summary: Evaluation results
            
        Returns:
            Path to annotated PDF
        """
        logger.info("Starting PDF annotation...")
        
        try:
            annotated_path = self.annotation_service.annotate_pdf(
                input_pdf_path,
                output_pdf_path,
                evaluation_summary.pages_evaluated,
                evaluation_summary
            )
            
            logger.info(f"PDF annotated successfully: {annotated_path}")
            
            return annotated_path
            
        except Exception as e:
            logger.error(f"Annotation phase failed: {str(e)}", exc_info=True)
            raise