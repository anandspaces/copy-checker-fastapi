# src/services/evaluation_service.py

import time
import logging
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from src.schemas import (
    EvaluationRequest,
    PageEvaluation,
    EvaluationSummary,
    LLMEvaluationRequest,
    PageMetadata,
    OCRResult,
    AnnotationConfig
)
from src.services.llm_service import LLMService, GeminiOCRService
from src.services.pdf_annotation_service import PDFAnnotationService

# Configure logger
logger = logging.getLogger(__name__)


class GeminiEvaluationService:
    """Evaluation service using Gemini for both OCR and evaluation"""
    
    def __init__(
        self,
        llm_service: LLMService,
        annotation_config: AnnotationConfig = None
    ):
        """
        Initialize Gemini only evaluation service
        
        Args:
            llm_service: Configured LLM service (Gemini)
            annotation_config: Configuration for PDF annotations
        """
        logger.info("Initializing GeminiEvaluationService")
        self.ocr_service = GeminiOCRService()
        self.llm_service = llm_service
        self.annotation_service = PDFAnnotationService(annotation_config)
        logger.info("GeminiEvaluationService initialized successfully")
    
    def evaluate_pdf(
        self,
        pdf_path: Path,
        output_path: Path,
        request: EvaluationRequest
    ) -> Tuple[Path, EvaluationSummary]:
        """
        Evaluate PDF answer sheet using Gemini for OCR and evaluation
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output annotated PDF
            request: Evaluation request parameters
            
        Returns:
            Tuple of (output_path, evaluation_summary)
        """
        start_time = time.time()
        logger.info(f"Starting Gemini PDF evaluation: {pdf_path}")
        logger.info(f"Subject: {request.subject}, Max marks per page: {request.max_marks_per_page}")
        
        try:
            # Open PDF
            logger.debug(f"Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                logger.error("PDF has no pages")
                raise ValueError("PDF has no pages")
            
            logger.info(f"PDF opened successfully. Total pages: {total_pages}")
            
            # Process each page
            page_evaluations = []
            
            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages} with Gemini")
                page_eval = self._process_page(
                    doc,
                    page_num,
                    request
                )
                page_evaluations.append(page_eval)
                logger.info(f"Page {page_num + 1} evaluated: {page_eval.marks_awarded}/{page_eval.max_marks} marks")
            
            doc.close()
            logger.debug("PDF document closed")
            
            # Calculate summary
            logger.info("Calculating evaluation summary")
            summary = self._create_summary(page_evaluations, start_time)
            logger.info(f"Gemini evaluation complete: {summary.total_marks_awarded}/{summary.total_max_marks} ({summary.percentage}%)")
            
            # Annotate PDF
            logger.info(f"Annotating PDF to: {output_path}")
            annotated_path = self.annotation_service.annotate_pdf(
                pdf_path,
                output_path,
                page_evaluations,
                summary
            )
            logger.info(f"PDF annotation complete: {annotated_path}")
            
            return annotated_path, summary
            
        except Exception as e:
            logger.error(f"Gemini evaluation failed: {str(e)}", exc_info=True)
            raise Exception(f"Evaluation failed: {str(e)}")
    
    def _process_page(
        self,
        doc: fitz.Document,
        page_num: int,
        request: EvaluationRequest
    ) -> PageEvaluation:
        """
        Process a single page using Gemini for OCR and evaluation
        
        Args:
            doc: PyMuPDF document
            page_num: Page number (0-indexed)
            request: Evaluation request
            
        Returns:
            PageEvaluation for this page
        """
        page_number = page_num + 1  # Convert to 1-indexed
        logger.debug(f"[Page {page_number}] Starting Gemini page processing")
        
        try:
            # Step 1: Convert page to image (high quality)
            logger.debug(f"[Page {page_number}] Converting to high-quality image")
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_data = pix.tobytes("png")
            logger.debug(f"[Page {page_number}] Image conversion complete. Size: {len(img_data)} bytes")
            
            # Step 2: Extract text using Gemini OCR
            logger.debug(f"[Page {page_number}] Running Gemini OCR")
            ocr_result = self.ocr_service.extract_text_from_image_bytes(
                img_data,
                page_number,
                mime_type="image/png"
            )
            logger.debug(f"[Page {page_number}] Gemini OCR complete. Confidence: {ocr_result.confidence:.2f}, Text length: {len(ocr_result.raw_text)}")
            
            # Step 3: Create minimal metadata (no complex vision processing)
            metadata = PageMetadata(
                page_number=page_number,
                writing_density=0.5,  # Default placeholder
                has_diagrams=False,   # Not detected in this approach
                crossed_out_regions=0,
                text_blocks_count=1,
                skew_angle=0.0
            )
            logger.debug(f"[Page {page_number}] Using default metadata (no vision processing)")
            
            # Step 4: Evaluate using LLM (Gemini)
            logger.debug(f"[Page {page_number}] Sending to Gemini for evaluation")
            llm_request = LLMEvaluationRequest(
                page_number=page_number,
                ocr_text=ocr_result.raw_text,
                subject=request.subject,
                marking_scheme=request.marking_scheme,
                vision_metadata=metadata,
                max_marks=request.max_marks_per_page
            )
            
            llm_response = self.llm_service.evaluate_page(llm_request)
            logger.debug(f"[Page {page_number}] Gemini evaluation complete")
            
            # Step 5: Create page evaluation
            page_eval = PageEvaluation(
                page_number=page_number,
                marks_awarded=llm_response.marks_awarded,
                max_marks=llm_response.max_marks,
                remarks=llm_response.remarks,
                confidence=ocr_result.confidence  # Use Gemini OCR confidence
            )
            
            logger.debug(f"[Page {page_number}] Page evaluation created successfully")
            return page_eval
            
        except Exception as e:
            logger.error(f"[Page {page_number}] Gemini processing failed: {str(e)}", exc_info=True)
            raise
    
    def _create_summary(
        self,
        page_evaluations: List[PageEvaluation],
        start_time: float
    ) -> EvaluationSummary:
        """
        Create evaluation summary from page evaluations
        
        Args:
            page_evaluations: List of page evaluations
            start_time: Processing start time
            
        Returns:
            EvaluationSummary
        """
        logger.debug("Creating Gemini evaluation summary")
        
        total_marks_awarded = sum(e.marks_awarded for e in page_evaluations)
        total_max_marks = sum(e.max_marks for e in page_evaluations)
        
        percentage = (total_marks_awarded / total_max_marks * 100) if total_max_marks > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        summary = EvaluationSummary(
            total_pages=len(page_evaluations),
            total_marks_awarded=round(total_marks_awarded, 2),
            total_max_marks=round(total_max_marks, 2),
            percentage=round(percentage, 2),
            pages_evaluated=page_evaluations,
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.debug(f"Gemini summary created: {summary.total_pages} pages, {processing_time:.2f}s")
        return summary