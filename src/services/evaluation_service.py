import time
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
from src.services.ocr_service import OCRService
from src.services.vision_service import VisionService
from src.services.llm_service import LLMService
from src.services.pdf_annotation_service import PDFAnnotationService


class EvaluationService:
    """Main evaluation service orchestrating the entire pipeline"""
    
    def __init__(
        self,
        llm_service: LLMService,
        annotation_config: AnnotationConfig = None
    ):
        """
        Initialize evaluation service
        
        Args:
            llm_service: Configured LLM service
            annotation_config: Configuration for PDF annotations
        """
        self.ocr_service = OCRService()
        self.vision_service = VisionService()
        self.llm_service = llm_service
        self.annotation_service = PDFAnnotationService(annotation_config)
    
    def evaluate_pdf(
        self,
        pdf_path: Path,
        output_path: Path,
        request: EvaluationRequest
    ) -> Tuple[Path, EvaluationSummary]:
        """
        Evaluate PDF answer sheet and return annotated PDF
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output annotated PDF
            request: Evaluation request parameters
            
        Returns:
            Tuple of (output_path, evaluation_summary)
        """
        start_time = time.time()
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            # Process each page
            page_evaluations = []
            
            for page_num in range(total_pages):
                page_eval = self._process_page(
                    doc,
                    page_num,
                    request
                )
                page_evaluations.append(page_eval)
            
            doc.close()
            
            # Calculate summary
            summary = self._create_summary(page_evaluations, start_time)
            
            # Annotate PDF
            annotated_path = self.annotation_service.annotate_pdf(
                pdf_path,
                output_path,
                page_evaluations,
                summary
            )
            
            return annotated_path, summary
            
        except Exception as e:
            raise Exception(f"Evaluation failed: {str(e)}")
    
    def _process_page(
        self,
        doc: fitz.Document,
        page_num: int,
        request: EvaluationRequest
    ) -> PageEvaluation:
        """
        Process a single page through the entire pipeline
        
        Args:
            doc: PyMuPDF document
            page_num: Page number (0-indexed)
            request: Evaluation request
            
        Returns:
            PageEvaluation for this page
        """
        page_number = page_num + 1  # Convert to 1-indexed
        
        # Step 1: Convert page to image
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Step 2: Extract vision metadata
        metadata = self.vision_service.extract_metadata(img, page_number)
        
        # Step 3: Preprocess image for OCR
        preprocessed = self.vision_service.preprocess_image(img)
        
        # Step 4: Extract text using OCR
        ocr_result = self.ocr_service.extract_text_from_image(
            preprocessed,
            page_number
        )
        
        # Step 5: Evaluate using LLM
        llm_request = LLMEvaluationRequest(
            page_number=page_number,
            ocr_text=ocr_result.raw_text,
            subject=request.subject,
            marking_scheme=request.marking_scheme,
            vision_metadata=metadata,
            max_marks=request.max_marks_per_page
        )
        
        llm_response = self.llm_service.evaluate_page(llm_request)
        
        # Step 6: Create page evaluation
        page_eval = PageEvaluation(
            page_number=page_number,
            marks_awarded=llm_response.marks_awarded,
            max_marks=llm_response.max_marks,
            remarks=llm_response.remarks,
            confidence=0.8  # Can be adjusted based on OCR confidence + other factors
        )
        
        return page_eval
    
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
        total_marks_awarded = sum(e.marks_awarded for e in page_evaluations)
        total_max_marks = sum(e.max_marks for e in page_evaluations)
        
        percentage = (total_marks_awarded / total_max_marks * 100) if total_max_marks > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        return EvaluationSummary(
            total_pages=len(page_evaluations),
            total_marks_awarded=round(total_marks_awarded, 2),
            total_max_marks=round(total_max_marks, 2),
            percentage=round(percentage, 2),
            pages_evaluated=page_evaluations,
            processing_time_seconds=round(processing_time, 2)
        )