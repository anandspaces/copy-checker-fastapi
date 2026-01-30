# src/services/ocr_service.py

from abc import ABC, abstractmethod
import logging
import time
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from src.schemas import OCRResult, DocumentOCRResult

load_dotenv()
logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services"""
    
    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: Path) -> DocumentOCRResult:
        """
        Extract text from all pages of a PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentOCRResult with text from all pages
        """
        pass
    
    @abstractmethod
    def extract_text_from_page(self, image_bytes: bytes, page_number: int) -> OCRResult:
        """
        Extract text from a single page image
        
        Args:
            image_bytes: Image data as bytes
            page_number: Page number (1-indexed)
            
        Returns:
            OCRResult with extracted text
        """
        pass


class GeminiOCRService(OCRService):
    """OCR service using Google Gemini Vision API"""
    
    def __init__(self, model_name: str = "gemini-3-flash-preview", max_retries: int = 3):
        """
        Initialize Gemini OCR service
        
        Args:
            model_name: Gemini model to use
            max_retries: Maximum retry attempts for failed requests
        """
        logger.info(f"Initializing GeminiOCRService with model: {model_name}")
        
        self.model_name = model_name
        self.max_retries = max_retries
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiOCRService initialized successfully")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> DocumentOCRResult:
        """
        Extract text from all pages of a PDF using Gemini Vision
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentOCRResult with all pages processed
        """
        logger.info(f"Starting OCR for PDF: {pdf_path}")
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"PDF has {total_pages} pages. Starting extraction...")
            
            page_results: List[OCRResult] = []
            
            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                # Convert page to high-quality image
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Extract text using Gemini
                ocr_result = self.extract_text_from_page(img_data, page_num + 1)
                page_results.append(ocr_result)
                
                logger.info(f"Page {page_num + 1} OCR complete. "
                          f"Confidence: {ocr_result.confidence:.2f}, "
                          f"Text length: {len(ocr_result.raw_text)}")
            
            doc.close()
            
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = DocumentOCRResult(
                total_pages=total_pages,
                pages=page_results,
                total_processing_time_ms=round(total_time, 2),
                ocr_provider="Gemini Vision"
            )
            
            logger.info(f"OCR complete for all {total_pages} pages. "
                       f"Total time: {total_time:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR failed for PDF: {str(e)}", exc_info=True)
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    def extract_text_from_page(self, image_bytes: bytes, page_number: int) -> OCRResult:
        """
        Extract text from a single page image using Gemini Vision
        
        Args:
            image_bytes: PNG image data
            page_number: Page number (1-indexed)
            
        Returns:
            OCRResult with extracted text and confidence
        """
        start_time = time.time()
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[Page {page_number}] OCR attempt {attempt}/{self.max_retries}")
                
                # Build OCR prompt
                prompt = self._build_ocr_prompt()
                
                # Create content for Gemini
                contents = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png"
                    )
                ]
                
                # Call Gemini Vision API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Deterministic extraction
                        top_p=0.95,
                        max_output_tokens=4096,
                    )
                )
                
                extracted_text = response.text.strip()
                processing_time = (time.time() - start_time) * 1000
                
                # Validate extraction
                if not extracted_text:
                    logger.warning(f"[Page {page_number}] Empty OCR result")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        extracted_text = "[No text could be extracted from this page]"
                
                confidence = self._estimate_confidence(extracted_text)
                
                return OCRResult(
                    page_number=page_number,
                    raw_text=extracted_text,
                    confidence=confidence,
                    processing_time_ms=round(processing_time, 2)
                )
                
            except Exception as e:
                logger.error(f"[Page {page_number}] OCR attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"[Page {page_number}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    processing_time = (time.time() - start_time) * 1000
                    logger.error(f"[Page {page_number}] All OCR attempts failed")
                    
                    return OCRResult(
                        page_number=page_number,
                        raw_text=f"[OCR Error: {str(e)}]",
                        confidence=0.0,
                        processing_time_ms=round(processing_time, 2)
                    )
    
    def _build_ocr_prompt(self) -> str:
        """Build optimized OCR extraction prompt"""
        return """You are an expert OCR system. Extract ALL text from this answer sheet image.

**Instructions:**
1. Extract handwritten and printed text exactly as written
2. Preserve line breaks and paragraph structure
3. Include mathematical equations, formulas, and symbols
4. Include diagram labels and annotations
5. If text is unclear or illegible, mark it as [UNCLEAR]
6. Maintain the reading order (top to bottom, left to right)
7. Do not add any commentary - only extract the actual text

**Output Format:**
Return ONLY the extracted text, nothing else.

Begin extraction:"""
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate OCR confidence based on text quality
        
        Args:
            text: Extracted text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not text or len(text) < 10:
            return 0.3
        
        if text.startswith('[') and 'Error' in text:
            return 0.0
        
        # Check for unclear markers
        unclear_count = text.count('[UNCLEAR]')
        if unclear_count > 0:
            unclear_ratio = unclear_count / max(len(text) / 100, 1)
            confidence = max(0.5, 1.0 - unclear_ratio)
            return round(confidence, 2)
        
        # Estimate based on length and structure
        if len(text) > 500:
            return 0.95
        elif len(text) > 200:
            return 0.9
        elif len(text) > 100:
            return 0.85
        elif len(text) > 50:
            return 0.75
        else:
            return 0.6