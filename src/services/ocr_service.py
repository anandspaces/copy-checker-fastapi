import time
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional

from src.schemas import OCRResult


class OCRService:
    """OCR service using Tesseract for text extraction"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR service
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_text_from_image(
        self, 
        image: np.ndarray, 
        page_number: int,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        Extract text from preprocessed image using OCR
        
        Args:
            image: Preprocessed numpy array image (grayscale)
            page_number: Page number for tracking
            lang: Tesseract language code
            
        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Perform OCR with configuration for better accuracy
            # --psm 6: Assume a single uniform block of text
            # --oem 3: Use default OCR Engine Mode (LSTM)
            custom_config = r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(
                pil_image,
                lang=lang,
                config=custom_config
            )
            
            # Get confidence data
            data = pytesseract.image_to_data(
                pil_image,
                lang=lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [
                float(conf) for conf in data['conf'] 
                if conf != '-1' and str(conf).strip()
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return OCRResult(
                page_number=page_number,
                raw_text=text.strip(),
                confidence=avg_confidence / 100.0,  # Normalize to 0-1
                processing_time_ms=round(processing_time, 2)
            )
            
        except Exception as e:
            # Return empty result on failure with error info
            processing_time = (time.time() - start_time) * 1000
            return OCRResult(
                page_number=page_number,
                raw_text=f"[OCR Error: {str(e)}]",
                confidence=0.0,
                processing_time_ms=round(processing_time, 2)
            )
    
    def extract_text_from_file(
        self, 
        image_path: Path, 
        page_number: int,
        lang: str = 'eng'
    ) -> OCRResult:
        """
        Extract text from image file
        
        Args:
            image_path: Path to image file
            page_number: Page number for tracking
            lang: Tesseract language code
            
        Returns:
            OCRResult with extracted text and metadata
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            return self.extract_text_from_image(image, page_number, lang)
            
        except Exception as e:
            return OCRResult(
                page_number=page_number,
                raw_text=f"[File Read Error: {str(e)}]",
                confidence=0.0,
                processing_time_ms=0.0
            )
    
    @staticmethod
    def is_text_valid(ocr_result: OCRResult, min_length: int = 10) -> bool:
        """
        Check if OCR result contains valid text
        
        Args:
            ocr_result: OCR result to validate
            min_length: Minimum text length to consider valid
            
        Returns:
            True if text appears valid
        """
        text = ocr_result.raw_text.strip()
        
        # Check for error markers
        if text.startswith('[') and 'Error' in text:
            return False
        
        # Check minimum length
        if len(text) < min_length:
            return False
        
        # Check if mostly whitespace
        if len(text.replace(' ', '').replace('\n', '')) < min_length // 2:
            return False
        
        return True