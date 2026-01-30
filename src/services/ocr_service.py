# src/services/ocr_service.py

from abc import ABC, abstractmethod
import logging
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.schemas import OCRResult, PaperMetadata, QuestionData

load_dotenv()
logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services"""
    
    @abstractmethod
    def extract_from_pdf(self, pdf_path: Path) -> OCRResult:
        """
        Extract metadata and questions from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCRResult with metadata and questions
        """
        pass


class GeminiOCRService(OCRService):
    """OCR service using Google Gemini Vision API - Question-wise extraction"""
    
    def __init__(
        self, 
        model_name: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        max_workers: int = 10
    ):
        """
        Initialize Gemini OCR service
        
        Args:
            model_name: Gemini model to use
            max_retries: Maximum retry attempts
            max_workers: Maximum concurrent threads
        """
        logger.info(f"Initializing GeminiOCRService with model: {model_name}")
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiOCRService initialized successfully")
    
    def extract_from_pdf(self, pdf_path: Path) -> OCRResult:
        """
        Extract metadata and questions from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCRResult with structured data
        """
        logger.info(f"Starting OCR extraction from: {pdf_path}")
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"PDF has {total_pages} pages")
            
            # Step 1: Extract metadata from first page
            logger.info("Extracting metadata from first page...")
            metadata = self._extract_metadata(doc[0])
            logger.info(f"Metadata extracted: Subject={metadata.subject}, Total Marks={metadata.total_marks}")
            
            # Step 2: Extract all page images
            logger.info("Converting pages to images...")
            page_images = []
            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                page_images.append((page_num + 1, img_data))
            
            doc.close()
            
            # Step 3: Extract questions from all pages
            logger.info(f"Extracting questions from all {total_pages} pages...")
            questions = self._extract_questions_from_pages(page_images)
            
            logger.info(f"Extracted {len(questions)} questions")
            
            processing_time = (time.time() - start_time) * 1000
            
            result = OCRResult(
                metadata=metadata,
                questions=questions,
                total_pages=total_pages,
                processing_time_ms=round(processing_time, 2),
                ocr_provider="Gemini Vision (Question-wise)"
            )
            
            logger.info(f"OCR complete. Time: {processing_time:.0f}ms")
            return result
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}", exc_info=True)
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    def _extract_metadata(self, first_page: fitz.Page) -> PaperMetadata:
        """
        Extract metadata from first page
        
        Args:
            first_page: First page of PDF
            
        Returns:
            PaperMetadata with extracted information
        """
        try:
            # Convert first page to image
            pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Prompt for metadata extraction
            prompt = """Extract the following information from this answer sheet's first page:

1. Subject name
2. Total marks/Maximum marks
3. Student name (if visible)
4. Roll number (if visible)
5. Exam date (if visible)

Return ONLY a JSON object with this exact structure:
{
  "subject": "subject name or null",
  "total_marks": numeric value or null,
  "student_name": "name or null",
  "roll_number": "roll number or null",
  "exam_date": "date or null",
  "additional_info": {}
}

If any information is not found, use null. Be precise with numbers."""

            contents = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=img_data, mime_type="image/png")
            ]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text.strip())
            
            return PaperMetadata(
                subject=data.get("subject"),
                total_marks=float(data["total_marks"]) if data.get("total_marks") else None,
                student_name=data.get("student_name"),
                roll_number=data.get("roll_number"),
                exam_date=data.get("exam_date"),
                additional_info=data.get("additional_info", {})
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return PaperMetadata()
    
    def _extract_questions_from_pages(self, page_images: List[tuple]) -> List[QuestionData]:
        """
        Extract questions from all pages using parallel processing
        
        Args:
            page_images: List of (page_number, image_bytes) tuples
            
        Returns:
            List of QuestionData objects
        """
        logger.info("Starting parallel question extraction...")
        
        # Extract text from each page first
        page_texts = self._extract_page_texts_parallel(page_images)
        
        # Combine all page texts
        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
            f"[PAGE {page_num}]\n{text}" 
            for page_num, text in page_texts
        )
        
        # Now extract structured questions from combined text
        logger.info("Parsing combined text into questions...")
        questions = self._parse_into_questions(combined_text, page_texts)
        
        return questions
    
    def _extract_page_texts_parallel(self, page_images: List[tuple]) -> List[tuple]:
        """
        Extract text from all pages in parallel
        
        Returns:
            List of (page_number, text) tuples
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self._extract_text_from_page, img_data, page_num): page_num
                for page_num, img_data in page_images
            }
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    text = future.result()
                    results.append((page_num, text))
                    logger.info(f"✓ Page {page_num}/{len(page_images)} text extracted")
                except Exception as e:
                    logger.error(f"✗ Page {page_num} failed: {str(e)}")
                    results.append((page_num, f"[OCR Error on page {page_num}]"))
        
        # Sort by page number
        results.sort(key=lambda x: x[0])
        return results
    
    def _extract_text_from_page(self, image_bytes: bytes, page_number: int) -> str:
        """
        Extract text from a single page
        
        Args:
            image_bytes: PNG image data
            page_number: Page number
            
        Returns:
            Extracted text
        """
        prompt = """Extract ALL text from this answer sheet page.

Include:
- Question numbers and text
- Student's handwritten answers
- Any marks written (like "5/10" or "Marks: 7")
- Diagrams labels

Preserve structure and reading order (top to bottom).

Return ONLY the extracted text."""

        contents = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                
                return response.text.strip()
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def _parse_into_questions(
        self, 
        combined_text: str, 
        page_texts: List[tuple]
    ) -> List[QuestionData]:
        """
        Parse combined text into structured questions using Gemini
        
        Args:
            combined_text: All pages combined
            page_texts: List of (page_num, text) for page mapping
            
        Returns:
            List of QuestionData
        """
        prompt = f"""You are parsing an answer sheet. Extract each question with its answer.

ANSWER SHEET TEXT:
{combined_text[:15000]}  

Parse into questions and return ONLY a JSON array:
[
  {{
    "question_number": "1" or "2a" or "3.i",
    "question_text": "the actual question",
    "student_answer": "student's written answer",
    "allocated_marks": 10 (or null if not found),
    "page_numbers": [1, 2] (pages where this question appears)
  }},
  ...
]

Rules:
- Extract question number, question text, and student's answer separately
- Look for marks like "10 marks", "[10]", "Marks: 5" 
- Track which pages each question spans
- Maintain question order
- If marks not found, use null

Return ONLY valid JSON array, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=8000
                )
            )
            
            questions_data = json.loads(response.text.strip())
            
            # Convert to QuestionData objects
            questions = []
            for q in questions_data:
                questions.append(QuestionData(
                    question_number=str(q.get("question_number", "?")),
                    question_text=q.get("question_text", ""),
                    student_answer=q.get("student_answer", ""),
                    allocated_marks=float(q["allocated_marks"]) if q.get("allocated_marks") else None,
                    page_numbers=q.get("page_numbers", [1])
                ))
            
            logger.info(f"Parsed {len(questions)} questions from text")
            return questions
            
        except Exception as e:
            logger.error(f"Question parsing failed: {e}")
            # Fallback: create single question from all text
            return [QuestionData(
                question_number="1",
                question_text="Complete answer sheet",
                student_answer=combined_text[:3000],
                allocated_marks=None,
                page_numbers=list(range(1, len(page_texts) + 1))
            )]