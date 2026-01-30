# src/services/llm_service.py
# FIXED VERSION - More robust with better error handling and retries

import os
import logging
import json
import base64
from typing import Optional
import time

from google import genai
from google.genai import types

from src.schemas import (
    LLMEvaluationRequest,
    LLMEvaluationResponse,
    OCRResult,
    LLMProvider
)
from dotenv import load_dotenv
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = 'gemini-3-flash-preview'  # DO NOT CHANGE


class LLMService:
    """Service for LLM-based answer evaluation using Google Gemini"""
    
    def __init__(self, provider: LLMProvider = LLMProvider.GEMINI, max_retries: int = 3):
        """
        Initialize LLM service
        
        Args:
            provider: LLM provider to use (currently only Gemini supported)
            max_retries: Maximum number of retries for API calls
        """
        logger.info(f"Initializing LLMService with provider: {provider}")
        
        self.provider = provider
        self.max_retries = max_retries
        
        if self.provider == LLMProvider.GEMINI:
            self._initialize_gemini()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_gemini(self):
        """Initialize Google Gemini using google-genai"""
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please set it before running the application."
            )
        
        # Create client with API key
        self.client = genai.Client(api_key=api_key)
        
        # Model for evaluation
        self.model_name = GEMINI_MODEL
        
        logger.info(f"Gemini initialized successfully with model: {self.model_name}")
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """
        Evaluate a single page using LLM with retry logic
        
        Args:
            request: Evaluation request with OCR text and context
            
        Returns:
            LLMEvaluationResponse with marks and remarks
        """
        logger.debug(f"[Page {request.page_number}] Starting LLM evaluation")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Build evaluation prompt
                prompt = self._build_evaluation_prompt(request)
                logger.debug(f"[Page {request.page_number}] Prompt built, length: {len(prompt)}")
                
                # Call Gemini
                logger.debug(f"[Page {request.page_number}] Calling Gemini API (attempt {attempt}/{self.max_retries})...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistent grading
                        top_p=0.95,
                        top_k=40,
                        candidate_count=1,
                        max_output_tokens=1024,  # Sufficient for JSON response
                        response_mime_type="application/json",  # Force JSON response
                    )
                )
                
                logger.debug(f"[Page {request.page_number}] Gemini API call complete")
                
                # Parse response
                result = self._parse_llm_response(response, request)
                logger.info(f"[Page {request.page_number}] LLM evaluation: {result.marks_awarded}/{result.max_marks}")
                
                return result
                
            except Exception as e:
                logger.error(f"[Page {request.page_number}] LLM evaluation attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt  # 2, 4, 8 seconds
                    logger.info(f"[Page {request.page_number}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, return fallback
                    logger.error(f"[Page {request.page_number}] All retry attempts exhausted")
                    return self._create_fallback_evaluation(request, str(e))
    
    def _build_evaluation_prompt(self, request: LLMEvaluationRequest) -> str:
        """Build evaluation prompt for LLM"""
        
        # Determine max marks
        max_marks = request.max_marks if request.max_marks else 10.0
        
        # Clean and truncate OCR text
        ocr_text = str(request.ocr_text).strip()
        if not ocr_text or ocr_text == "":
            ocr_text = "[No text extracted from answer sheet]"
        
        # Truncate if too long (keep within token limits)
        max_text_length = 3000
        if len(ocr_text) > max_text_length:
            ocr_text = ocr_text[:max_text_length] + "...[truncated]"
        
        # IMPROVED PROMPT - clearer instructions
        prompt = f"""You are an expert examiner evaluating a student's answer sheet.

**Subject**: {request.subject}

**Marking Scheme**:
{request.marking_scheme}

**Student's Answer** (Page {request.page_number}):
{ocr_text}

**Maximum Marks for this page**: {max_marks}

**Instructions**:
1. Evaluate based on correctness, completeness, and clarity
2. Award marks from 0 to {max_marks}
3. Provide brief, constructive feedback (50-100 characters)
4. If the answer is blank or completely wrong, award 0 marks
5. If the answer is partially correct, award proportional marks

**CRITICAL**: Respond with ONLY valid JSON. No extra text before or after.

{{
  "marks_awarded": <number between 0 and {max_marks}>,
  "max_marks": {max_marks},
  "remarks": "<brief feedback, max 100 chars>",
  "reasoning": "<internal note, optional>"
}}

Respond now with valid JSON only:"""
        
        return prompt
    
    def _parse_llm_response(
        self, 
        response, 
        request: LLMEvaluationRequest
    ) -> LLMEvaluationResponse:
        """Parse LLM response and extract structured data"""
        
        try:
            # Get text from response
            response_text = response.text.strip()
            logger.debug(f"[Page {request.page_number}] Raw response length: {len(response_text)}")
            
            # Handle empty response
            if not response_text or response_text == "":
                logger.warning(f"[Page {request.page_number}] Empty response from LLM")
                raise ValueError("Empty response from LLM")
            
            # Try to extract JSON if embedded or truncated
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx+1]
            else:
                json_text = response_text
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate and extract fields with defaults
            marks_awarded = float(data.get("marks_awarded", 0))
            max_marks = float(data.get("max_marks", request.max_marks or 10.0))
            remarks = str(data.get("remarks", "Answer evaluated."))
            reasoning = str(data.get("reasoning", ""))
            
            # Sanitize remarks
            remarks = remarks.strip()
            if not remarks or remarks == "":
                remarks = "Answer reviewed."
            
            # Truncate remarks
            if len(remarks) > 150:
                remarks = remarks[:147] + "..."
            
            # Validate marks range
            if marks_awarded < 0:
                logger.warning(f"[Page {request.page_number}] Negative marks, setting to 0")
                marks_awarded = 0
            if marks_awarded > max_marks:
                logger.warning(f"[Page {request.page_number}] Marks exceed max, capping at {max_marks}")
                marks_awarded = max_marks
            
            # Round marks to 2 decimal places
            marks_awarded = round(marks_awarded, 2)
            max_marks = round(max_marks, 2)
            
            return LLMEvaluationResponse(
                marks_awarded=marks_awarded,
                max_marks=max_marks,
                remarks=remarks,
                reasoning=reasoning[:200] if reasoning else None
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"[Page {request.page_number}] JSON parse error: {e}")
            logger.error(f"[Page {request.page_number}] Response: {response_text[:500]}")
            
            # Fallback: extract what we can from partial JSON or text
            return self._extract_from_text_fallback(response_text, request)
        
        except Exception as e:
            logger.error(f"[Page {request.page_number}] Parsing error: {e}")
            return self._create_fallback_evaluation(request, str(e))
    
    def _extract_from_text_fallback(
        self, 
        text: str, 
        request: LLMEvaluationRequest
    ) -> LLMEvaluationResponse:
        """Fallback parser when JSON fails"""
        logger.warning(f"[Page {request.page_number}] Using fallback parser")
        
        max_marks = request.max_marks or 10.0
        marks = None
        
        # Try to extract marks using regex
        import re
        patterns = [
            r'"?marks[_\s]*awarded"?\s*:\s*(\d+\.?\d*)',
            r'marks\s*:\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:out of|/)\s*(\d+\.?\d*)',
            r'awarded?\s*[:=]\s*(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    marks = float(match.group(1))
                    logger.debug(f"[Page {request.page_number}] Extracted marks using pattern: {pattern}")
                    break
                except:
                    continue
        
        if marks is None:
            # Default to middle value if can't extract
            marks = max_marks * 0.5
            logger.warning(f"[Page {request.page_number}] Could not extract marks, using default: {marks}")
        
        # Validate and clamp marks
        marks = min(max(0, marks), max_marks)
        
        # Extract remarks if possible
        remarks = "Answer evaluated."
        remarks_match = re.search(r'"?remarks"?\s*:\s*"([^"]+)"', text)
        if remarks_match:
            remarks = remarks_match.group(1)[:150]
            logger.debug(f"[Page {request.page_number}] Extracted remarks: {remarks[:50]}...")
        
        return LLMEvaluationResponse(
            marks_awarded=round(marks, 2),
            max_marks=round(max_marks, 2),
            remarks=remarks,
            reasoning="Fallback parser used"
        )
    
    def _create_fallback_evaluation(
        self, 
        request: LLMEvaluationRequest, 
        error: str
    ) -> LLMEvaluationResponse:
        """Create fallback evaluation on error"""
        logger.warning(f"[Page {request.page_number}] Creating fallback evaluation due to: {error}")
        
        max_marks = request.max_marks or 10.0
        
        # Conservative fallback: give partial credit
        fallback_marks = max_marks * 0.3  # 30% as conservative estimate
        
        return LLMEvaluationResponse(
            marks_awarded=round(fallback_marks, 2),
            max_marks=round(max_marks, 2),
            remarks="Evaluation incomplete. Manual review recommended.",
            reasoning=f"Fallback: {error[:100]}"
        )


class GeminiOCRService:
    """OCR service using Gemini Vision for text extraction"""
    
    def __init__(self, max_retries: int = 3):
        """Initialize Gemini OCR service with retry logic"""
        logger.info("Initializing GeminiOCRService")
        
        self.max_retries = max_retries
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please set it before running the application."
            )
        
        # Create client
        self.client = genai.Client(api_key=api_key)
        
        # Model for OCR (vision)
        self.model_name = GEMINI_MODEL
        
        logger.info(f"Gemini OCR initialized with model: {self.model_name}")
    
    def extract_text_from_image_bytes(
        self,
        image_bytes: bytes,
        page_number: int,
        mime_type: str = "image/png"
    ) -> OCRResult:
        """
        Extract text from image bytes using Gemini Vision with retry logic
        
        Args:
            image_bytes: Image data as bytes
            page_number: Page number for tracking
            mime_type: MIME type of the image
            
        Returns:
            OCRResult with extracted text and confidence
        """
        start_time = time.time()
        
        logger.debug(f"[Page {page_number}] Starting Gemini OCR")
        logger.debug(f"[Page {page_number}] Image: {len(image_bytes)} bytes, {mime_type}")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # IMPROVED OCR prompt - more specific
                prompt = """Extract ALL text from this answer sheet image:

**Instructions**:
1. Extract handwritten and printed text exactly as written
2. Preserve line breaks and spacing
3. Include mathematical equations/symbols
4. Include diagrams labels and annotations
5. If text is unclear, use [?] to mark it
6. Be thorough - don't skip any visible text

Extract the text now:"""
                
                # Create content
                contents = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                ]
                
                logger.debug(f"[Page {page_number}] Calling Gemini Vision (attempt {attempt}/{self.max_retries})...")
                
                # Call Gemini Vision
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Deterministic
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=4096,  # Sufficient for long text
                    )
                )
                
                extracted_text = response.text.strip()
                processing_time = (time.time() - start_time) * 1000
                
                logger.info(f"[Page {page_number}] OCR done. "
                           f"Time: {processing_time:.0f}ms, Length: {len(extracted_text)}")
                
                # Validate extracted text
                if not extracted_text or extracted_text == "":
                    logger.warning(f"[Page {page_number}] Empty OCR result")
                    if attempt < self.max_retries:
                        logger.info(f"[Page {page_number}] Retrying OCR...")
                        time.sleep(2)
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
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"[Page {page_number}] Retrying OCR in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # All retries failed
                    processing_time = (time.time() - start_time) * 1000
                    logger.error(f"[Page {page_number}] OCR failed after {attempt} attempts")
                    
                    return OCRResult(
                        page_number=page_number,
                        raw_text=f"[Gemini OCR Error: {str(e)}]",
                        confidence=0.0,
                        processing_time_ms=round(processing_time, 2)
                    )
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on text quality"""
        
        if not text or len(text) < 10:
            return 0.3
        
        if text.startswith('[') and 'Error' in text:
            return 0.0
        
        # Check for common OCR issues
        unclear_markers = text.count('[?]')
        total_chars = len(text)
        
        if unclear_markers > 0:
            unclear_ratio = unclear_markers / max(total_chars / 100, 1)
            confidence = max(0.5, 1.0 - unclear_ratio)
            return round(confidence, 2)
        
        # Estimate based on length and structure
        if len(text) > 500:
            return 0.95
        elif len(text) > 200:
            return 0.9
        elif len(text) > 100:
            return 0.8
        elif len(text) > 50:
            return 0.75
        else:
            return 0.6