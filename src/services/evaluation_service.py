# src/services/evaluation_service.py

from abc import ABC, abstractmethod
import logging
import time
import json
from typing import List
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from src.schemas import (
    PageEvaluationRequest,
    PageEvaluationResponse,
    PageEvaluation,
    EvaluationSummary
)

load_dotenv()
logger = logging.getLogger(__name__)


class EvaluationService(ABC):
    """Abstract base class for answer evaluation services"""
    
    @abstractmethod
    def evaluate_page(self, request: PageEvaluationRequest) -> PageEvaluation:
        """
        Evaluate a single page
        
        Args:
            request: Page evaluation request with extracted text
            
        Returns:
            PageEvaluation with marks and feedback
        """
        pass
    
    @abstractmethod
    def evaluate_document(
        self,
        extracted_texts: List[str],
        subject: str,
        marking_scheme: str,
        max_marks_per_page: float,
        strict_marking: bool = False,
        include_partial_credit: bool = True
    ) -> EvaluationSummary:
        """
        Evaluate entire document page by page
        
        Args:
            extracted_texts: List of extracted text for each page
            subject: Subject name
            marking_scheme: Marking criteria
            max_marks_per_page: Maximum marks per page
            strict_marking: Apply strict grading
            include_partial_credit: Award partial marks
            
        Returns:
            EvaluationSummary with all page evaluations
        """
        pass


class GeminiEvaluationService(EvaluationService):
    """Evaluation service using Google Gemini LLM"""
    
    def __init__(self, model_name: str = "gemini-3-flash-preview", max_retries: int = 3):
        """
        Initialize Gemini evaluation service
        
        Args:
            model_name: Gemini model to use
            max_retries: Maximum retry attempts
        """
        logger.info(f"Initializing GeminiEvaluationService with model: {model_name}")
        
        self.model_name = model_name
        self.max_retries = max_retries
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("GeminiEvaluationService initialized successfully")
    
    def evaluate_document(
        self,
        extracted_texts: List[str],
        subject: str,
        marking_scheme: str,
        max_marks_per_page: float,
        strict_marking: bool = False,
        include_partial_credit: bool = True
    ) -> EvaluationSummary:
        """
        Evaluate entire document page by page
        
        Args:
            extracted_texts: List of OCR texts for each page
            subject: Subject name
            marking_scheme: Marking criteria
            max_marks_per_page: Maximum marks per page
            strict_marking: Apply strict grading
            include_partial_credit: Award partial marks
            
        Returns:
            Complete evaluation summary
        """
        logger.info(f"Starting document evaluation. Pages: {len(extracted_texts)}")
        start_time = time.time()
        
        page_evaluations: List[PageEvaluation] = []
        
        for page_num, extracted_text in enumerate(extracted_texts, start=1):
            logger.info(f"Evaluating page {page_num}/{len(extracted_texts)}")
            
            request = PageEvaluationRequest(
                page_number=page_num,
                extracted_text=extracted_text,
                subject=subject,
                marking_scheme=marking_scheme,
                max_marks=max_marks_per_page,
                strict_marking=strict_marking,
                include_partial_credit=include_partial_credit
            )
            
            page_eval = self.evaluate_page(request)
            page_evaluations.append(page_eval)
            
            logger.info(f"Page {page_num} evaluated: "
                       f"{page_eval.marks_awarded}/{page_eval.max_marks} "
                       f"({page_eval.marks_awarded/page_eval.max_marks*100:.1f}%)")
        
        # Calculate summary
        total_marks_awarded = sum(e.marks_awarded for e in page_evaluations)
        total_max_marks = sum(e.max_marks for e in page_evaluations)
        percentage = (total_marks_awarded / total_max_marks * 100) if total_max_marks > 0 else 0.0
        
        # Determine grade
        grade = self._calculate_grade(percentage)
        
        # Generate overall remarks
        overall_remarks = self._generate_overall_remarks(page_evaluations, percentage)
        
        processing_time = time.time() - start_time
        
        summary = EvaluationSummary(
            total_pages=len(extracted_texts),
            total_marks_awarded=round(total_marks_awarded, 2),
            total_max_marks=round(total_max_marks, 2),
            percentage=round(percentage, 2),
            grade=grade,
            pages_evaluated=page_evaluations,
            overall_remarks=overall_remarks,
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"Document evaluation complete: "
                   f"{summary.total_marks_awarded}/{summary.total_max_marks} "
                   f"({summary.percentage}%) - Grade: {grade}")
        
        return summary
    
    def evaluate_page(self, request: PageEvaluationRequest) -> PageEvaluation:
        """
        Evaluate a single page using Gemini LLM
        
        Args:
            request: Page evaluation request
            
        Returns:
            PageEvaluation with marks and feedback
        """
        logger.debug(f"[Page {request.page_number}] Starting evaluation")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Build evaluation prompt
                prompt = self._build_evaluation_prompt(request)
                
                # Call Gemini
                logger.debug(f"[Page {request.page_number}] Calling Gemini (attempt {attempt})")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistent grading
                        top_p=0.95,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                    )
                )
                
                # Parse response
                llm_response = self._parse_llm_response(response, request)
                
                # Create page evaluation
                page_eval = PageEvaluation(
                    page_number=request.page_number,
                    marks_awarded=llm_response.marks_awarded,
                    max_marks=llm_response.max_marks,
                    remarks=llm_response.remarks,
                    confidence=0.85,  # Default confidence for LLM evaluation
                    strengths=llm_response.strengths,
                    improvements=llm_response.improvements
                )
                
                logger.debug(f"[Page {request.page_number}] Evaluation successful")
                return page_eval
                
            except Exception as e:
                logger.error(f"[Page {request.page_number}] Attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"[Page {request.page_number}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Return fallback evaluation
                    logger.error(f"[Page {request.page_number}] All attempts failed")
                    return self._create_fallback_evaluation(request, str(e))
    
    def _build_evaluation_prompt(self, request: PageEvaluationRequest) -> str:
        """Build evaluation prompt for Gemini"""
        
        # Truncate text if too long
        max_text_length = 3000
        text = request.extracted_text
        if len(text) > max_text_length:
            text = text[:max_text_length] + "...[truncated]"
        
        # Handle empty text
        if not text or text.strip() == "":
            text = "[No text extracted - blank page or OCR failure]"
        
        marking_mode = "strict" if request.strict_marking else "fair"
        partial_credit = "with partial credit" if request.include_partial_credit else "without partial credit"
        
        prompt = f"""You are an experienced examiner evaluating a student's answer sheet.

**Subject:** {request.subject}

**Marking Scheme:**
{request.marking_scheme}

**Student's Answer (Page {request.page_number}):**
{text}

**Evaluation Guidelines:**
- Maximum marks for this page: {request.max_marks}
- Marking mode: {marking_mode}
- Partial credit: {partial_credit}
- Evaluate based on: correctness, completeness, clarity, and understanding
- If the answer is blank or completely wrong, award 0 marks
- If the answer shows partial understanding, award proportional marks (if partial credit allowed)
- Be fair and consistent in your evaluation

**Required Response Format (JSON only):**
{{
  "marks_awarded": <number between 0 and {request.max_marks}>,
  "max_marks": {request.max_marks},
  "remarks": "<concise feedback, 100-300 characters>",
  "strengths": ["<what was done well>", "..."],
  "improvements": ["<what could be improved>", "..."],
  "reasoning": "<internal reasoning for marks awarded>"
}}

**Important:**
- Return ONLY valid JSON, no extra text
- marks_awarded must be between 0 and {request.max_marks}
- remarks should be constructive and specific
- strengths and improvements should each have 1-3 items

Evaluate now:"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response,
        request: PageEvaluationRequest
    ) -> PageEvaluationResponse:
        """Parse and validate LLM response"""
        
        try:
            response_text = response.text.strip()
            
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx+1]
            else:
                json_text = response_text
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Extract and validate fields
            marks_awarded = float(data.get("marks_awarded", 0))
            max_marks = float(data.get("max_marks", request.max_marks))
            remarks = str(data.get("remarks", "Answer evaluated.")).strip()
            strengths = data.get("strengths", [])
            improvements = data.get("improvements", [])
            reasoning = data.get("reasoning", "")
            
            # Validate marks
            marks_awarded = max(0, min(marks_awarded, max_marks))
            
            # Validate remarks
            if not remarks:
                remarks = "Answer reviewed."
            if len(remarks) > 500:
                remarks = remarks[:497] + "..."
            
            # Validate lists
            if not isinstance(strengths, list):
                strengths = []
            if not isinstance(improvements, list):
                improvements = []
            
            strengths = [str(s)[:200] for s in strengths[:5]]
            improvements = [str(i)[:200] for i in improvements[:5]]
            
            return PageEvaluationResponse(
                marks_awarded=round(marks_awarded, 2),
                max_marks=round(max_marks, 2),
                remarks=remarks,
                strengths=strengths,
                improvements=improvements,
                reasoning=reasoning[:500] if reasoning else None
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"[Page {request.page_number}] JSON parse error: {e}")
            return self._extract_from_text_fallback(response.text, request)
        
        except Exception as e:
            logger.error(f"[Page {request.page_number}] Parse error: {e}")
            raise
    
    def _extract_from_text_fallback(
        self,
        text: str,
        request: PageEvaluationRequest
    ) -> PageEvaluationResponse:
        """Fallback parser when JSON parsing fails"""
        
        import re
        
        logger.warning(f"[Page {request.page_number}] Using fallback parser")
        
        # Try to extract marks
        marks = request.max_marks * 0.5  # Default to 50%
        
        patterns = [
            r'"?marks[_\s]*awarded"?\s*:\s*(\d+\.?\d*)',
            r'marks\s*:\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:out of|/)\s*\d+',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    marks = float(match.group(1))
                    break
                except:
                    continue
        
        marks = max(0, min(marks, request.max_marks))
        
        # Try to extract remarks
        remarks = "Answer evaluated (automated fallback)."
        remarks_match = re.search(r'"?remarks"?\s*:\s*"([^"]+)"', text)
        if remarks_match:
            remarks = remarks_match.group(1)[:500]
        
        return PageEvaluationResponse(
            marks_awarded=round(marks, 2),
            max_marks=round(request.max_marks, 2),
            remarks=remarks,
            strengths=[],
            improvements=[],
            reasoning="Fallback parser used due to JSON parse error"
        )
    
    def _create_fallback_evaluation(
        self,
        request: PageEvaluationRequest,
        error: str
    ) -> PageEvaluation:
        """Create fallback evaluation on complete failure"""
        
        logger.warning(f"[Page {request.page_number}] Creating fallback evaluation")
        
        # Conservative fallback: 40% of max marks
        fallback_marks = request.max_marks * 0.4
        
        return PageEvaluation(
            page_number=request.page_number,
            marks_awarded=round(fallback_marks, 2),
            max_marks=round(request.max_marks, 2),
            remarks="Evaluation incomplete. Manual review recommended.",
            confidence=0.3,
            strengths=[],
            improvements=["Requires manual review due to evaluation error"]
        )
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage"""
        
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B+"
        elif percentage >= 60:
            return "B"
        elif percentage >= 50:
            return "C"
        elif percentage >= 40:
            return "D"
        else:
            return "F"
    
    def _generate_overall_remarks(
        self,
        page_evaluations: List[PageEvaluation],
        percentage: float
    ) -> str:
        """Generate overall feedback based on all pages"""
        
        avg_confidence = sum(e.confidence for e in page_evaluations) / len(page_evaluations)
        
        if percentage >= 80:
            performance = "Excellent work overall"
        elif percentage >= 60:
            performance = "Good performance"
        elif percentage >= 40:
            performance = "Satisfactory effort"
        else:
            performance = "Needs improvement"
        
        return f"{performance}. Average evaluation confidence: {avg_confidence*100:.0f}%"