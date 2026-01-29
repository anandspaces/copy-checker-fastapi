import os
import json
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas import (
    LLMEvaluationRequest,
    LLMEvaluationResponse
)

# Load environment variables
load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = 'gemini-3-flash-preview'  # Using latest Gemini model


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate a page and return structured response"""
        pass
    
    def _build_evaluation_prompt(self, request: LLMEvaluationRequest) -> str:
        """Build evaluation prompt for LLM"""
        prompt = f"""You are an expert examiner evaluating a student's answer sheet for {request.subject}.

MARKING SCHEME:
{request.marking_scheme}

PAGE METADATA (from computer vision):
- Writing density: {request.vision_metadata.writing_density:.1%}
- Has diagrams: {request.vision_metadata.has_diagrams}
- Crossed-out regions: {request.vision_metadata.crossed_out_regions}
- Text blocks: {request.vision_metadata.text_blocks_count}

STUDENT'S ANSWER (Page {request.page_number}, OCR-extracted):
{request.ocr_text}

EVALUATION INSTRUCTIONS:
1. Award marks based on the marking scheme and the student's answer
2. Be tolerant of OCR errors (spelling mistakes, garbled text)
3. Do NOT hallucinate answers if the OCR text is empty or unclear
4. Award partial marks for partially correct answers
5. Provide short, professional examiner remarks (1-2 lines max)
6. If OCR text is mostly noise/unreadable, award 0 marks and note this

RESPOND ONLY WITH VALID JSON in this EXACT format:
{{
  "marks_awarded": <number>,
  "max_marks": <number>,
  "remarks": "<short examiner comment>",
  "reasoning": "<internal reasoning, optional>"
}}

Do not include any text before or after the JSON."""
        
        return prompt


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM client using NEW google-genai SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with NEW SDK"""
        if genai is None:
            raise ImportError(
                "google-genai not installed. "
                "Run: pip install google-genai"
            )
        
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        # ✅ Create client using NEW SDK pattern
        self.client = genai.Client(api_key=self.api_key)
        
        # ✅ Configure generation settings using NEW SDK types
        self.generation_config = types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_output_tokens=500,
        )
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate page using Gemini with NEW SDK"""
        prompt = self._build_evaluation_prompt(request)
        
        try:
            # First attempt using NEW SDK pattern
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=self.generation_config
            )
            
            # Extract text from response
            response_text = response.text
            
            return self._parse_response(response_text, request)
            
        except Exception as e:
            # Retry once on failure
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=self.generation_config
                )
                return self._parse_response(response.text, request)
            except Exception as retry_error:
                # Return zero marks on total failure
                return LLMEvaluationResponse(
                    marks_awarded=0.0,
                    max_marks=request.max_marks or 10.0,
                    remarks=f"Evaluation failed: {str(retry_error)[:100]}"
                )
    
    def _parse_response(self, response_text: str, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_str = response_text.strip()
            
            # Remove markdown code blocks if present
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                json_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_str
            
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate and create response
            response = LLMEvaluationResponse(**data)
            
            # Ensure marks don't exceed maximum
            if request.max_marks and response.marks_awarded > response.max_marks:
                response.marks_awarded = response.max_marks
            
            return response
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def __del__(self):
        """Cleanup: Close the client when object is destroyed"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass


class LLMService:
    """Service for managing LLM evaluation (Gemini only)"""
    
    def __init__(self, api_key: Optional[str] = None, provider: LLMProvider = LLMProvider.GEMINI):
        """Initialize LLM service with Gemini"""
        if provider != LLMProvider.GEMINI:
            raise ValueError("Only Gemini provider is supported")
        
        self.provider = provider
        self.client = GeminiLLMClient(api_key)
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate a single page"""
        return self.client.evaluate_page(request)