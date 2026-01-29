import os
import json
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.schemas import (
    LLMEvaluationRequest,
    LLMEvaluationResponse,
    LLMProvider
)


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
    """Google Gemini LLM client"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for fast, accurate evaluation
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configure for JSON output
        self.generation_config = {
            "temperature": 0.3,  # Lower temperature for consistency
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 500,
        }
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate page using Gemini"""
        prompt = self._build_evaluation_prompt(request)
        
        try:
            # First attempt
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return self._parse_response(response.text, request)
            
        except Exception as e:
            # Retry once on failure
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
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


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM client"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize OpenAI client"""
        if OpenAI is None:
            raise ImportError("openai not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate page using OpenAI"""
        prompt = self._build_evaluation_prompt(request)
        
        try:
            # First attempt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert examiner. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}  # Force JSON mode
            )
            
            return self._parse_response(response.choices[0].message.content, request)
            
        except Exception as e:
            # Retry once on failure
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert examiner. Respond ONLY with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                return self._parse_response(response.choices[0].message.content, request)
            except Exception as retry_error:
                return LLMEvaluationResponse(
                    marks_awarded=0.0,
                    max_marks=request.max_marks or 10.0,
                    remarks=f"Evaluation failed: {str(retry_error)[:100]}"
                )
    
    def _parse_response(self, response_text: str, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Parse LLM response into structured format"""
        try:
            data = json.loads(response_text)
            response = LLMEvaluationResponse(**data)
            
            # Ensure marks don't exceed maximum
            if request.max_marks and response.marks_awarded > response.max_marks:
                response.marks_awarded = response.max_marks
            
            return response
            
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")


class LLMService:
    """Service for managing LLM evaluation"""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.GEMINI,
        api_key: Optional[str] = None
    ):
        """Initialize LLM service with specified provider"""
        self.provider = provider
        
        if provider == LLMProvider.GEMINI:
            self.client = GeminiLLMClient(api_key)
        elif provider == LLMProvider.OPENAI:
            self.client = OpenAILLMClient(api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def evaluate_page(self, request: LLMEvaluationRequest) -> LLMEvaluationResponse:
        """Evaluate a single page"""
        return self.client.evaluate_page(request)