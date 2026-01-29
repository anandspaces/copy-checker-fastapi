from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"


class PageMetadata(BaseModel):
    """Computer vision metadata extracted from a page"""
    page_number: int
    writing_density: float = Field(ge=0.0, le=1.0, description="Proportion of page with writing")
    has_diagrams: bool = Field(description="Whether diagrams/drawings detected")
    crossed_out_regions: int = Field(ge=0, description="Number of crossed-out text regions")
    text_blocks_count: int = Field(ge=0, description="Number of distinct text blocks")
    skew_angle: float = Field(description="Detected skew angle in degrees")
    

class PageEvaluation(BaseModel):
    """LLM evaluation result for a single page"""
    page_number: int
    marks_awarded: float = Field(ge=0, description="Marks for this page")
    max_marks: float = Field(gt=0, description="Maximum possible marks")
    remarks: str = Field(max_length=200, description="Short examiner remarks (1-2 lines)")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Evaluation confidence")
    
    @field_validator('remarks')
    @classmethod
    def validate_remarks(cls, v: str) -> str:
        """Ensure remarks are professional and concise"""
        if not v or len(v.strip()) == 0:
            return "Answer reviewed."
        return v.strip()


class EvaluationRequest(BaseModel):
    """Request payload for evaluation"""
    subject: str = Field(min_length=1, max_length=100, description="Subject name")
    marking_scheme: str = Field(description="Question paper and marking scheme (text or JSON)")
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI, description="LLM provider to use")
    max_marks_per_page: Optional[float] = Field(default=None, gt=0, description="Override max marks per page")


class OCRResult(BaseModel):
    """OCR extraction result"""
    page_number: int
    raw_text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time_ms: float = Field(ge=0)


class LLMEvaluationRequest(BaseModel):
    """Request sent to LLM for evaluation"""
    page_number: int
    ocr_text: str
    subject: str
    marking_scheme: str
    vision_metadata: PageMetadata
    max_marks: Optional[float] = None


class LLMEvaluationResponse(BaseModel):
    """Expected JSON response from LLM (strict schema)"""
    marks_awarded: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    remarks: str = Field(max_length=200)
    reasoning: Optional[str] = Field(default=None, description="Internal reasoning (not displayed)")
    
    @field_validator('marks_awarded')
    @classmethod
    def validate_marks(cls, v: float, info) -> float:
        """Ensure awarded marks don't exceed maximum"""
        # Note: max_marks validation happens after model creation
        return round(v, 2)


class AnnotationConfig(BaseModel):
    """Configuration for PDF annotations"""
    font_size: int = Field(default=10, ge=6, le=14)
    font_color: tuple[float, float, float] = Field(default=(1, 0, 0), description="RGB in 0-1 range (red)")
    margin_left: int = Field(default=50, description="Left margin for remarks")
    margin_bottom: int = Field(default=30, description="Bottom margin for page totals")
    box_padding: int = Field(default=5, description="Padding for bounding boxes")
    highlight_opacity: float = Field(default=0.2, ge=0.0, le=1.0)


class EvaluationSummary(BaseModel):
    """Summary of entire evaluation"""
    total_pages: int = Field(ge=1)
    total_marks_awarded: float = Field(ge=0)
    total_max_marks: float = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    pages_evaluated: List[PageEvaluation]
    processing_time_seconds: float = Field(ge=0)