# src/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"


class OCRResult(BaseModel):
    """OCR extraction result for a single page"""
    page_number: int = Field(description="Page number (1-indexed)")
    raw_text: str = Field(description="Extracted text from page")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="OCR confidence score")
    processing_time_ms: float = Field(ge=0, description="Time taken for OCR in milliseconds")
    metadata: dict = Field(default_factory=dict, description="Additional OCR metadata")


class DocumentOCRResult(BaseModel):
    """Complete OCR results for entire document"""
    total_pages: int = Field(ge=1, description="Total number of pages")
    pages: List[OCRResult] = Field(description="OCR results for each page")
    total_processing_time_ms: float = Field(ge=0, description="Total OCR processing time")
    ocr_provider: str = Field(description="OCR provider used")
    timestamp: datetime = Field(default_factory=datetime.now, description="When OCR was performed")


class PageEvaluation(BaseModel):
    """Evaluation result for a single page"""
    page_number: int = Field(description="Page number (1-indexed)")
    marks_awarded: float = Field(ge=0, description="Marks awarded for this page")
    max_marks: float = Field(gt=0, description="Maximum possible marks for this page")
    remarks: str = Field(max_length=500, description="Examiner remarks")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Evaluation confidence")
    strengths: List[str] = Field(default_factory=list, description="Positive aspects")
    improvements: List[str] = Field(default_factory=list, description="Areas for improvement")
    
    @field_validator('remarks')
    @classmethod
    def validate_remarks(cls, v: str) -> str:
        """Ensure remarks are professional and non-empty"""
        if not v or len(v.strip()) == 0:
            return "Answer reviewed."
        return v.strip()


class EvaluationSummary(BaseModel):
    """Summary of entire evaluation"""
    total_pages: int = Field(ge=1)
    total_marks_awarded: float = Field(ge=0)
    total_max_marks: float = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    grade: Optional[str] = Field(default=None, description="Letter grade based on percentage")
    pages_evaluated: List[PageEvaluation]
    overall_remarks: str = Field(default="", description="Overall feedback")
    processing_time_seconds: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationRequest(BaseModel):
    """Request for answer sheet evaluation"""
    subject: str = Field(min_length=1, max_length=100, description="Subject name")
    marking_scheme: str = Field(description="Detailed marking scheme or rubric")
    max_marks_per_page: float = Field(gt=0, description="Maximum marks per page")
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI)
    
    # Optional configurations
    strict_marking: bool = Field(default=False, description="Apply strict marking criteria")
    include_partial_credit: bool = Field(default=True, description="Award partial marks")
    
    @field_validator('marking_scheme')
    @classmethod
    def validate_marking_scheme(cls, v: str) -> str:
        """Ensure marking scheme is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Marking scheme cannot be empty")
        return v.strip()


class PageEvaluationRequest(BaseModel):
    """Request for evaluating a single page"""
    page_number: int = Field(description="Page number being evaluated")
    extracted_text: str = Field(description="OCR extracted text from page")
    subject: str = Field(description="Subject name")
    marking_scheme: str = Field(description="Marking criteria")
    max_marks: float = Field(gt=0, description="Maximum marks for this page")
    strict_marking: bool = Field(default=False)
    include_partial_credit: bool = Field(default=True)


class PageEvaluationResponse(BaseModel):
    """LLM response for page evaluation"""
    marks_awarded: float = Field(ge=0, description="Marks awarded")
    max_marks: float = Field(gt=0, description="Maximum marks")
    remarks: str = Field(description="Detailed feedback")
    strengths: List[str] = Field(default_factory=list, description="What was done well")
    improvements: List[str] = Field(default_factory=list, description="What could be improved")
    reasoning: Optional[str] = Field(default=None, description="Internal reasoning (for debugging)")


class AnnotationConfig(BaseModel):
    """Configuration for PDF annotations"""
    font_size: int = Field(default=10, ge=6, le=14)
    font_color: tuple[float, float, float] = Field(default=(1, 0, 0), description="RGB in 0-1 range")
    show_remarks: bool = Field(default=True, description="Show remarks on pages")
    show_marks: bool = Field(default=True, description="Show marks on pages")
    show_summary: bool = Field(default=True, description="Show summary on last page")


class ProcessingStatus(str, Enum):
    """Processing status for async operations"""
    PENDING = "pending"
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETE = "ocr_complete"
    EVALUATION_IN_PROGRESS = "evaluation_in_progress"
    EVALUATION_COMPLETE = "evaluation_complete"
    ANNOTATION_IN_PROGRESS = "annotation_in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class EvaluationJob(BaseModel):
    """Complete evaluation job with all stages"""
    job_id: str = Field(description="Unique job identifier")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Input
    input_filename: str
    request: EvaluationRequest
    
    # OCR Phase
    ocr_result: Optional[DocumentOCRResult] = None
    
    # Evaluation Phase
    evaluation_summary: Optional[EvaluationSummary] = None
    
    # Output
    annotated_pdf_path: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None