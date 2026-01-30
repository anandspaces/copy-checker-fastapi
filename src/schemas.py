# src/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"


class PaperMetadata(BaseModel):
    """Metadata extracted from first page"""
    subject: Optional[str] = Field(default=None, description="Subject name from paper")
    total_marks: Optional[float] = Field(default=None, description="Total marks from paper")
    student_name: Optional[str] = Field(default=None, description="Student name if found")
    roll_number: Optional[str] = Field(default=None, description="Roll number if found")
    exam_date: Optional[str] = Field(default=None, description="Exam date if found")
    additional_info: Dict[str, str] = Field(default_factory=dict, description="Any other metadata found")


class QuestionData(BaseModel):
    """Data for a single question"""
    question_number: str = Field(description="Question number (e.g., '1', '2a', '3.i')")
    page_numbers: List[int] = Field(description="Pages where this question appears")
    question_text: str = Field(description="The actual question text")
    student_answer: str = Field(description="Student's written answer")
    allocated_marks: Optional[float] = Field(default=None, description="Marks allocated to this question")
    
    # Evaluation results (filled by LLM)
    marks_awarded: Optional[float] = Field(default=None, description="Marks awarded by evaluator")
    remarks: Optional[str] = Field(default=None, description="Evaluator's remarks")
    strengths: List[str] = Field(default_factory=list, description="Positive aspects")
    improvements: List[str] = Field(default_factory=list, description="Areas for improvement")
    is_correct: Optional[bool] = Field(default=None, description="Whether answer is correct")


class OCRResult(BaseModel):
    """OCR extraction result for entire paper"""
    metadata: PaperMetadata = Field(description="Paper metadata from first page")
    questions: List[QuestionData] = Field(description="All questions with answers")
    total_pages: int = Field(ge=1, description="Total number of pages")
    processing_time_ms: float = Field(ge=0, description="Time taken for OCR")
    ocr_provider: str = Field(description="OCR provider used")
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationSummary(BaseModel):
    """Summary of entire evaluation"""
    total_questions: int = Field(ge=1, description="Total number of questions")
    total_marks_awarded: float = Field(ge=0, description="Total marks awarded")
    total_max_marks: float = Field(gt=0, description="Total maximum marks")
    percentage: float = Field(ge=0, le=100, description="Percentage score")
    grade: Optional[str] = Field(default=None, description="Letter grade")
    
    questions_evaluated: List[QuestionData] = Field(description="All evaluated questions")
    overall_remarks: str = Field(default="", description="Overall feedback")
    
    processing_time_seconds: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationRequest(BaseModel):
    """Request for answer sheet evaluation"""
    # Optional - will be extracted from PDF if not provided
    subject: Optional[str] = Field(default=None, description="Subject name (auto-extracted if not provided)")
    total_marks: Optional[float] = Field(default=None, description="Total marks (auto-extracted if not provided)")
    
    # Required marking scheme
    marking_scheme: Optional[str] = Field(
        default=None,
        description="Detailed marking scheme or rubric (optional - generic if not provided)"
    )
    
    # Optional configurations
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI)
    strict_marking: bool = Field(default=False, description="Apply strict marking criteria")
    include_partial_credit: bool = Field(default=True, description="Award partial marks")
    auto_extract_metadata: bool = Field(default=True, description="Extract subject/marks from PDF")


class QuestionEvaluationRequest(BaseModel):
    """Request for evaluating a single question"""
    question: QuestionData = Field(description="Question with student answer")
    subject: str = Field(description="Subject name")
    marking_scheme: Optional[str] = Field(default=None, description="Marking criteria")
    strict_marking: bool = Field(default=False)
    include_partial_credit: bool = Field(default=True)


class AnnotationConfig(BaseModel):
    """Configuration for PDF annotations"""
    font_size: int = Field(default=10, ge=6, le=14)
    show_remarks: bool = Field(default=True, description="Show remarks on pages")
    show_marks: bool = Field(default=True, description="Show marks on pages")
    show_summary: bool = Field(default=True, description="Show summary on last page")
    
    # Colors (RGB in 0-1 range)
    correct_color: tuple[float, float, float] = Field(default=(0, 0.6, 0), description="Green for correct")
    partial_color: tuple[float, float, float] = Field(default=(0.8, 0.5, 0), description="Orange for partial")
    incorrect_color: tuple[float, float, float] = Field(default=(0.8, 0, 0), description="Red for incorrect")


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
    ocr_result: Optional[OCRResult] = None
    
    # Evaluation Phase
    evaluation_summary: Optional[EvaluationSummary] = None
    
    # Output
    annotated_pdf_path: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None