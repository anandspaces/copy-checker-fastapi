# src/services/evaluation_service.py

from abc import ABC, abstractmethod
import logging
import time
import json
from typing import List, Optional
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from src.schemas import (
    QuestionData,
    EvaluationSummary,
    QuestionEvaluationRequest
)

load_dotenv()
logger = logging.getLogger(__name__)


class EvaluationService(ABC):
    """Abstract base class for answer evaluation services"""
    
    @abstractmethod
    def evaluate_questions(
        self,
        questions: List[QuestionData],
        subject: str,
        marking_scheme: Optional[str],
        total_marks: Optional[float],
        strict_marking: bool,
        include_partial_credit: bool
    ) -> EvaluationSummary:
        """
        Evaluate all questions
        
        Args:
            questions: List of questions with student answers
            subject: Subject name
            marking_scheme: Marking criteria (optional)
            total_marks: Total marks for paper (optional)
            strict_marking: Apply strict grading
            include_partial_credit: Award partial marks
            
        Returns:
            EvaluationSummary with results
        """
        pass


class GeminiEvaluationService(EvaluationService):
    """Evaluation service using Google Gemini LLM - Question-wise grading"""
    
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
    
    def evaluate_questions(
        self,
        questions: List[QuestionData],
        subject: str,
        marking_scheme: Optional[str],
        total_marks: Optional[float],
        strict_marking: bool = False,
        include_partial_credit: bool = True
    ) -> EvaluationSummary:
        """
        Evaluate all questions using Gemini
        
        Args:
            questions: List of QuestionData with student answers
            subject: Subject name
            marking_scheme: Marking scheme (optional)
            total_marks: Total marks (optional)
            strict_marking: Strict grading mode
            include_partial_credit: Award partial marks
            
        Returns:
            EvaluationSummary with evaluated questions
        """
        logger.info(f"Starting evaluation of {len(questions)} questions")
        logger.info(f"Subject: {subject}, Strict: {strict_marking}, Partial: {include_partial_credit}")
        
        start_time = time.time()
        
        evaluated_questions = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: Q{question.question_number}")
            
            request = QuestionEvaluationRequest(
                question=question,
                subject=subject,
                marking_scheme=marking_scheme,
                strict_marking=strict_marking,
                include_partial_credit=include_partial_credit
            )
            
            evaluated_q = self._evaluate_single_question(request)
            evaluated_questions.append(evaluated_q)
            
            logger.info(f"Q{question.question_number}: "
                       f"{evaluated_q.marks_awarded}/{evaluated_q.allocated_marks or 'N/A'} marks")
        
        # Calculate summary
        total_awarded = sum(q.marks_awarded or 0 for q in evaluated_questions)
        total_max = sum(q.allocated_marks or 0 for q in evaluated_questions)
        
        # Use provided total_marks if allocated marks not found
        if total_max == 0 and total_marks:
            total_max = total_marks
        
        percentage = (total_awarded / total_max * 100) if total_max > 0 else 0
        grade = self._calculate_grade(percentage)
        
        overall_remarks = self._generate_overall_remarks(evaluated_questions, percentage)
        
        processing_time = time.time() - start_time
        
        summary = EvaluationSummary(
            total_questions=len(evaluated_questions),
            total_marks_awarded=round(total_awarded, 2),
            total_max_marks=round(total_max, 2),
            percentage=round(percentage, 2),
            grade=grade,
            questions_evaluated=evaluated_questions,
            overall_remarks=overall_remarks,
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"Evaluation complete: {summary.total_marks_awarded}/{summary.total_max_marks} "
                   f"({summary.percentage}%) - Grade: {grade}")
        
        return summary
    
    def _evaluate_single_question(
        self, 
        request: QuestionEvaluationRequest
    ) -> QuestionData:
        """
        Evaluate a single question using Gemini
        
        Args:
            request: Question evaluation request
            
        Returns:
            QuestionData with evaluation results filled
        """
        question = request.question
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Build evaluation prompt
                prompt = self._build_evaluation_prompt(request)
                
                # Call Gemini
                logger.debug(f"Calling Gemini for Q{question.question_number} (attempt {attempt})")
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        max_output_tokens=2048
                    )
                )
                
                # Parse response
                result = json.loads(response.text.strip())
                
                # Update question with evaluation
                question.marks_awarded = round(float(result.get("marks_awarded", 0)), 2)
                question.remarks = result.get("remarks", "Evaluated")
                question.strengths = result.get("strengths", [])
                question.improvements = result.get("improvements", [])
                question.is_correct = result.get("is_correct", False)
                
                # Validate marks
                if question.allocated_marks:
                    question.marks_awarded = min(question.marks_awarded, question.allocated_marks)
                
                logger.debug(f"Q{question.question_number} evaluated successfully")
                return question
                
            except Exception as e:
                logger.error(f"Q{question.question_number} attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    # Fallback evaluation
                    logger.error(f"All attempts failed for Q{question.question_number}")
                    question.marks_awarded = 0.0
                    question.remarks = "Evaluation failed - requires manual review"
                    question.improvements = ["Manual review required"]
                    return question
    
    def _build_evaluation_prompt(self, request: QuestionEvaluationRequest) -> str:
        """Build evaluation prompt for Gemini"""
        
        question = request.question
        
        marking_mode = "strict" if request.strict_marking else "fair"
        partial_credit = "with partial credit" if request.include_partial_credit else "without partial credit"
        
        # Handle allocated marks
        max_marks_text = f"{question.allocated_marks}" if question.allocated_marks else "appropriate marks"
        
        # Marking scheme info
        scheme_text = f"\n**Marking Scheme:**\n{request.marking_scheme}\n" if request.marking_scheme else ""
        
        prompt = f"""You are an experienced examiner grading a **{request.subject}** answer sheet.

**Question {question.question_number}:**
{question.question_text}

**Allocated Marks:** {max_marks_text}
{scheme_text}
**Student's Answer:**
{question.student_answer[:2000]}

**Evaluation Guidelines:**
- Marking mode: {marking_mode}
- Partial credit: {partial_credit}
- Evaluate based on: correctness, completeness, clarity, understanding
- If answer is blank/completely wrong → 0 marks
- If answer shows partial understanding → proportional marks (if partial credit allowed)
- If answer is mostly/fully correct → near full/full marks

**Required JSON Response:**
{{
  "marks_awarded": <number between 0 and {max_marks_text}>,
  "remarks": "<concise feedback, 50-200 characters>",
  "strengths": ["<what was done well>", "..."],
  "improvements": ["<what could be improved>", "..."],
  "is_correct": <true/false>,
  "reasoning": "<why these marks were awarded>"
}}

Rules:
- marks_awarded must be ≤ allocated marks
- remarks must be constructive and specific
- strengths/improvements: 1-3 items each
- Return ONLY valid JSON

Evaluate now:"""
        
        return prompt
    
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
        questions: List[QuestionData],
        percentage: float
    ) -> str:
        """Generate overall feedback"""
        
        correct_count = sum(1 for q in questions if q.is_correct)
        total_questions = len(questions)
        
        if percentage >= 80:
            performance = f"Excellent work! {correct_count}/{total_questions} questions answered well"
        elif percentage >= 60:
            performance = f"Good performance. {correct_count}/{total_questions} questions correct"
        elif percentage >= 40:
            performance = f"Satisfactory effort. {correct_count}/{total_questions} questions correct"
        else:
            performance = f"Needs improvement. {correct_count}/{total_questions} questions correct"
        
        return performance