# src/services/pdf_annotation_service.py

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict

from src.schemas import QuestionData, AnnotationConfig, EvaluationSummary

logger = logging.getLogger(__name__)


class PDFAnnotationService:
    """Service for annotating PDFs with question-wise evaluation results"""
    
    def __init__(self, config: AnnotationConfig = None):
        """
        Initialize PDF annotation service
        
        Args:
            config: Annotation configuration
        """
        logger.info("Initializing PDFAnnotationService (Question-wise)")
        self.config = config or AnnotationConfig()
    
    def annotate_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        questions: List[QuestionData],
        summary: EvaluationSummary
    ) -> Path:
        """
        Annotate PDF with question-wise evaluation results
        
        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path for annotated output
            questions: List of evaluated questions
            summary: Overall evaluation summary
            
        Returns:
            Path to annotated PDF
        """
        logger.info(f"Annotating PDF: {input_pdf_path}")
        logger.info(f"Total questions to annotate: {len(questions)}")
        
        doc = None
        try:
            doc = fitz.open(input_pdf_path)
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Group questions by page
            page_questions_map = self._group_questions_by_page(questions)
            
            # Annotate each page with its questions
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_questions = page_questions_map.get(page_num + 1, [])
                
                if page_questions:
                    try:
                        self._annotate_page_with_questions(page, page_questions, page_num + 1)
                        logger.debug(f"Page {page_num + 1}: annotated {len(page_questions)} questions")
                    except Exception as e:
                        logger.error(f"Failed to annotate page {page_num + 1}: {e}")
            
            # Add summary to last page
            if self.config.show_summary and len(doc) > 0:
                try:
                    self._add_summary_page(doc[-1], summary)
                    logger.debug("Summary added to last page")
                except Exception as e:
                    logger.error(f"Failed to add summary: {e}")
            
            # Save
            doc.save(str(output_pdf_path), garbage=4, deflate=True, clean=True)
            logger.info(f"Annotated PDF saved: {output_pdf_path}")
            
            return output_pdf_path
            
        except Exception as e:
            logger.error(f"PDF annotation failed: {e}", exc_info=True)
            raise Exception(f"Failed to annotate PDF: {str(e)}")
        
        finally:
            if doc:
                doc.close()
    
    def _group_questions_by_page(
        self, 
        questions: List[QuestionData]
    ) -> Dict[int, List[QuestionData]]:
        """
        Group questions by their page numbers
        
        Args:
            questions: List of questions
            
        Returns:
            Dict mapping page_number -> list of questions on that page
        """
        page_map = {}
        
        for question in questions:
            # A question can span multiple pages
            for page_num in question.page_numbers:
                if page_num not in page_map:
                    page_map[page_num] = []
                page_map[page_num].append(question)
        
        return page_map
    
    def _annotate_page_with_questions(
        self,
        page: fitz.Page,
        questions: List[QuestionData],
        page_num: int
    ) -> None:
        """
        Annotate a page with question marks and remarks
        
        Args:
            page: PyMuPDF page object
            questions: Questions appearing on this page
            page_num: Page number
        """
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Validate page dimensions
        if page_width < 100 or page_height < 100:
            logger.warning(f"Page {page_num} too small for annotation")
            return
        
        # Starting position for annotations (top-right)
        y_offset = 15
        spacing = 80  # Space between question annotations
        
        for i, question in enumerate(questions):
            # Determine color based on correctness
            color = self._get_color_for_question(question)
            
            # Add marks box
            if self.config.show_marks and question.marks_awarded is not None:
                self._add_question_marks_box(
                    page, 
                    question, 
                    page_width, 
                    y_offset + (i * spacing),
                    color
                )
            
            # Add remarks
            if self.config.show_remarks and question.remarks:
                self._add_question_remarks(
                    page,
                    question,
                    page_width,
                    y_offset + (i * spacing) + 55,
                    color
                )
    
    def _get_color_for_question(self, question: QuestionData) -> tuple:
        """
        Get color based on question performance
        
        Args:
            question: QuestionData
            
        Returns:
            RGB tuple (0-1 range)
        """
        if not question.allocated_marks or not question.marks_awarded:
            return self.config.incorrect_color
        
        percentage = (question.marks_awarded / question.allocated_marks) * 100
        
        if percentage >= 80:
            return self.config.correct_color  # Green
        elif percentage >= 40:
            return self.config.partial_color  # Orange
        else:
            return self.config.incorrect_color  # Red
    
    def _add_question_marks_box(
        self,
        page: fitz.Page,
        question: QuestionData,
        page_width: float,
        y_pos: float,
        color: tuple
    ) -> None:
        """Add marks box for a question"""
        
        box_width = 140
        box_height = 45
        margin = 15
        
        x = page_width - box_width - margin
        y = y_pos
        
        # Draw box
        box_rect = fitz.Rect(x, y, x + box_width, y + box_height)
        
        # Light background based on color
        bg_color = tuple(min(c + 0.3, 1.0) for c in color)
        
        page.draw_rect(
            box_rect,
            color=color,
            fill=bg_color,
            width=2
        )
        
        # Text content
        marks_text = f"Q{question.question_number}: {question.marks_awarded}"
        if question.allocated_marks:
            marks_text += f"/{question.allocated_marks}"
            percentage = (question.marks_awarded / question.allocated_marks * 100)
            marks_text += f"\n({percentage:.0f}%)"
        
        # Add text
        text_rect = fitz.Rect(x + 5, y + 5, x + box_width - 5, y + box_height - 5)
        
        page.insert_textbox(
            text_rect,
            marks_text,
            fontsize=10,
            fontname="hebo",
            color=color,
            align=fitz.TEXT_ALIGN_CENTER
        )
    
    def _add_question_remarks(
        self,
        page: fitz.Page,
        question: QuestionData,
        page_width: float,
        y_pos: float,
        color: tuple
    ) -> None:
        """Add remarks for a question"""
        
        box_width = 200
        margin = 15
        
        x = page_width - box_width - margin
        y = y_pos
        
        # Prepare remark
        remark = f"Q{question.question_number}: {question.remarks[:120]}"
        
        # Calculate height
        font_size = self.config.font_size
        line_height = font_size + 3
        lines = self._wrap_text(remark, box_width - 10, font_size)
        text_height = len(lines) * line_height + 8
        
        # Draw background
        bg_color = tuple(min(c + 0.4, 1.0) for c in color)
        bg_rect = fitz.Rect(x - 5, y - 5, x + box_width + 5, y + text_height + 5)
        
        page.draw_rect(
            bg_rect,
            color=color,
            fill=bg_color,
            width=1
        )
        
        # Add text
        text_rect = fitz.Rect(x, y, x + box_width, y + text_height)
        
        page.insert_textbox(
            text_rect,
            '\n'.join(lines),
            fontsize=font_size,
            fontname="helv",
            color=color,
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _add_summary_page(self, page: fitz.Page, summary: EvaluationSummary) -> None:
        """Add evaluation summary to last page"""
        
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Position: bottom center
        box_width = 320
        box_height = 160
        x = (page_width - box_width) / 2
        y = page_height - box_height - 30
        
        # Draw box
        box_rect = fitz.Rect(x - 10, y - 10, x + box_width + 10, y + box_height + 10)
        
        page.draw_rect(
            box_rect,
            color=(0, 0, 0.8),
            fill=(0.95, 0.95, 1),
            width=2.5
        )
        
        # Summary text
        summary_text = f"""╔══════════════════════════════════╗
      EVALUATION SUMMARY
╚══════════════════════════════════╝

Total Questions: {summary.total_questions}
Total Marks: {summary.total_marks_awarded} / {summary.total_max_marks}
Percentage: {summary.percentage:.1f}%
Grade: {summary.grade}

{summary.overall_remarks}

Processing Time: {summary.processing_time_seconds:.1f}s"""
        
        # Add text
        text_rect = fitz.Rect(x, y, x + box_width - 20, y + box_height - 10)
        
        page.insert_textbox(
            text_rect,
            summary_text,
            fontsize=self.config.font_size,
            fontname="hebo",
            color=(0, 0, 0.6),
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _wrap_text(self, text: str, max_width: float, font_size: int) -> List[str]:
        """Wrap text to fit within width"""
        words = text.split()
        lines = []
        current_line = []
        
        char_width = font_size * 0.5
        max_chars = int(max_width / char_width)
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines[:4]  # Limit lines