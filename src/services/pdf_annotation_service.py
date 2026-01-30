# src/services/pdf_annotation_service.py

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List

from src.schemas import PageEvaluation, AnnotationConfig, EvaluationSummary

logger = logging.getLogger(__name__)


class PDFAnnotationService:
    """Service for annotating PDFs with evaluation results"""
    
    def __init__(self, config: AnnotationConfig = None):
        """
        Initialize PDF annotation service
        
        Args:
            config: Annotation configuration
        """
        logger.info("Initializing PDFAnnotationService")
        self.config = config or AnnotationConfig()
        logger.info(f"Configuration: font_size={self.config.font_size}, "
                   f"show_remarks={self.config.show_remarks}, "
                   f"show_marks={self.config.show_marks}")
    
    def annotate_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        evaluations: List[PageEvaluation],
        summary: EvaluationSummary
    ) -> Path:
        """
        Annotate PDF with evaluation results
        
        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path for annotated output
            evaluations: List of page evaluations
            summary: Overall evaluation summary
            
        Returns:
            Path to annotated PDF
        """
        logger.info(f"Annotating PDF: {input_pdf_path}")
        logger.info(f"Output path: {output_pdf_path}")
        
        doc = None
        try:
            # Open PDF
            doc = fitz.open(input_pdf_path)
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Create evaluation lookup
            eval_map = {eval.page_number: eval for eval in evaluations}
            
            # Annotate each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_eval = eval_map.get(page_num + 1)
                
                if page_eval:
                    try:
                        self._annotate_page(page, page_eval)
                        logger.debug(f"Page {page_num + 1} annotated successfully")
                    except Exception as e:
                        logger.error(f"Failed to annotate page {page_num + 1}: {e}")
                        continue
            
            # Add summary to last page
            if self.config.show_summary and len(doc) > 0:
                try:
                    self._add_summary_page(doc[-1], summary)
                    logger.debug("Summary added to last page")
                except Exception as e:
                    logger.error(f"Failed to add summary: {e}")
            
            # Save annotated PDF
            doc.save(str(output_pdf_path), garbage=4, deflate=True, clean=True)
            logger.info(f"Annotated PDF saved: {output_pdf_path}")
            
            return output_pdf_path
            
        except Exception as e:
            logger.error(f"PDF annotation failed: {e}", exc_info=True)
            raise Exception(f"Failed to annotate PDF: {str(e)}")
        
        finally:
            if doc:
                doc.close()
    
    def _annotate_page(self, page: fitz.Page, evaluation: PageEvaluation) -> None:
        """
        Annotate a single page with marks and remarks
        
        Args:
            page: PyMuPDF page object
            evaluation: Page evaluation data
        """
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Validate page dimensions
        if page_width < 100 or page_height < 100:
            logger.warning(f"Page {evaluation.page_number} too small for annotation")
            return
        
        # Add remarks at top-right
        if self.config.show_remarks and evaluation.remarks:
            self._add_remarks_box(page, evaluation, page_width, page_height)
        
        # Add marks at bottom-right
        if self.config.show_marks:
            self._add_marks_box(page, evaluation, page_width, page_height)
    
    def _add_remarks_box(
        self,
        page: fitz.Page,
        evaluation: PageEvaluation,
        page_width: float,
        page_height: float
    ) -> None:
        """Add remarks box to page"""
        
        # Position: top-right corner
        box_width = min(220, page_width * 0.35)
        x = page_width - box_width - 15
        y = 15
        
        # Prepare remark text
        remark = evaluation.remarks[:150]
        
        # Create text box dimensions
        font_size = self.config.font_size
        line_height = font_size + 4
        
        # Split text into lines
        lines = self._wrap_text(remark, box_width - 10, font_size)
        text_height = len(lines) * line_height + 10
        
        # Draw background
        bg_rect = fitz.Rect(
            x - 5,
            y - 5,
            min(x + box_width + 5, page_width),
            min(y + text_height + 5, page_height)
        )
        
        page.draw_rect(
            bg_rect,
            color=(0.8, 0.8, 0),  # Dark yellow border
            fill=(1, 1, 0.9),     # Light yellow fill
            width=1.5
        )
        
        # Add text
        text_rect = fitz.Rect(
            x,
            y,
            min(x + box_width, page_width - 10),
            min(y + text_height, page_height - 10)
        )
        
        text_content = '\n'.join(lines)
        page.insert_textbox(
            text_rect,
            text_content,
            fontsize=font_size,
            fontname="helv",
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _add_marks_box(
        self,
        page: fitz.Page,
        evaluation: PageEvaluation,
        page_width: float,
        page_height: float
    ) -> None:
        """Add marks box to page"""
        
        logger.debug(f"Adding marks box to page {evaluation.page_number}")
        logger.debug(f"  Marks: {evaluation.marks_awarded}/{evaluation.max_marks}")
            
        # Position: bottom-right corner with better margins
        box_width = 140
        box_height = 50  # Increased height
        margin = 20
            
        x = page_width - box_width - margin
        y = page_height - box_height - margin
            
        logger.debug(f"  Box position: x={x}, y={y}, w={box_width}, h={box_height}")
            
        # Draw background rectangle
        box_rect = fitz.Rect(x, y, x + box_width, y + box_height)
            
        page.draw_rect(
            box_rect,
            color=(0.8, 0, 0),      # Dark red border
            fill=(1, 0.95, 0.95),   # Light red fill
            width=2
        )
        
        # Calculate percentage
        percentage = (evaluation.marks_awarded / evaluation.max_marks * 100) if evaluation.max_marks > 0 else 0
        
        # Create combined text
        marks_text = f"Marks: {evaluation.marks_awarded}/{evaluation.max_marks}\n({percentage:.0f}%)"
        
        # Single textbox with all content
        text_rect = fitz.Rect(
            x + 5,
            y + 5,
            x + box_width - 5,
            y + box_height - 5
        )
        
        rc = page.insert_textbox(
            text_rect,
            marks_text,
            fontsize=11,  # Slightly larger
            fontname="hebo",  # Bold font
            color=(0.8, 0, 0),
            align=fitz.TEXT_ALIGN_CENTER
        )
        
        if rc < 0:
            logger.warning(f"Text insertion failed for page {evaluation.page_number}, return code: {rc}")
        else:
            logger.debug(f"  Marks box added successfully")
    
    def _add_summary_page(self, page: fitz.Page, summary: EvaluationSummary) -> None:
        """Add evaluation summary to last page"""
        
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Position: bottom center
        box_width = 300
        box_height = 120
        x = (page_width - box_width) / 2
        y = page_height - box_height - 30
        
        # Draw box
        box_rect = fitz.Rect(
            x - 10,
            y - 10,
            min(x + box_width + 10, page_width),
            min(y + box_height + 10, page_height)
        )
        
        page.draw_rect(
            box_rect,
            color=(0, 0, 0.8),      # Blue border
            fill=(0.95, 0.95, 1),   # Light blue fill
            width=2
        )
        
        # Create summary text
        summary_text = f"""EVALUATION SUMMARY

Total Pages: {summary.total_pages}
Total Marks: {summary.total_marks_awarded} / {summary.total_max_marks}
Percentage: {summary.percentage:.1f}%
Grade: {summary.grade}

{summary.overall_remarks}"""
        
        # Add text
        text_rect = fitz.Rect(
            x,
            y,
            min(x + box_width - 20, page_width - 20),
            min(y + box_height - 10, page_height - 10)
        )
        
        page.insert_textbox(
            text_rect,
            summary_text,
            fontsize=self.config.font_size,
            fontname="hebo",
            color=(0, 0, 0.6),
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _wrap_text(self, text: str, max_width: float, font_size: int) -> List[str]:
        """
        Wrap text to fit within width
        
        Args:
            text: Text to wrap
            max_width: Maximum width in points
            font_size: Font size
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        # Approximate character width
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
        
        return lines[:5]  # Limit to 5 lines