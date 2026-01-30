# src/services/pdf_annotation_service.py
# FIXED VERSION - More robust annotation with better error handling

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List

from src.schemas import PageEvaluation, AnnotationConfig, EvaluationSummary

# Configure logger
logger = logging.getLogger(__name__)


class PDFAnnotationService:
    """Service for annotating PDFs with evaluation marks and remarks"""
    
    def __init__(self, config: AnnotationConfig = None):
        """Initialize PDF annotation service"""
        logger.info("Initializing PDFAnnotationService")
        self.config = config or AnnotationConfig()
        logger.info(f"Annotation config: font_size={self.config.font_size}, "
                   f"color={self.config.font_color}")
    
    def annotate_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        evaluations: List[PageEvaluation],
        summary: EvaluationSummary
    ) -> Path:
        """
        Annotate PDF with evaluation marks and remarks
        
        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to save annotated PDF
            evaluations: List of page evaluations
            summary: Overall evaluation summary
            
        Returns:
            Path to annotated PDF
        """
        logger.info(f"Starting PDF annotation: {input_pdf_path} -> {output_pdf_path}")
        logger.info(f"Annotating {len(evaluations)} pages")
        
        doc = None
        try:
            # Open original PDF
            logger.debug(f"Opening PDF: {input_pdf_path}")
            doc = fitz.open(input_pdf_path)
            logger.debug(f"PDF opened. Total pages: {len(doc)}")
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Create evaluation lookup
            eval_map = {eval.page_number: eval for eval in evaluations}
            
            # Annotate each page with comprehensive error handling
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_eval = eval_map.get(page_num + 1)  # 1-indexed
                
                if page_eval:
                    logger.debug(f"Annotating page {page_num + 1}")
                    try:
                        self._annotate_page_safe(page, page_eval, page_num + 1)
                    except Exception as e:
                        logger.error(f"Failed to annotate page {page_num + 1}: {str(e)}", exc_info=True)
                        # Continue to next page instead of failing entire process
                        continue
                else:
                    logger.warning(f"No evaluation found for page {page_num + 1}")
            
            # Add summary to last page
            if len(doc) > 0:
                logger.debug("Adding evaluation summary to last page")
                try:
                    last_page = doc[-1]
                    self._add_summary_to_page_safe(last_page, summary)
                except Exception as e:
                    logger.error(f"Failed to add summary: {str(e)}", exc_info=True)
                    # Non-critical, continue anyway
            
            # Save annotated PDF
            logger.debug(f"Saving annotated PDF to: {output_pdf_path}")
            doc.save(str(output_pdf_path), garbage=4, deflate=True, clean=True)
            logger.info(f"PDF annotation complete: {output_pdf_path}")
            
            return output_pdf_path
            
        except Exception as e:
            logger.error(f"Failed to annotate PDF: {str(e)}", exc_info=True)
            raise Exception(f"Failed to annotate PDF: {str(e)}")
        
        finally:
            # Always close document
            if doc:
                doc.close()
                logger.debug("PDF document closed")
    
    def _annotate_page_safe(
        self,
        page: fitz.Page,
        evaluation: PageEvaluation,
        page_number: int
    ) -> None:
        """
        Annotate a single page with marks and remarks (with error handling)
        
        Args:
            page: PyMuPDF page object
            evaluation: Page evaluation data
            page_number: Page number (1-indexed)
        """
        logger.debug(f"[Page {page_number}] Starting page annotation")
        
        try:
            # Get page dimensions
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
            logger.debug(f"[Page {page_number}] Page dimensions: {page_width:.1f}x{page_height:.1f}")
            
            # Validate page dimensions
            if page_width < 100 or page_height < 100:
                logger.warning(f"[Page {page_number}] Page too small, skipping annotation")
                return
            
            # Position for remarks (top-right area)
            remarks_x = max(page_width - 210, 10)  # Ensure not off-page
            remarks_y = 20
            
            # Position for marks (bottom-right corner)
            marks_x = max(page_width - 160, 10)
            marks_y = max(page_height - 60, page_height - 80)
            
            # Add remarks with background box
            logger.debug(f"[Page {page_number}] Adding remark box at ({remarks_x:.1f}, {remarks_y:.1f})")
            try:
                self._add_remark_box_safe(page, remarks_x, remarks_y, evaluation.remarks, page_width, page_height)
            except Exception as e:
                logger.error(f"[Page {page_number}] Failed to add remark box: {e}")
                # Try simple text fallback
                try:
                    self._add_simple_text(page, remarks_x, remarks_y, evaluation.remarks[:50])
                except:
                    pass
            
            # Add marks at bottom
            marks_text = f"Marks: {evaluation.marks_awarded}/{evaluation.max_marks}"
            logger.debug(f"[Page {page_number}] Adding marks text at ({marks_x:.1f}, {marks_y:.1f}): {marks_text}")
            try:
                self._add_marks_text_safe(page, marks_x, marks_y, marks_text, page_width, page_height)
            except Exception as e:
                logger.error(f"[Page {page_number}] Failed to add marks text: {e}")
                # Try simple text fallback
                try:
                    self._add_simple_text(page, marks_x, marks_y, marks_text)
                except:
                    pass
            
            logger.debug(f"[Page {page_number}] Page annotation complete")
            
        except Exception as e:
            logger.error(f"[Page {page_number}] Error in page annotation: {e}", exc_info=True)
            raise
    
    def _add_remark_box_safe(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        remark: str,
        page_width: float,
        page_height: float
    ) -> None:
        """
        Add remark with background box (safe version)
        
        Args:
            page: PyMuPDF page object
            x, y: Position coordinates
            remark: Remark text
            page_width, page_height: Page dimensions
        """
        try:
            # Sanitize and truncate remark
            remark = str(remark).strip()
            if not remark or remark == "":
                remark = "Answer evaluated."
            
            max_length = 120
            original_length = len(remark)
            if len(remark) > max_length:
                remark = remark[:max_length-3] + "..."
                logger.debug(f"Remark truncated from {original_length} to {len(remark)} characters")
            
            # Font configuration - use built-in fonts only
            font_size = max(8, min(self.config.font_size, 12))  # Clamp between 8-12
            fontname = "helv"  # Built-in Helvetica
            
            # Calculate box dimensions
            max_width = min(190, page_width - x - 10)  # Don't overflow page
            if max_width < 50:
                logger.warning("Not enough space for remark box")
                return
            
            # Split into lines if needed
            words = remark.split()
            lines = []
            current_line = []
            current_width = 0
            char_width = font_size * 0.5
            
            for word in words:
                word_width = len(word) * char_width
                space_width = char_width
                
                if current_width + word_width + space_width < max_width:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 3 lines
            lines = lines[:3]
            remark_text = '\n'.join(lines)
            
            line_height = font_size + 4
            text_height = (len(lines) * line_height) + 8
            
            # Ensure box doesn't overflow page
            if y + text_height > page_height:
                y = page_height - text_height - 10
            
            # Draw semi-transparent background box
            box_rect = fitz.Rect(
                max(x - 5, 0),
                max(y - 5, 0),
                min(x + max_width + 5, page_width),
                min(y + text_height, page_height)
            )
            
            logger.debug(f"Drawing remark box: {box_rect}")
            
            # Yellow background with border
            page.draw_rect(
                box_rect,
                color=(0.8, 0.8, 0),  # Dark yellow border
                fill=(1, 1, 0.9),     # Light yellow fill
                width=1.5
            )
            
            # Add text using insert_textbox
            text_rect = fitz.Rect(
                x,
                y,
                min(x + max_width, page_width - 5),
                min(y + text_height - 5, page_height - 5)
            )
            
            # Insert text
            rc = page.insert_textbox(
                text_rect,
                remark_text,
                fontsize=font_size,
                fontname=fontname,
                color=(0, 0, 0),  # Black text
                align=fitz.TEXT_ALIGN_LEFT
            )
            
            if rc < 0:
                logger.warning(f"insert_textbox returned {rc}, text may not fit")
            
            logger.debug("Remark box added successfully")
            
        except Exception as e:
            logger.error(f"Error in _add_remark_box_safe: {e}", exc_info=True)
            raise
    
    def _add_marks_text_safe(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        marks_text: str,
        page_width: float,
        page_height: float
    ) -> None:
        """
        Add marks text with box (safe version)
        
        Args:
            page: PyMuPDF page object
            x, y: Position coordinates
            marks_text: Marks text (e.g., "Marks: 8/10")
            page_width, page_height: Page dimensions
        """
        try:
            logger.debug(f"Adding marks text: {marks_text}")
            
            font_size = max(10, min(self.config.font_size + 2, 14))  # Clamp 10-14
            fontname = "hebo"  # Built-in Helvetica Bold
            
            # Calculate box dimensions
            box_width = 145
            box_height = 28
            
            # Ensure box fits on page
            if x + box_width > page_width:
                x = page_width - box_width - 5
            if y + box_height > page_height:
                y = page_height - box_height - 5
            
            # Draw box
            box_rect = fitz.Rect(
                max(x - 5, 0),
                max(y - 15, 0),
                min(x + box_width, page_width),
                min(y + 12, page_height)
            )
            
            logger.debug(f"Drawing marks box: {box_rect}")
            
            page.draw_rect(
                box_rect,
                color=(0.8, 0, 0),      # Dark red border
                fill=(1, 0.95, 0.95),   # Light red background
                width=2
            )
            
            # Add text using insert_textbox
            text_rect = fitz.Rect(
                x,
                y - 12,
                min(x + box_width - 10, page_width - 5),
                min(y + 8, page_height - 5)
            )
            
            # Insert text
            rc = page.insert_textbox(
                text_rect,
                marks_text,
                fontsize=font_size,
                fontname=fontname,
                color=(0.8, 0, 0),  # Dark red
                align=fitz.TEXT_ALIGN_LEFT
            )
            
            if rc < 0:
                logger.warning(f"insert_textbox returned {rc}, text may not fit")
            
            logger.debug("Marks text added successfully")
            
        except Exception as e:
            logger.error(f"Error in _add_marks_text_safe: {e}", exc_info=True)
            raise
    
    def _add_simple_text(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        text: str
    ) -> None:
        """
        Fallback: Add simple text without box
        """
        try:
            logger.debug(f"Adding simple fallback text at ({x:.1f}, {y:.1f})")
            point = fitz.Point(x, y)
            page.insert_text(
                point,
                text[:100],  # Limit length
                fontsize=10,
                fontname="helv",
                color=(0, 0, 0)
            )
            logger.debug("Simple text added")
        except Exception as e:
            logger.error(f"Even simple text failed: {e}")
    
    def _add_summary_to_page_safe(
        self,
        page: fitz.Page,
        summary: EvaluationSummary
    ) -> None:
        """
        Add evaluation summary to page (safe version)
        
        Args:
            page: PyMuPDF page object
            summary: Evaluation summary
        """
        try:
            logger.debug("Adding evaluation summary")
            
            # Get page dimensions
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
            
            # Position for summary (bottom center)
            box_width = 240
            box_height = 80
            summary_x = max((page_width / 2) - (box_width / 2), 10)
            summary_y = max(page_height - box_height - 20, 10)
            
            logger.debug(f"Summary position: ({summary_x:.1f}, {summary_y:.1f})")
            
            # Create summary text
            summary_text = f"""EVALUATION SUMMARY
Total Pages: {summary.total_pages}
Total Marks: {summary.total_marks_awarded}/{summary.total_max_marks}
Percentage: {summary.percentage:.1f}%"""
            
            logger.debug(f"Summary text:\n{summary_text}")
            
            # Ensure box fits on page
            if summary_x + box_width > page_width:
                summary_x = page_width - box_width - 10
            if summary_y + box_height > page_height:
                summary_y = page_height - box_height - 10
            
            # Draw box
            box_rect = fitz.Rect(
                max(summary_x - 10, 0),
                max(summary_y - 10, 0),
                min(summary_x + box_width, page_width),
                min(summary_y + box_height, page_height)
            )
            
            logger.debug(f"Drawing summary box: {box_rect}")
            
            page.draw_rect(
                box_rect,
                color=(0, 0, 0.8),      # Blue border
                fill=(0.95, 0.95, 1),   # Light blue background
                width=2
            )
            
            # Add text using insert_textbox
            text_rect = fitz.Rect(
                summary_x,
                summary_y,
                min(summary_x + box_width - 20, page_width - 10),
                min(summary_y + box_height - 10, page_height - 10)
            )
            
            # Use built-in bold font
            font_size = 10
            fontname = "hebo"  # Built-in Helvetica Bold
            
            rc = page.insert_textbox(
                text_rect,
                summary_text,
                fontsize=font_size,
                fontname=fontname,
                color=(0, 0, 0.6),  # Dark blue
                align=fitz.TEXT_ALIGN_LEFT
            )
            
            if rc < 0:
                logger.warning(f"Summary textbox returned {rc}, text may not fit")
            
            logger.debug("Evaluation summary added successfully")
            
        except Exception as e:
            logger.error(f"Error adding summary: {e}", exc_info=True)
            # Don't raise - summary is non-critical