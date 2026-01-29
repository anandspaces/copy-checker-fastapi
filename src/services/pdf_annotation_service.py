import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from src.schemas import PageEvaluation, AnnotationConfig, EvaluationSummary


class PDFAnnotationService:
    """Service for annotating PDFs with evaluation marks and remarks"""
    
    def __init__(self, config: AnnotationConfig = None):
        """Initialize PDF annotation service"""
        self.config = config or AnnotationConfig()
    
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
        try:
            # Open original PDF
            doc = fitz.open(input_pdf_path)
            
            # Create evaluation lookup
            eval_map = {eval.page_number: eval for eval in evaluations}
            
            # Annotate each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_eval = eval_map.get(page_num + 1)  # 1-indexed
                
                if page_eval:
                    self._annotate_page(page, page_eval, page_num + 1)
            
            # Add summary to last page
            if len(doc) > 0:
                last_page = doc[-1]
                self._add_summary_to_page(last_page, summary)
            
            # Save annotated PDF
            doc.save(str(output_pdf_path))
            doc.close()
            
            return output_pdf_path
            
        except Exception as e:
            raise Exception(f"Failed to annotate PDF: {str(e)}")
    
    def _annotate_page(
        self,
        page: fitz.Page,
        evaluation: PageEvaluation,
        page_number: int
    ) -> None:
        """
        Annotate a single page with marks and remarks
        
        Args:
            page: PyMuPDF page object
            evaluation: Page evaluation data
            page_number: Page number (1-indexed)
        """
        # Get page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Position for remarks (top-right area)
        remarks_x = page_width - 200
        remarks_y = 30
        
        # Position for marks (bottom-right corner)
        marks_x = page_width - 150
        marks_y = page_height - 50
        
        # Add remarks with background box
        self._add_remark_box(page, remarks_x, remarks_y, evaluation.remarks)
        
        # Add marks at bottom
        marks_text = f"Marks: {evaluation.marks_awarded}/{evaluation.max_marks}"
        self._add_marks_text(page, marks_x, marks_y, marks_text)
    
    def _add_remark_box(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        remark: str
    ) -> None:
        """
        Add remark with background box
        
        Args:
            page: PyMuPDF page object
            x, y: Position coordinates
            remark: Remark text
        """
        # Truncate remark if too long
        max_length = 150
        if len(remark) > max_length:
            remark = remark[:max_length-3] + "..."
        
        # Font configuration - use built-in fonts (no external file needed)
        font_size = self.config.font_size
        # Use built-in font name (no file needed)
        fontname = "helv"  # Built-in Helvetica
        
        # Estimate box dimensions
        text_width = len(remark) * font_size * 0.5
        text_height = font_size + 10
        
        # Adjust if text is too wide - split into multiple lines
        max_width = 180
        if text_width > max_width:
            words = remark.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                word_width = len(word) * font_size * 0.5
                space_width = font_size * 0.3
                
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
            
            remark = '\n'.join(lines[:3])  # Max 3 lines
            text_height = (len(lines) * (font_size + 3)) + 10
        
        # Draw semi-transparent background box
        box_rect = fitz.Rect(
            x - 5,
            y - 5,
            x + 195,
            y + text_height
        )
        
        # Light yellow background with border
        page.draw_rect(
            box_rect,
            color=(0.8, 0.8, 0),  # Darker yellow border
            fill=(1, 1, 0.9),     # Light yellow fill
            width=1
        )
        
        # Add text using insert_textbox for better multi-line support
        text_rect = fitz.Rect(x, y, x + 185, y + text_height)
        
        # Use insert_textbox with built-in font (more reliable)
        page.insert_textbox(
            text_rect,
            remark,
            fontsize=font_size,
            fontname=fontname,
            color=self.config.font_color,
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _add_marks_text(
        self,
        page: fitz.Page,
        x: float,
        y: float,
        marks_text: str
    ) -> None:
        """
        Add marks text with box
        
        Args:
            page: PyMuPDF page object
            x, y: Position coordinates
            marks_text: Marks text (e.g., "Marks: 8/10")
        """
        font_size = self.config.font_size + 2  # Slightly larger
        # Use built-in bold font
        fontname = "hebo"  # Built-in Helvetica Bold
        
        # Draw box
        box_rect = fitz.Rect(x - 5, y - 15, x + 140, y + 10)
        page.draw_rect(
            box_rect,
            color=(0.8, 0, 0),      # Dark red border
            fill=(1, 0.95, 0.95),   # Light red background
            width=2
        )
        
        # Add text using insert_textbox
        text_rect = fitz.Rect(x, y - 12, x + 130, y + 8)
        
        # Use insert_textbox with built-in font
        page.insert_textbox(
            text_rect,
            marks_text,
            fontsize=font_size,
            fontname=fontname,
            color=(0.8, 0, 0),  # Dark red
            align=fitz.TEXT_ALIGN_LEFT
        )
    
    def _add_summary_to_page(
        self,
        page: fitz.Page,
        summary: EvaluationSummary
    ) -> None:
        """
        Add evaluation summary to page (typically last page)
        
        Args:
            page: PyMuPDF page object
            summary: Evaluation summary
        """
        # Get page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Position for summary (bottom center)
        summary_x = page_width / 2 - 110
        summary_y = page_height - 100
        
        # Create summary text
        summary_text = f"""EVALUATION SUMMARY
Total Pages: {summary.total_pages}
Total Marks: {summary.total_marks_awarded}/{summary.total_max_marks}
Percentage: {summary.percentage:.1f}%"""
        
        # Draw box
        box_rect = fitz.Rect(
            summary_x - 10,
            summary_y - 10,
            summary_x + 240,
            summary_y + 65
        )
        
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
            summary_x + 220,
            summary_y + 60
        )
        
        # Use built-in bold font
        font_size = 10
        fontname = "hebo"  # Built-in Helvetica Bold
        
        page.insert_textbox(
            text_rect,
            summary_text,
            fontsize=font_size,
            fontname=fontname,
            color=(0, 0, 0.6),  # Dark blue
            align=fitz.TEXT_ALIGN_LEFT
        )