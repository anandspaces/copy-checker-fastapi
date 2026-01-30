# src/services/pdf_annotation_service.py

import fitz  # PyMuPDF
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from src.schemas import QuestionData, AnnotationConfig, EvaluationSummary

load_dotenv()
logger = logging.getLogger(__name__)


class PDFAnnotationService:
    """
    Human-like PDF annotation service that:
    1. Uses Gemini Vision to find blank space coordinates
    2. Writes marks and remarks naturally like a teacher
    3. Uses handwriting-style fonts and red pen effect
    4. Handles multi-page questions intelligently
    """
    
    def __init__(self, config: AnnotationConfig = None):
        """
        Initialize PDF annotation service
        
        Args:
            config: Annotation configuration
        """
        logger.info("Initializing PDFAnnotationService (Human-like)")
        self.config = config or AnnotationConfig()
        
        # Initialize Gemini for coordinate detection
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"
        
        logger.info("PDFAnnotationService initialized successfully")
    
    def annotate_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        questions: List[QuestionData],
        summary: EvaluationSummary
    ) -> Path:
        """
        Annotate PDF with human-like remarks and marks
        
        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path for annotated output
            questions: List of evaluated questions
            summary: Overall evaluation summary
            
        Returns:
            Path to annotated PDF
        """
        logger.info(f"Starting human-like annotation: {input_pdf_path}")
        logger.info(f"Total questions to annotate: {len(questions)}")
        
        doc = None
        try:
            doc = fitz.open(input_pdf_path)
            
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            
            # Process each question
            for idx, question in enumerate(questions, 1):
                logger.info(f"Annotating Q{question.question_number} ({idx}/{len(questions)})")
                
                try:
                    # Find the best page to annotate (last page of question)
                    target_page_num = question.page_numbers[-1] if question.page_numbers else 1
                    
                    if target_page_num > len(doc):
                        logger.warning(f"Q{question.question_number}: Page {target_page_num} out of range")
                        target_page_num = len(doc)
                    
                    page = doc[target_page_num - 1]  # 0-indexed
                    
                    # Get page image for coordinate detection
                    page_image = self._get_page_image(page)
                    
                    # Find optimal annotation coordinates using Gemini Vision
                    coords = self._find_annotation_coordinates(
                        page_image, 
                        question,
                        target_page_num
                    )
                    
                    if coords:
                        # Add human-like annotations
                        self._add_human_like_annotations(
                            page, 
                            question, 
                            coords
                        )
                        logger.info(f"✓ Q{question.question_number} annotated on page {target_page_num}")
                    else:
                        # Fallback: annotate in margin
                        self._add_margin_annotations(page, question)
                        logger.warning(f"⚠ Q{question.question_number}: Used fallback margin annotation")
                        
                except Exception as e:
                    logger.error(f"Failed to annotate Q{question.question_number}: {e}")
                    # Try fallback
                    try:
                        page = doc[min(question.page_numbers[-1] - 1 if question.page_numbers else 0, len(doc) - 1)]
                        self._add_margin_annotations(page, question)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for Q{question.question_number}: {fallback_error}")
            
            # Add summary on last page
            if self.config.show_summary and len(doc) > 0:
                try:
                    self._add_summary_section(doc[-1], summary)
                    logger.info("✓ Summary added to last page")
                except Exception as e:
                    logger.error(f"Failed to add summary: {e}")
            
            # Save annotated PDF
            doc.save(str(output_pdf_path), garbage=4, deflate=True, clean=True)
            logger.info(f"✓ Annotated PDF saved: {output_pdf_path}")
            
            return output_pdf_path
            
        except Exception as e:
            logger.error(f"PDF annotation failed: {e}", exc_info=True)
            raise Exception(f"Failed to annotate PDF: {str(e)}")
        
        finally:
            if doc:
                doc.close()
    
    def _get_page_image(self, page: fitz.Page) -> bytes:
        """Convert PDF page to image for Gemini Vision"""
        try:
            # High resolution for better coordinate detection
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            return img_data
        except Exception as e:
            logger.error(f"Failed to convert page to image: {e}")
            raise
    
    def _find_annotation_coordinates(
        self, 
        page_image: bytes, 
        question: QuestionData,
        page_num: int
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Use Gemini Vision to find optimal coordinates for annotations
        
        Args:
            page_image: Page image as PNG bytes
            question: Question data
            page_num: Page number
            
        Returns:
            Dict with 'marks' and 'remarks' coordinates as (x, y) tuples
            or None if detection fails
        """
        try:
            # Create prompt for coordinate detection
            prompt = f"""You are analyzing an answer sheet page to find optimal locations for handwritten teacher annotations.

TASK: Find the best blank space coordinates to write:
1. Marks: "Q{question.question_number}: {question.marks_awarded}/{question.allocated_marks or '?'}" (needs ~100-120 pixels width)
2. Remarks: "{question.remarks[:100] if question.remarks else 'Good attempt'}" (needs ~200-250 pixels width)

REQUIREMENTS:
- Find blank spaces near the ANSWER content for Q{question.question_number}
- Prefer RIGHT MARGIN or CENTER-RIGHT areas (x > 60% of page width)
- Avoid overlapping with existing text or answers
- Marks should be ABOVE remarks
- Both should be visible and not cramped

COORDINATE SYSTEM:
- Origin (0, 0) is TOP-LEFT corner
- Page dimensions are typically 595x842 (A4 portrait) at 72 DPI
- At 2x zoom (current image), multiply by 2: 1190x1684
- Return coordinates for 1x (normal PDF coordinates)

Return ONLY valid JSON:
{{
  "marks_position": {{"x": <number 350-550>, "y": <number 50-750>}},
  "remarks_position": {{"x": <number 300-550>, "y": <number 70-770>}},
  "page_width": <detected page width at 1x>,
  "page_height": <detected page height at 1x>,
  "confidence": <"high" or "medium" or "low">,
  "reasoning": "<brief explanation of chosen positions>"
}}

EDGE CASES:
- If page is very full, use top-right or bottom-right margins
- If answer is multi-column, choose between columns
- Ensure remarks_y > marks_y + 20 (remarks below marks)
"""
            
            # Call Gemini Vision
            contents = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=page_image, mime_type="image/png")
            ]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            result = json.loads(response.text.strip())
            
            confidence = result.get("confidence", "low")
            logger.info(f"Coordinate detection confidence: {confidence}")
            
            # Validate coordinates
            marks_pos = result.get("marks_position", {})
            remarks_pos = result.get("remarks_position", {})
            
            if not marks_pos or not remarks_pos:
                logger.warning("Missing coordinate data from Gemini")
                return None
            
            # Extract and validate
            coords = {
                "marks": (float(marks_pos["x"]), float(marks_pos["y"])),
                "remarks": (float(remarks_pos["x"]), float(remarks_pos["y"])),
                "confidence": confidence
            }
            
            # Sanity checks
            page_width = result.get("page_width", 595)
            page_height = result.get("page_height", 842)
            
            # Ensure coordinates are within page bounds
            coords["marks"] = (
                min(max(coords["marks"][0], 50), page_width - 50),
                min(max(coords["marks"][1], 30), page_height - 30)
            )
            coords["remarks"] = (
                min(max(coords["remarks"][0], 50), page_width - 50),
                min(max(coords["remarks"][1], 50), page_height - 50)
            )
            
            logger.info(f"Coordinates found: Marks={coords['marks']}, Remarks={coords['remarks']}")
            return coords
            
        except Exception as e:
            logger.error(f"Coordinate detection failed: {e}", exc_info=True)
            return None
    
    def _add_human_like_annotations(
        self,
        page: fitz.Page,
        question: QuestionData,
        coords: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Add human-like handwritten annotations with red pen effect
        
        Args:
            page: PyMuPDF page object
            question: Question data with evaluation
            coords: Coordinate dictionary
        """
        # Red pen color (like teacher's red pen)
        red_pen = (0.8, 0.1, 0.1)  # Slightly darker red
        
        # Font sizes (slightly irregular like handwriting)
        marks_fontsize = 12
        remarks_fontsize = 10
        
        # Get coordinates
        marks_x, marks_y = coords["marks"]
        remarks_x, remarks_y = coords["remarks"]
        
        try:
            # 1. Write marks (larger, bold-ish)
            marks_text = f"Q{question.question_number}: {question.marks_awarded}"
            if question.allocated_marks:
                marks_text += f"/{question.allocated_marks}"
                percentage = (question.marks_awarded / question.allocated_marks * 100)
                marks_text += f" ({percentage:.0f}%)"
            
            # Add marks with slight hand-written feel
            page.insert_text(
                (marks_x, marks_y),
                marks_text,
                fontsize=marks_fontsize,
                fontname="helv",  # Helvetica
                color=red_pen,
                rotate=0
            )
            
            # Add oval around marks (FIXED - removed get_text_selection)
            oval_rect = fitz.Rect(
                marks_x - 5, 
                marks_y - marks_fontsize - 2, 
                marks_x + len(marks_text) * 6,  # Approximate width
                marks_y + 3
            )
            page.draw_oval(oval_rect, color=red_pen, width=1.5)
            
        except Exception as e:
            logger.error(f"Failed to add marks annotation: {e}")
        
        try:
            # 2. Write remarks (multi-line if needed)
            if self.config.show_remarks and question.remarks:
                remarks_text = question.remarks
                
                # Truncate if too long
                max_chars_per_line = 40
                max_lines = 3
                
                # Wrap text
                words = remarks_text.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    if len(test_line) <= max_chars_per_line:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        if len(lines) >= max_lines:
                            break
                
                if current_line and len(lines) < max_lines:
                    lines.append(' '.join(current_line))
                
                # Add ellipsis if truncated
                if len(lines) == max_lines and len(words) > len(' '.join(lines).split()):
                    lines[-1] = lines[-1][:max_chars_per_line - 3] + "..."
                
                # Draw remarks line by line
                y_offset = remarks_y
                line_height = remarks_fontsize + 4
                
                for line in lines:
                    page.insert_text(
                        (remarks_x, y_offset),
                        line,
                        fontsize=remarks_fontsize,
                        fontname="helv",
                        color=red_pen,
                        rotate=0
                    )
                    y_offset += line_height
                
                # Add underline for emphasis (teacher style)
                if lines:
                    underline_y = y_offset - 2
                    page.draw_line(
                        (remarks_x, underline_y),
                        (remarks_x + 200, underline_y),
                        color=red_pen,
                        width=0.8
                    )
                    
        except Exception as e:
            logger.error(f"Failed to add remarks annotation: {e}")
        
        try:
            # 3. Add checkmark or cross (visual indicator)
            icon_x = marks_x - 15
            icon_y = marks_y - 8
            
            if question.is_correct:
                # Draw checkmark (✓)
                page.draw_line((icon_x, icon_y), (icon_x + 3, icon_y + 5), 
                              color=(0, 0.6, 0), width=2)
                page.draw_line((icon_x + 3, icon_y + 5), (icon_x + 8, icon_y - 3), 
                              color=(0, 0.6, 0), width=2)
            else:
                # Draw cross (✗)
                page.draw_line((icon_x, icon_y), (icon_x + 8, icon_y + 8), 
                              color=red_pen, width=2)
                page.draw_line((icon_x, icon_y + 8), (icon_x + 8, icon_y), 
                              color=red_pen, width=2)
                
        except Exception as e:
            logger.error(f"Failed to add icon: {e}")
    
    def _add_margin_annotations(
        self,
        page: fitz.Page,
        question: QuestionData
    ) -> None:
        """
        Fallback: Add annotations in right margin
        
        Args:
            page: PyMuPDF page object
            question: Question data
        """
        rect = page.rect
        page_width = rect.width
        
        # Right margin position
        margin_x = page_width - 160
        margin_y = 80 + (hash(question.question_number) % 500)
        
        red_pen = (0.8, 0.1, 0.1)
        
        try:
            # Marks
            marks_text = f"Q{question.question_number}: {question.marks_awarded}"
            if question.allocated_marks:
                marks_text += f"/{question.allocated_marks}"
            
            page.insert_text(
                (margin_x, margin_y),
                marks_text,
                fontsize=11,
                fontname="helv",
                color=red_pen
            )
            
            # Draw box around it
            box_rect = fitz.Rect(margin_x - 5, margin_y - 12, 
                                margin_x + 100, margin_y + 5)
            page.draw_rect(box_rect, color=red_pen, width=1.5)
            
            # Remarks below
            if self.config.show_remarks and question.remarks:
                remarks_y = margin_y + 20
                remarks_text = question.remarks[:60] + "..." if len(question.remarks) > 60 else question.remarks
                
                page.insert_text(
                    (margin_x, remarks_y),
                    remarks_text,
                    fontsize=9,
                    fontname="helv",
                    color=red_pen
                )
                
        except Exception as e:
            logger.error(f"Margin annotation failed: {e}")
    
    def _add_summary_section(
        self,
        page: fitz.Page,
        summary: EvaluationSummary
    ) -> None:
        """
        Add evaluation summary in teacher-like style
        
        Args:
            page: Last page of PDF
            summary: Evaluation summary
        """
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Position at bottom
        summary_x = 50
        summary_y = page_height - 180
        
        # Colors
        header_color = (0.1, 0.1, 0.6)
        text_color = (0.2, 0.2, 0.2)
        red_pen = (0.8, 0.1, 0.1)
        
        try:
            # Draw box
            box_rect = fitz.Rect(summary_x - 10, summary_y - 20, 
                                page_width - 50, summary_y + 150)
            page.draw_rect(box_rect, color=header_color, width=2)
            
            # Header
            page.insert_text(
                (summary_x, summary_y),
                "EVALUATION SUMMARY",
                fontsize=14,
                fontname="helv",  # FIXED - using helv instead of hebo
                color=header_color
            )
            
            # Draw underline
            page.draw_line(
                (summary_x, summary_y + 3),
                (summary_x + 200, summary_y + 3),
                color=header_color,
                width=1.5
            )
            
            # Summary details
            y_pos = summary_y + 25
            line_height = 18
            
            details = [
                f"Total Questions Evaluated: {summary.total_questions}",
                f"Marks Obtained: {summary.total_marks_awarded} / {summary.total_max_marks}",
                f"Percentage: {summary.percentage:.1f}%",
                f"Grade: {summary.grade}",
                "",
                f"Remarks: {summary.overall_remarks}"
            ]
            
            for detail in details:
                if detail:
                    # Highlight grade in red if it's the grade line
                    if detail.startswith("Grade:"):
                        page.insert_text(
                            (summary_x, y_pos),
                            detail,
                            fontsize=12,
                            fontname="helv",  # FIXED
                            color=red_pen
                        )
                    else:
                        page.insert_text(
                            (summary_x, y_pos),
                            detail,
                            fontsize=10,
                            fontname="helv",  # FIXED
                            color=text_color
                        )
                y_pos += line_height
            
            # Add signature line (teacher style)
            sig_y = y_pos + 10
            page.draw_line(
                (summary_x, sig_y),
                (summary_x + 200, sig_y),
                color=text_color,
                width=0.8,
                dashes=[3, 2]  # Dashed line
            )
            
            page.insert_text(
                (summary_x, sig_y + 15),
                "Checked by AI Evaluator",
                fontsize=8,
                fontname="helv",  # FIXED
                color=text_color
            )
            
        except Exception as e:
            logger.error(f"Failed to add summary: {e}")