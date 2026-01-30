# src/services/vision_service.py

import cv2
import logging
import numpy as np
from typing import Tuple, Optional
import math

from src.schemas import PageMetadata

# Configure logger
logger = logging.getLogger(__name__)


class VisionService:
    """Computer vision service for image preprocessing and metadata extraction"""
    
    def __init__(self):
        """Initialize vision service"""
        logger.info("Initializing VisionService")
        logger.info("VisionService initialized successfully")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        logger.debug(f"Starting image preprocessing. Input shape: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted BGR image to grayscale")
        else:
            gray = image.copy()
            logger.debug("Image already in grayscale")
        
        # Detect and correct skew
        logger.debug("Detecting and correcting skew")
        gray = self._deskew_image(gray)
        
        # Noise reduction using bilateral filter (preserves edges)
        logger.debug("Applying bilateral filter for noise reduction")
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding for better text extraction
        # ADAPTIVE_THRESH_GAUSSIAN_C works well for varied lighting
        logger.debug("Applying adaptive thresholding")
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Morphological operations to clean up
        logger.debug("Applying morphological operations")
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        logger.debug(f"Image preprocessing complete. Output shape: {cleaned.shape}")
        return cleaned
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in image
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        # Calculate skew angle
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 10:
            logger.debug("Insufficient points for skew detection, skipping deskew")
            return image  # Not enough points to calculate skew
        
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(angle) < 0.5:
            logger.debug(f"Skew angle {angle:.2f}° is insignificant, skipping correction")
            return image
        
        logger.debug(f"Detected skew angle: {angle:.2f}°, correcting...")
        
        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            M, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug(f"Deskew complete. Corrected by {angle:.2f}°")
        return rotated
    
    def extract_metadata(self, image: np.ndarray, page_number: int) -> PageMetadata:
        """
        Extract metadata from image for evaluation context
        
        Args:
            image: Input image (BGR or grayscale)
            page_number: Page number
            
        Returns:
            PageMetadata with vision analysis
        """
        logger.debug(f"[Page {page_number}] Starting metadata extraction")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate writing density
        logger.debug(f"[Page {page_number}] Calculating writing density")
        writing_density = self._calculate_writing_density(gray)
        
        # Detect diagrams/drawings
        logger.debug(f"[Page {page_number}] Detecting diagrams")
        has_diagrams = self._detect_diagrams(gray)
        
        # Detect crossed-out regions
        logger.debug(f"[Page {page_number}] Detecting crossed-out regions")
        crossed_out_regions = self._detect_crossed_out_regions(gray)
        
        # Count text blocks
        logger.debug(f"[Page {page_number}] Counting text blocks")
        text_blocks_count = self._count_text_blocks(gray)
        
        # Detect skew angle
        logger.debug(f"[Page {page_number}] Detecting skew angle")
        skew_angle = self._get_skew_angle(gray)
        
        metadata = PageMetadata(
            page_number=page_number,
            writing_density=writing_density,
            has_diagrams=has_diagrams,
            crossed_out_regions=crossed_out_regions,
            text_blocks_count=text_blocks_count,
            skew_angle=skew_angle
        )
        
        logger.info(f"[Page {page_number}] Metadata extracted: density={writing_density:.3f}, "
                   f"diagrams={has_diagrams}, blocks={text_blocks_count}, skew={skew_angle:.2f}°")
        return metadata
    
    def _calculate_writing_density(self, gray_image: np.ndarray) -> float:
        """Calculate proportion of page with writing"""
        # Threshold to binary
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate ratio of dark pixels (writing) to total pixels
        total_pixels = binary.shape[0] * binary.shape[1]
        dark_pixels = np.sum(binary > 0)
        
        density = dark_pixels / total_pixels
        logger.debug(f"Writing density: {density:.3f} ({dark_pixels}/{total_pixels} pixels)")
        return round(min(density, 1.0), 3)
    
    def _detect_diagrams(self, gray_image: np.ndarray) -> bool:
        """Detect presence of diagrams or drawings"""
        # Use edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Look for large, non-text-like contours
        diagram_like_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Diagrams typically have larger area and different area/perimeter ratio
            if area > 5000:  # Significant size
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter ** 2)
                    # Circles, shapes have circularity closer to 1
                    if 0.3 < circularity < 1.2:
                        diagram_like_contours += 1
        
        has_diagrams = diagram_like_contours > 0
        logger.debug(f"Diagram detection: {diagram_like_contours} diagram-like contours found")
        return has_diagrams
    
    def _detect_crossed_out_regions(self, gray_image: np.ndarray) -> int:
        """Detect crossed-out text regions"""
        # Use line detection (Hough Transform)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None:
            logger.debug("No lines detected for cross-out analysis")
            return 0
        
        # Count diagonal lines (potential cross-outs)
        diagonal_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            
            # Diagonal lines are between 30-60 or 120-150 degrees
            if (30 < angle < 60) or (120 < angle < 150):
                diagonal_lines += 1
        
        # Crossed-out regions typically have multiple diagonal lines
        crossed_regions = diagonal_lines // 2
        logger.debug(f"Cross-out detection: {diagonal_lines} diagonal lines, {crossed_regions} estimated crossed regions")
        return crossed_regions
    
    def _count_text_blocks(self, gray_image: np.ndarray) -> int:
        """Count distinct text blocks on page"""
        # Threshold
        _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate to connect nearby text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours (text blocks)
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter small noise
        significant_blocks = [
            c for c in contours 
            if cv2.contourArea(c) > 1000
        ]
        
        count = len(significant_blocks)
        logger.debug(f"Text block count: {count} significant blocks (from {len(contours)} total contours)")
        return count
    
    def _get_skew_angle(self, gray_image: np.ndarray) -> float:
        """Get detected skew angle"""
        coords = np.column_stack(np.where(gray_image < 200))
        
        if len(coords) < 10:
            logger.debug("Insufficient points for skew angle calculation")
            return 0.0
        
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        logger.debug(f"Detected skew angle: {angle:.2f}°")
        return round(angle, 2)