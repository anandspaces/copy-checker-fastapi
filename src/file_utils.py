# src/file_utils.py

import os
import tempfile
import uuid
import logging
from pathlib import Path
from typing import BinaryIO
import aiofiles
from fastapi import UploadFile

# Configure logger
logger = logging.getLogger(__name__)


class TempFileManager:
    """Manages temporary files with automatic cleanup"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "answer_sheet_eval"
        self.temp_dir.mkdir(exist_ok=True)
        self._created_files = []
        logger.info(f"TempFileManager initialized. Temp directory: {self.temp_dir}")
    
    def create_temp_path(self, suffix: str = ".pdf") -> Path:
        """Create a unique temporary file path"""
        filename = f"{uuid.uuid4()}{suffix}"
        filepath = self.temp_dir / filename
        self._created_files.append(filepath)
        logger.debug(f"Created temp path: {filepath}")
        return filepath
    
    async def save_upload(self, upload_file: UploadFile, suffix: str = ".pdf") -> Path:
        """Save uploaded file to temporary location"""
        temp_path = self.create_temp_path(suffix)
        logger.info(f"Saving upload to: {temp_path}")
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            file_size = temp_path.stat().st_size
            logger.info(f"Upload saved successfully. Size: {file_size} bytes")
            return temp_path
            
        except Exception as e:
            # Clean up on failure
            logger.error(f"Failed to save upload: {str(e)}", exc_info=True)
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up failed upload: {temp_path}")
            raise Exception(f"Failed to save upload: {str(e)}")
    
    def cleanup_file(self, filepath: Path) -> None:
        """Remove a specific temporary file"""
        try:
            if filepath.exists():
                filepath.unlink()
                if filepath in self._created_files:
                    self._created_files.remove(filepath)
                logger.debug(f"Cleaned up temp file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {filepath}: {e}")
    
    def cleanup_all(self) -> None:
        """Remove all tracked temporary files"""
        logger.info(f"Cleaning up {len(self._created_files)} temporary files")
        for filepath in self._created_files[:]:
            self.cleanup_file(filepath)
        self._created_files.clear()
        logger.info("Cleanup complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()


def validate_pdf(filepath: Path) -> bool:
    """Validate that file is a valid PDF"""
    logger.debug(f"Validating PDF: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            is_valid = header == b'%PDF'
            
            if is_valid:
                logger.debug(f"PDF validation passed: {filepath}")
            else:
                logger.warning(f"PDF validation failed: {filepath} (header: {header})")
            
            return is_valid
    except Exception as e:
        logger.error(f"PDF validation error for {filepath}: {str(e)}")
        return False


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        logger.debug(f"File size: {filepath.name} = {size_mb:.2f} MB ({size_bytes} bytes)")
        return size_mb
    except Exception as e:
        logger.error(f"Error getting file size for {filepath}: {str(e)}")
        return 0.0


def ensure_dir(directory: Path) -> None:
    """Ensure directory exists"""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")
    except Exception as e:
        logger.error(f"Failed to ensure directory {directory}: {str(e)}")
        raise