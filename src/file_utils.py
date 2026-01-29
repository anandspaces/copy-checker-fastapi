import os
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO
import aiofiles
from fastapi import UploadFile


class TempFileManager:
    """Manages temporary files with automatic cleanup"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "answer_sheet_eval"
        self.temp_dir.mkdir(exist_ok=True)
        self._created_files = []
    
    def create_temp_path(self, suffix: str = ".pdf") -> Path:
        """Create a unique temporary file path"""
        filename = f"{uuid.uuid4()}{suffix}"
        filepath = self.temp_dir / filename
        self._created_files.append(filepath)
        return filepath
    
    async def save_upload(self, upload_file: UploadFile, suffix: str = ".pdf") -> Path:
        """Save uploaded file to temporary location"""
        temp_path = self.create_temp_path(suffix)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            return temp_path
        except Exception as e:
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            raise Exception(f"Failed to save upload: {str(e)}")
    
    def cleanup_file(self, filepath: Path) -> None:
        """Remove a specific temporary file"""
        try:
            if filepath.exists():
                filepath.unlink()
                if filepath in self._created_files:
                    self._created_files.remove(filepath)
        except Exception as e:
            print(f"Warning: Failed to cleanup {filepath}: {e}")
    
    def cleanup_all(self) -> None:
        """Remove all tracked temporary files"""
        for filepath in self._created_files[:]:
            self.cleanup_file(filepath)
        self._created_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()


def validate_pdf(filepath: Path) -> bool:
    """Validate that file is a valid PDF"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes"""
    return filepath.stat().st_size / (1024 * 1024)


def ensure_dir(directory: Path) -> None:
    """Ensure directory exists"""
    directory.mkdir(parents=True, exist_ok=True)