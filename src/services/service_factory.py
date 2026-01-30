# src/services/service_factory.py

import logging
from typing import Optional

from src.schemas import LLMProvider, AnnotationConfig
from src.services.ocr_service import OCRService, GeminiOCRService
from src.services.evaluation_service import EvaluationService, GeminiEvaluationService
from src.services.pdf_annotation_service import PDFAnnotationService
from src.services.orchestrator_service import EvaluationOrchestrator

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating and managing service instances
    Implements dependency injection pattern
    """
    
    _ocr_service: Optional[OCRService] = None
    _evaluation_service: Optional[EvaluationService] = None
    _annotation_service: Optional[PDFAnnotationService] = None
    _orchestrator: Optional[EvaluationOrchestrator] = None
    
    @classmethod
    def get_ocr_service(cls, provider: str = "gemini") -> OCRService:
        """
        Get or create OCR service instance
        
        Args:
            provider: OCR provider name
            
        Returns:
            OCR service instance
        """
        if cls._ocr_service is None:
            logger.info(f"Creating OCR service: {provider}")
            
            if provider.lower() == "gemini":
                cls._ocr_service = GeminiOCRService()
            else:
                raise ValueError(f"Unsupported OCR provider: {provider}")
        
        return cls._ocr_service
    
    @classmethod
    def get_evaluation_service(cls, provider: LLMProvider = LLMProvider.GEMINI) -> EvaluationService:
        """
        Get or create evaluation service instance
        
        Args:
            provider: LLM provider
            
        Returns:
            Evaluation service instance
        """
        if cls._evaluation_service is None:
            logger.info(f"Creating evaluation service: {provider}")
            
            if provider == LLMProvider.GEMINI:
                cls._evaluation_service = GeminiEvaluationService()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return cls._evaluation_service
    
    @classmethod
    def get_annotation_service(cls, config: Optional[AnnotationConfig] = None) -> PDFAnnotationService:
        """
        Get or create annotation service instance
        
        Args:
            config: Annotation configuration
            
        Returns:
            Annotation service instance
        """
        if cls._annotation_service is None:
            logger.info("Creating annotation service")
            cls._annotation_service = PDFAnnotationService(config)
        
        return cls._annotation_service
    
    @classmethod
    def get_orchestrator(
        cls,
        ocr_provider: str = "gemini",
        llm_provider: LLMProvider = LLMProvider.GEMINI,
        annotation_config: Optional[AnnotationConfig] = None
    ) -> EvaluationOrchestrator:
        """
        Get or create orchestrator with all dependencies
        
        Args:
            ocr_provider: OCR provider name
            llm_provider: LLM provider
            annotation_config: Annotation configuration
            
        Returns:
            Evaluation orchestrator instance
        """
        logger.info("Creating orchestrator with dependencies")
        
        # Get or create services
        ocr_service = cls.get_ocr_service(ocr_provider)
        evaluation_service = cls.get_evaluation_service(llm_provider)
        annotation_service = cls.get_annotation_service(annotation_config)
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(
            ocr_service=ocr_service,
            evaluation_service=evaluation_service,
            annotation_service=annotation_service
        )
        
        return orchestrator
    
    @classmethod
    def reset(cls):
        """Reset all service instances (useful for testing)"""
        logger.info("Resetting service factory")
        cls._ocr_service = None
        cls._evaluation_service = None
        cls._annotation_service = None
        cls._orchestrator = None