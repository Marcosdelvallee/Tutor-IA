"""Módulo de Visión Multimodal."""
from .image_processor import ImageProcessor, ProcessedImage
from .correction_logic import SolutionCorrector, CorrectionResult, ErrorType

__all__ = [
    "ImageProcessor",
    "ProcessedImage",
    "SolutionCorrector",
    "CorrectionResult", 
    "ErrorType"
]
