"""
=============================================================================
Tutor IA Socrático - Procesador de Imágenes
=============================================================================
Procesa imágenes de resoluciones hechas a mano para:
- Validar legibilidad
- Preparar para análisis multimodal
- Extraer metadatos de imagen
=============================================================================
"""

import logging
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


class ImageQuality(Enum):
    """Calidad de imagen detectada."""
    HIGH = "high"           # Clara y legible
    MEDIUM = "medium"       # Aceptable con algunos problemas
    LOW = "low"             # Difícil de leer
    UNREADABLE = "unreadable"  # No se puede procesar


@dataclass
class ProcessedImage:
    """
    Imagen procesada lista para análisis.
    
    Attributes:
        original_path: Ruta original de la imagen
        base64_data: Imagen en formato base64
        mime_type: Tipo MIME de la imagen
        width: Ancho en píxeles
        height: Alto en píxeles
        quality: Calidad estimada
        metadata: Metadatos adicionales
    """
    original_path: Optional[str]
    base64_data: str
    mime_type: str
    width: int
    height: int
    quality: ImageQuality = ImageQuality.MEDIUM
    metadata: dict = field(default_factory=dict)
    
    @property
    def aspect_ratio(self) -> float:
        """Ratio de aspecto de la imagen."""
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def is_readable(self) -> bool:
        """Indica si la imagen es legible."""
        return self.quality != ImageQuality.UNREADABLE
    
    def to_langchain_format(self) -> dict:
        """
        Convierte a formato compatible con LangChain multimodal.
        
        Returns:
            Dict con formato de imagen para mensajes multimodales
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{self.base64_data}"
            }
        }


class ImageProcessor:
    """
    Procesador de imágenes para análisis multimodal.
    
    Prepara imágenes de resoluciones escritas a mano para ser
    analizadas por modelos multimodales como Gemini o GPT-4o.
    
    Example:
        >>> processor = ImageProcessor()
        >>> 
        >>> # Desde archivo
        >>> image = processor.process_file("solucion.jpg")
        >>> 
        >>> # Desde bytes
        >>> image = processor.process_bytes(image_bytes, "image/jpeg")
        >>> 
        >>> # Verificar calidad
        >>> if image.is_readable:
        ...     # Enviar a análisis
    """
    
    # Formatos soportados
    SUPPORTED_FORMATS = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", 
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    
    # Límites de tamaño
    MIN_DIMENSION = 100      # Mínimo px para ser legible
    MAX_DIMENSION = 4096     # Máximo px antes de redimensionar
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    
    def __init__(
        self,
        max_dimension: int = 2048,
        quality: int = 85
    ):
        """
        Inicializa el procesador.
        
        Args:
            max_dimension: Dimensión máxima para redimensionar
            quality: Calidad de compresión JPEG (1-100)
        """
        self.max_dimension = max_dimension
        self.quality = quality
        logger.info(f"ImageProcessor inicializado: max_dim={max_dimension}")
    
    def process_file(self, file_path: str | Path) -> ProcessedImage:
        """
        Procesa una imagen desde archivo.
        
        Args:
            file_path: Ruta al archivo de imagen
            
        Returns:
            ProcessedImage lista para análisis
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {path}")
        
        # Verificar formato
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato no soportado: {suffix}. "
                f"Formatos válidos: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Verificar tamaño de archivo
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"Archivo muy grande: {file_size / 1024 / 1024:.1f}MB. "
                f"Máximo: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Cargar imagen
        with open(path, "rb") as f:
            image_bytes = f.read()
        
        mime_type = self.SUPPORTED_FORMATS[suffix]
        
        return self._process_bytes(
            image_bytes, 
            mime_type,
            original_path=str(path)
        )
    
    def process_bytes(
        self, 
        image_bytes: bytes, 
        mime_type: str
    ) -> ProcessedImage:
        """
        Procesa una imagen desde bytes.
        
        Args:
            image_bytes: Bytes de la imagen
            mime_type: Tipo MIME
            
        Returns:
            ProcessedImage lista para análisis
        """
        return self._process_bytes(image_bytes, mime_type)
    
    def _process_bytes(
        self,
        image_bytes: bytes,
        mime_type: str,
        original_path: Optional[str] = None
    ) -> ProcessedImage:
        """
        Procesa bytes de imagen internamente.
        
        Args:
            image_bytes: Bytes de la imagen
            mime_type: Tipo MIME
            original_path: Ruta original (opcional)
            
        Returns:
            ProcessedImage procesada
        """
        try:
            # Abrir con PIL
            img = Image.open(BytesIO(image_bytes))
            
            # Obtener dimensiones originales
            original_width, original_height = img.size
            
            # Estimar calidad inicial
            quality = self._estimate_quality(img, original_width, original_height)
            
            # Redimensionar si es necesario
            if max(original_width, original_height) > self.max_dimension:
                img = self._resize_image(img)
                logger.debug(f"Imagen redimensionada: {img.size}")
            
            # Convertir a RGB si tiene canal alpha
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convertir a base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=self.quality)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            final_width, final_height = img.size
            
            processed = ProcessedImage(
                original_path=original_path,
                base64_data=base64_data,
                mime_type="image/jpeg",  # Siempre JPEG después de procesar
                width=final_width,
                height=final_height,
                quality=quality,
                metadata={
                    "original_width": original_width,
                    "original_height": original_height,
                    "original_mime_type": mime_type,
                    "was_resized": max(original_width, original_height) > self.max_dimension
                }
            )
            
            logger.info(
                f"Imagen procesada: {final_width}x{final_height}, "
                f"calidad={quality.value}"
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            raise ValueError(f"No se pudo procesar la imagen: {e}")
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Redimensiona la imagen manteniendo aspecto.
        
        Args:
            img: Imagen PIL
            
        Returns:
            Imagen redimensionada
        """
        width, height = img.size
        
        if width > height:
            new_width = self.max_dimension
            new_height = int(height * (self.max_dimension / width))
        else:
            new_height = self.max_dimension
            new_width = int(width * (self.max_dimension / height))
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _estimate_quality(
        self, 
        img: Image.Image,
        width: int,
        height: int
    ) -> ImageQuality:
        """
        Estima la calidad/legibilidad de la imagen.
        
        Criterios:
        - Resolución mínima
        - Contraste (varianza de píxeles)
        - Nitidez básica
        
        Args:
            img: Imagen PIL
            width: Ancho original
            height: Alto original
            
        Returns:
            ImageQuality estimada
        """
        # Verificar resolución mínima
        if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
            return ImageQuality.UNREADABLE
        
        try:
            # Convertir a escala de grises para análisis
            gray = img.convert('L')
            
            # Calcular varianza (proxy de contraste)
            import statistics
            pixels = list(gray.getdata())
            
            # Muestrear para eficiencia
            sample_size = min(10000, len(pixels))
            sample = pixels[::max(1, len(pixels) // sample_size)]
            
            if len(sample) < 100:
                return ImageQuality.LOW
            
            variance = statistics.variance(sample)
            
            # Clasificar por varianza
            if variance < 500:
                return ImageQuality.LOW  # Bajo contraste
            elif variance < 1500:
                return ImageQuality.MEDIUM
            else:
                return ImageQuality.HIGH
                
        except Exception as e:
            logger.warning(f"Error estimando calidad: {e}")
            return ImageQuality.MEDIUM  # Default
    
    def validate_for_analysis(self, image: ProcessedImage) -> tuple[bool, str]:
        """
        Valida si la imagen es apta para análisis.
        
        Args:
            image: Imagen procesada
            
        Returns:
            Tuple de (es_válida, mensaje)
        """
        if not image.is_readable:
            return False, "La imagen tiene muy baja calidad o resolución para ser analizada."
        
        if image.width < self.MIN_DIMENSION or image.height < self.MIN_DIMENSION:
            return False, f"La imagen es muy pequeña. Mínimo requerido: {self.MIN_DIMENSION}px"
        
        if image.quality == ImageQuality.LOW:
            return True, "⚠️ La imagen tiene baja calidad. El análisis podría ser impreciso."
        
        return True, "✅ Imagen apta para análisis."


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    processor = ImageProcessor()
    
    if len(sys.argv) > 1:
        try:
            image = processor.process_file(sys.argv[1])
            print(f"\n📷 Imagen procesada:")
            print(f"   Tamaño: {image.width}x{image.height}")
            print(f"   Calidad: {image.quality.value}")
            print(f"   Legible: {image.is_readable}")
            print(f"   Base64 length: {len(image.base64_data)} chars")
            
            is_valid, message = processor.validate_for_analysis(image)
            print(f"   Validación: {message}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Uso: python image_processor.py <ruta_imagen>")
