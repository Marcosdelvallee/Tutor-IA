"""
=============================================================================
Tutor IA Socrático - Cargador de PDFs
=============================================================================
Módulo para la extracción de texto desde archivos PDF usando PyMuPDF (fitz).

Características:
- Extracción página por página con metadatos
- Manejo robusto de errores (corrupto, encriptado, vacío)
- Soporte para carga individual y por lotes
- Preservación de estructura del documento
=============================================================================
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
import logging

from src.utils.exceptions import (
    PDFNotFoundError,
    PDFCorruptedError,
    PDFEncryptedError,
    PDFEmptyError
)

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """
    Representa una página individual de un PDF con su contenido y metadatos.
    
    Attributes:
        content: Texto extraído de la página
        page_number: Número de página (1-indexed)
        source_file: Nombre del archivo fuente
        source_path: Ruta completa al archivo
        char_count: Número de caracteres en la página
    """
    content: str
    page_number: int
    source_file: str
    source_path: str
    char_count: int = field(init=False)
    
    def __post_init__(self):
        self.char_count = len(self.content)
    
    def to_dict(self) -> dict:
        """Convierte la página a diccionario para serialización."""
        return {
            "content": self.content,
            "page_number": self.page_number,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "char_count": self.char_count
        }


@dataclass
class PDFDocument:
    """
    Representa un documento PDF completo con todas sus páginas.
    
    Attributes:
        file_path: Ruta al archivo PDF
        pages: Lista de páginas extraídas
        total_pages: Número total de páginas
        title: Título del documento (si está disponible en metadatos)
        author: Autor del documento (si está disponible)
    """
    file_path: str
    pages: list[PDFPage]
    total_pages: int
    title: str = ""
    author: str = ""
    
    @property
    def full_text(self) -> str:
        """Retorna todo el texto del documento concatenado."""
        return "\n\n".join(page.content for page in self.pages)
    
    @property
    def total_chars(self) -> int:
        """Retorna el total de caracteres del documento."""
        return sum(page.char_count for page in self.pages)
    
    def __repr__(self) -> str:
        return f"PDFDocument('{Path(self.file_path).name}', {self.total_pages} páginas, {self.total_chars} chars)"


class PDFLoader:
    """
    Cargador de documentos PDF con extracción de texto robusta.
    
    Utiliza PyMuPDF (fitz) para una extracción eficiente y de alta calidad.
    Maneja casos de error comunes: archivos inexistentes, corruptos, 
    encriptados o sin texto extraíble.
    
    Example:
        >>> loader = PDFLoader()
        >>> doc = loader.load("documento.pdf")
        >>> print(doc.full_text)
        
        # Carga por lotes
        >>> docs = loader.load_directory("data/pdfs/")
    """
    
    def __init__(self, extract_images: bool = False):
        """
        Inicializa el cargador de PDFs.
        
        Args:
            extract_images: Si True, extrae también texto de imágenes (OCR).
                           Requiere tesseract instalado. Default: False
        """
        self.extract_images = extract_images
        logger.info("PDFLoader inicializado")
    
    def load(self, file_path: str | Path) -> PDFDocument:
        """
        Carga un archivo PDF y extrae su contenido.
        
        Args:
            file_path: Ruta al archivo PDF
            
        Returns:
            PDFDocument con el contenido extraído
            
        Raises:
            PDFNotFoundError: Si el archivo no existe
            PDFEncryptedError: Si el PDF está protegido con contraseña
            PDFCorruptedError: Si el PDF está dañado o es ilegible
            PDFEmptyError: Si el PDF no contiene texto extraíble
        """
        path = Path(file_path)
        
        # === Validación de existencia ===
        if not path.exists():
            logger.error(f"Archivo no encontrado: {path}")
            raise PDFNotFoundError(str(path))
        
        if not path.suffix.lower() == ".pdf":
            logger.warning(f"Extensión no es .pdf: {path}")
        
        logger.info(f"Cargando PDF: {path.name}")
        
        try:
            # === Abrir documento con PyMuPDF ===
            doc = fitz.open(str(path))
            
            # === Verificar encriptación ===
            if doc.is_encrypted:
                doc.close()
                logger.error(f"PDF encriptado: {path.name}")
                raise PDFEncryptedError(str(path))
            
            # === Extraer metadatos ===
            metadata = doc.metadata or {}
            title = metadata.get("title", "")
            author = metadata.get("author", "")
            
            # === Extraer texto página por página ===
            pages: list[PDFPage] = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extraer texto con opciones de formato
                # flags: preservar whitespace, no desglosar ligaduras
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                
                # Limpiar texto extraído
                text = self._clean_text(text)
                
                if text.strip():  # Solo agregar páginas con contenido
                    pdf_page = PDFPage(
                        content=text,
                        page_number=page_num + 1,  # 1-indexed
                        source_file=path.name,
                        source_path=str(path.absolute())
                    )
                    pages.append(pdf_page)
            
            doc.close()
            
            # === Validar que hay contenido ===
            if not pages:
                logger.warning(f"PDF sin texto extraíble: {path.name}")
                raise PDFEmptyError(str(path))
            
            pdf_document = PDFDocument(
                file_path=str(path),
                pages=pages,
                total_pages=len(pages),
                title=title,
                author=author
            )
            
            logger.info(f"PDF cargado exitosamente: {pdf_document}")
            return pdf_document
            
        except fitz.FileDataError as e:
            logger.error(f"PDF corrupto: {path.name} - {str(e)}")
            raise PDFCorruptedError(str(path), str(e))
        
        except (PDFEncryptedError, PDFEmptyError):
            # Re-lanzar excepciones propias
            raise
        
        except Exception as e:
            logger.error(f"Error inesperado cargando PDF: {path.name} - {str(e)}")
            raise PDFCorruptedError(str(path), str(e))
    
    def load_directory(
        self, 
        directory: str | Path,
        recursive: bool = False
    ) -> Generator[PDFDocument, None, None]:
        """
        Carga todos los PDFs de un directorio.
        
        Args:
            directory: Ruta al directorio con PDFs
            recursive: Si True, busca también en subdirectorios
            
        Yields:
            PDFDocument por cada PDF cargado exitosamente
            
        Note:
            Los errores de carga individual se registran pero no detienen
            el proceso de carga del resto de archivos.
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directorio no encontrado: {dir_path}")
            return
        
        # Patrón de búsqueda
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(dir_path.glob(pattern))
        
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF en {dir_path}")
        
        successful = 0
        failed = 0
        
        for pdf_path in pdf_files:
            try:
                yield self.load(pdf_path)
                successful += 1
            except Exception as e:
                logger.warning(f"Error cargando {pdf_path.name}: {str(e)}")
                failed += 1
                continue
        
        logger.info(f"Carga completada: {successful} exitosos, {failed} fallidos")
    
    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído del PDF.
        
        - Normaliza saltos de línea
        - Elimina caracteres de control
        - Colapsa espacios múltiples
        
        Args:
            text: Texto crudo del PDF
            
        Returns:
            Texto limpio y normalizado
        """
        if not text:
            return ""
        
        # Normalizar saltos de línea
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Eliminar caracteres de control (excepto newline y tab)
        text = "".join(
            char for char in text 
            if char == "\n" or char == "\t" or not (0 <= ord(char) < 32)
        )
        
        # Colapsar múltiples espacios en uno solo
        import re
        text = re.sub(r"[ \t]+", " ", text)
        
        # Colapsar más de 2 newlines consecutivos
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    loader = PDFLoader()
    
    if len(sys.argv) > 1:
        # Cargar archivo específico
        try:
            doc = loader.load(sys.argv[1])
            print(f"\n📄 {doc}")
            print(f"   Título: {doc.title or 'N/A'}")
            print(f"   Autor: {doc.author or 'N/A'}")
            print(f"\n📝 Primeros 500 caracteres:")
            print(doc.full_text[:500])
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Uso: python pdf_loader.py <ruta_al_pdf>")
