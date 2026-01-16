"""
=============================================================================
Tutor IA Socrático - Fragmentador de Documentos (Chunker)
=============================================================================
Implementa la estrategia de chunking para RAG con:
- Fragmentación basada en tokens (no caracteres)
- Overlap configurable para preservar contexto
- División inteligente respetando límites semánticos

Estrategia por defecto: 1000 tokens con 200 de overlap
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Callable
import tiktoken

from src.ingestion.pdf_loader import PDFDocument, PDFPage
from src.utils.exceptions import EmptyDocumentError

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Representa un fragmento de documento para el vector store.
    
    Attributes:
        content: Texto del fragmento
        chunk_index: Índice del chunk dentro del documento (0-indexed)
        token_count: Número de tokens en el fragmento
        source_file: Nombre del archivo fuente
        source_path: Ruta completa al archivo
        page_numbers: Lista de páginas de donde proviene el chunk
        metadata: Metadatos adicionales para búsqueda
    """
    content: str
    chunk_index: int
    token_count: int
    source_file: str
    source_path: str
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Genera un ID único para el chunk."""
        return f"{self.source_file}::chunk_{self.chunk_index}"
    
    def to_dict(self) -> dict:
        """Convierte el chunk a diccionario para almacenamiento."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "page_numbers": self.page_numbers,
            **self.metadata
        }


class DocumentChunker:
    """
    Fragmentador de documentos con estrategia de tokens y overlap.
    
    Utiliza tiktoken para conteo preciso de tokens compatible con
    modelos OpenAI/GPT. La estrategia de overlap asegura que no se
    pierda contexto en los límites de los fragmentos.
    
    Example:
        >>> chunker = DocumentChunker(chunk_size=1000, overlap=200)
        >>> chunks = chunker.chunk_document(pdf_document)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {chunk.token_count} tokens")
    """
    
    # Separadores por prioridad (de más a menos preferido)
    DEFAULT_SEPARATORS = [
        "\n\n",      # Párrafos
        "\n",        # Líneas
        ". ",        # Oraciones
        "? ",        # Preguntas  
        "! ",        # Exclamaciones
        "; ",        # Cláusulas
        ", ",        # Frases
        " ",         # Palabras
        ""           # Caracteres (último recurso)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        encoding_name: str = "cl100k_base",
        separators: list[str] = None
    ):
        """
        Inicializa el chunker con parámetros configurables.
        
        Args:
            chunk_size: Tamaño máximo de cada chunk en tokens (default: 1000)
            overlap: Tokens de solapamiento entre chunks (default: 200)
            encoding_name: Nombre del encoding de tiktoken (default: cl100k_base)
                          - cl100k_base: GPT-4, GPT-3.5-turbo
                          - p50k_base: GPT-3
            separators: Lista de separadores para división inteligente
        
        Raises:
            ValueError: Si overlap >= chunk_size
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) debe ser menor que chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        
        # Inicializar tokenizador
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(f"Encoding {encoding_name} no disponible, usando cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(
            f"DocumentChunker inicializado: "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Cuenta los tokens en un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Número de tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def chunk_document(
        self, 
        document: PDFDocument,
        additional_metadata: dict = None
    ) -> list[DocumentChunk]:
        """
        Fragmenta un PDFDocument completo en chunks.
        
        Args:
            document: Documento PDF cargado
            additional_metadata: Metadatos adicionales para cada chunk
            
        Returns:
            Lista de DocumentChunk
            
        Raises:
            EmptyDocumentError: Si el documento no tiene contenido
        """
        if not document.pages:
            raise EmptyDocumentError()
        
        logger.info(f"Fragmentando documento: {document.file_path}")
        
        # Preparar texto con marcadores de página para tracking
        pages_text = []
        page_boundaries = []  # Índices donde empieza cada página
        
        current_pos = 0
        for page in document.pages:
            page_boundaries.append({
                "page_number": page.page_number,
                "start": current_pos,
                "end": current_pos + len(page.content)
            })
            pages_text.append(page.content)
            current_pos += len(page.content) + 2  # +2 por "\n\n"
        
        full_text = "\n\n".join(pages_text)
        
        # Fragmentar el texto
        raw_chunks = self._split_text(full_text)
        
        # Crear DocumentChunks con metadatos
        chunks: list[DocumentChunk] = []
        char_position = 0
        
        for idx, chunk_text in enumerate(raw_chunks):
            # Determinar páginas de origen
            chunk_end = char_position + len(chunk_text)
            page_nums = self._get_page_numbers(
                char_position, 
                chunk_end, 
                page_boundaries
            )
            
            metadata = {
                "title": document.title,
                "author": document.author,
                **(additional_metadata or {})
            }
            
            chunk = DocumentChunk(
                content=chunk_text,
                chunk_index=idx,
                token_count=self.count_tokens(chunk_text),
                source_file=Path(document.file_path).name if hasattr(Path, '__call__') else document.file_path.split('/')[-1].split('\\')[-1],
                source_path=document.file_path,
                page_numbers=page_nums,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Avanzar posición (restando overlap para chunks posteriores)
            if idx < len(raw_chunks) - 1:
                # Calcular overlap en caracteres aproximado
                overlap_chars = int(len(chunk_text) * (self.overlap / self.chunk_size))
                char_position = chunk_end - overlap_chars
            else:
                char_position = chunk_end
        
        logger.info(
            f"Documento fragmentado: {len(chunks)} chunks, "
            f"promedio {sum(c.token_count for c in chunks) // len(chunks)} tokens/chunk"
        )
        
        return chunks
    
    def chunk_text(self, text: str, source_name: str = "unknown") -> list[DocumentChunk]:
        """
        Fragmenta un texto plano en chunks.
        
        Útil para fragmentar texto que no proviene de un PDF.
        
        Args:
            text: Texto a fragmentar
            source_name: Nombre identificador del origen
            
        Returns:
            Lista de DocumentChunk
        """
        if not text or not text.strip():
            raise EmptyDocumentError()
        
        raw_chunks = self._split_text(text)
        
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = DocumentChunk(
                content=chunk_text,
                chunk_index=idx,
                token_count=self.count_tokens(chunk_text),
                source_file=source_name,
                source_path=source_name,
                page_numbers=[],
                metadata={}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str) -> list[str]:
        """
        Divide el texto en fragmentos respetando el tamaño máximo.
        
        Implementa división recursiva: intenta dividir por el separador
        de mayor prioridad, si el resultado es muy grande, recurre al
        siguiente separador.
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de fragmentos de texto
        """
        if not text.strip():
            return []
        
        # Si el texto cabe en un chunk, retornarlo
        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Encontrar el mejor separador
        chunks = []
        for separator in self.separators:
            if separator and separator in text:
                splits = text.split(separator)
                
                # Reconstruir fragmentos con overlap
                current_chunk = ""
                
                for split in splits:
                    # Agregar separador de vuelta (excepto para "")
                    segment = split + separator if separator else split
                    
                    # Verificar si cabe en el chunk actual
                    potential = current_chunk + segment
                    if self.count_tokens(potential) <= self.chunk_size:
                        current_chunk = potential
                    else:
                        # Guardar chunk actual si tiene contenido
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        # Si el segmento individual es muy grande, recursión
                        if self.count_tokens(segment) > self.chunk_size:
                            # Remover este separador y probar el siguiente
                            sub_chunks = self._split_text(segment)
                            if sub_chunks:
                                # Aplicar overlap con el chunk anterior
                                if chunks:
                                    overlap_text = self._get_overlap_text(chunks[-1])
                                    sub_chunks[0] = overlap_text + sub_chunks[0]
                                chunks.extend(sub_chunks)
                                current_chunk = ""
                            else:
                                current_chunk = segment
                        else:
                            # Aplicar overlap del chunk anterior
                            if chunks:
                                overlap_text = self._get_overlap_text(chunks[-1])
                                current_chunk = overlap_text + segment
                            else:
                                current_chunk = segment
                
                # Agregar último chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if chunks:
                    return chunks
        
        # Fallback: dividir por caracteres (último recurso)
        return self._split_by_tokens(text)
    
    def _split_by_tokens(self, text: str) -> list[str]:
        """
        División forzada por número de tokens (último recurso).
        
        Se usa cuando ningún separador funciona.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Aplicar overlap
            start = end - self.overlap if end < len(tokens) else end
        
        return chunks
    
    def _get_overlap_text(self, previous_chunk: str) -> str:
        """
        Extrae el texto de overlap del chunk anterior.
        
        Args:
            previous_chunk: Texto del chunk anterior
            
        Returns:
            Texto correspondiente al overlap
        """
        tokens = self.tokenizer.encode(previous_chunk)
        if len(tokens) <= self.overlap:
            return previous_chunk + " "
        
        overlap_tokens = tokens[-self.overlap:]
        return self.tokenizer.decode(overlap_tokens) + " "
    
    def _get_page_numbers(
        self, 
        start: int, 
        end: int, 
        boundaries: list[dict]
    ) -> list[int]:
        """
        Determina qué páginas están incluidas en un rango de caracteres.
        
        Args:
            start: Posición inicial del chunk
            end: Posición final del chunk
            boundaries: Lista de límites de página
            
        Returns:
            Lista de números de página
        """
        pages = []
        for boundary in boundaries:
            # Si hay solapamiento entre el chunk y la página
            if boundary["start"] < end and boundary["end"] > start:
                pages.append(boundary["page_number"])
        return pages


# Importar Path para el source_file
from pathlib import Path


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Demo con texto de ejemplo
    demo_text = """
    La inteligencia artificial (IA) es un campo de la informática que se centra 
    en la creación de sistemas capaces de realizar tareas que normalmente requieren 
    inteligencia humana. Esto incluye el aprendizaje, el razonamiento, la percepción 
    y la comprensión del lenguaje natural.
    
    El aprendizaje automático (Machine Learning) es una subdisciplina de la IA que 
    permite a las máquinas aprender de los datos sin ser programadas explícitamente. 
    Los algoritmos de ML pueden identificar patrones en grandes conjuntos de datos 
    y hacer predicciones o tomar decisiones basadas en esos patrones.
    
    El aprendizaje profundo (Deep Learning) es una rama del ML que utiliza redes 
    neuronales artificiales con múltiples capas para modelar abstracciones de alto 
    nivel en los datos. Ha revolucionado campos como la visión por computadora, 
    el procesamiento del lenguaje natural y el reconocimiento de voz.
    """
    
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(demo_text, source_name="demo.txt")
    
    print(f"\n📦 Generados {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"--- Chunk {chunk.chunk_index} ({chunk.token_count} tokens) ---")
        print(chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content)
        print()
