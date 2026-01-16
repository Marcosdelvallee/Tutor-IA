"""
=============================================================================
Tutor IA Socrático - Excepciones Personalizadas
=============================================================================
Define excepciones específicas para el manejo de errores del sistema.
Esto permite un control granular y mensajes de error informativos.
=============================================================================
"""


class TutorBaseException(Exception):
    """Excepción base para todas las excepciones del Tutor IA."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Detalles: {self.details}"
        return self.message


# =============================================================================
# EXCEPCIONES DE CARGA DE PDF
# =============================================================================
class PDFLoadError(TutorBaseException):
    """Error al cargar un archivo PDF."""
    pass


class PDFNotFoundError(PDFLoadError):
    """El archivo PDF especificado no existe."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"Archivo PDF no encontrado: {file_path}",
            details={"file_path": file_path}
        )


class PDFCorruptedError(PDFLoadError):
    """El archivo PDF está corrupto o dañado."""
    
    def __init__(self, file_path: str, original_error: str = None):
        super().__init__(
            message=f"PDF corrupto o ilegible: {file_path}",
            details={
                "file_path": file_path,
                "original_error": original_error
            }
        )


class PDFEncryptedError(PDFLoadError):
    """El archivo PDF está encriptado y requiere contraseña."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"PDF encriptado (requiere contraseña): {file_path}",
            details={"file_path": file_path}
        )


class PDFEmptyError(PDFLoadError):
    """El PDF no contiene texto extraíble."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"PDF sin texto extraíble (posiblemente escaneado): {file_path}",
            details={"file_path": file_path}
        )


# =============================================================================
# EXCEPCIONES DE EMBEDDINGS
# =============================================================================
class EmbeddingError(TutorBaseException):
    """Error al generar embeddings."""
    pass


class EmbeddingAPIError(EmbeddingError):
    """Error de comunicación con la API de embeddings."""
    
    def __init__(self, provider: str, original_error: str = None):
        super().__init__(
            message=f"Error de API de embeddings ({provider})",
            details={
                "provider": provider,
                "original_error": original_error
            }
        )


class EmbeddingRateLimitError(EmbeddingError):
    """Se excedió el límite de requests de la API."""
    
    def __init__(self, provider: str, retry_after: int = None):
        super().__init__(
            message=f"Rate limit excedido para {provider}",
            details={
                "provider": provider,
                "retry_after_seconds": retry_after
            }
        )


# =============================================================================
# EXCEPCIONES DE VECTOR STORE
# =============================================================================
class VectorStoreError(TutorBaseException):
    """Error relacionado con ChromaDB."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """No se puede conectar a la base de datos vectorial."""
    
    def __init__(self, db_path: str, original_error: str = None):
        super().__init__(
            message=f"Error de conexión a ChromaDB: {db_path}",
            details={
                "db_path": db_path,
                "original_error": original_error
            }
        )


class CollectionNotFoundError(VectorStoreError):
    """La colección especificada no existe."""
    
    def __init__(self, collection_name: str):
        super().__init__(
            message=f"Colección no encontrada: {collection_name}",
            details={"collection_name": collection_name}
        )


class DocumentNotFoundError(VectorStoreError):
    """El documento especificado no existe en la colección."""
    
    def __init__(self, document_id: str, collection_name: str = None):
        super().__init__(
            message=f"Documento no encontrado: {document_id}",
            details={
                "document_id": document_id,
                "collection_name": collection_name
            }
        )


# =============================================================================
# EXCEPCIONES DE CHUNKING
# =============================================================================
class ChunkingError(TutorBaseException):
    """Error durante la fragmentación de documentos."""
    pass


class EmptyDocumentError(ChunkingError):
    """El documento no tiene contenido para fragmentar."""
    
    def __init__(self, document_id: str = None):
        super().__init__(
            message="Documento vacío, no hay contenido para fragmentar",
            details={"document_id": document_id}
        )


# =============================================================================
# EXCEPCIONES DE CONFIGURACIÓN
# =============================================================================
class ConfigurationError(TutorBaseException):
    """Error de configuración del sistema."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Falta una API key requerida."""
    
    def __init__(self, key_name: str):
        super().__init__(
            message=f"API key faltante: {key_name}",
            details={
                "key_name": key_name,
                "hint": f"Configura {key_name} en tu archivo .env"
            }
        )
