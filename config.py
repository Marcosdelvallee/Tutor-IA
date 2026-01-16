"""
=============================================================================
Tutor IA Socrático - Configuración Central
=============================================================================
Este módulo centraliza toda la configuración del sistema, incluyendo:
- Variables de entorno y API keys
- Parámetros de chunking para RAG
- Configuración de modelos LLM
- Rutas de almacenamiento
=============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Cargar variables de entorno desde .env
load_dotenv()


# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
class PathConfig:
    """Configuración de rutas del sistema de archivos."""
    
    # Directorio raíz del proyecto
    ROOT_DIR: Path = Path(__file__).parent.resolve()
    
    # Directorio de datos
    DATA_DIR: Path = ROOT_DIR / "data"
    PDF_DIR: Path = DATA_DIR / "pdfs"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"
    PROFILES_DIR: Path = DATA_DIR / "profiles"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Crea los directorios necesarios si no existen."""
        for directory in [cls.DATA_DIR, cls.PDF_DIR, cls.CHROMA_DB_DIR, cls.PROFILES_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURACIÓN DE CHUNKING (RAG)
# =============================================================================
class ChunkingConfig(BaseModel):
    """
    Parámetros para la estrategia de fragmentación de documentos.
    
    Estrategia: Fragmentos de 1000 tokens con overlap de 200 para mantener
    el contexto entre chunks adyacentes. Esto es crucial para:
    - Preservar oraciones completas en los límites
    - Mantener coherencia semántica
    - Evitar pérdida de información contextual
    """
    
    chunk_size: int = Field(
        default=1000,
        description="Tamaño máximo de cada fragmento en tokens"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Tokens de solapamiento entre fragmentos consecutivos"
    )
    # Separadores para división inteligente (orden de prioridad)
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separadores para split jerárquico"
    )


# =============================================================================
# CONFIGURACIÓN DE MODELOS LLM
# =============================================================================
class ModelConfig(BaseModel):
    """Configuración de modelos de lenguaje e embeddings."""
    
    # API Keys (desde variables de entorno)
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    
    # Modelo principal para chat/razonamiento
    llm_model: str = Field(
        default="gemini-1.5-pro",
        description="Modelo multimodal para el tutor"
    )
    
    # Modelo de embeddings
    embedding_model: str = Field(
        default="models/embedding-001",
        description="Modelo para generar embeddings de Google"
    )
    
    # Parámetros de generación
    temperature: float = Field(
        default=0.7,
        description="Creatividad del modelo (0.0-1.0)"
    )
    max_output_tokens: int = Field(
        default=2048,
        description="Máximo de tokens en la respuesta"
    )


# =============================================================================
# CONFIGURACIÓN DE CHROMADB
# =============================================================================
class ChromaConfig(BaseModel):
    """Configuración de la base de datos vectorial ChromaDB."""
    
    collection_name: str = Field(
        default="tutor_documents",
        description="Nombre de la colección principal"
    )
    # Función de distancia para similitud
    # "cosine" es ideal para embeddings normalizados
    distance_function: str = Field(
        default="cosine",
        description="Métrica de distancia: cosine, l2, ip"
    )
    # Número de resultados por búsqueda
    n_results: int = Field(
        default=5,
        description="Documentos a recuperar por query"
    )


# =============================================================================
# CONFIGURACIÓN DEL TUTOR SOCRÁTICO
# =============================================================================
class TutorConfig(BaseModel):
    """Configuración del comportamiento del agente tutor."""
    
    # Intentos antes de revelar la solución
    max_socratic_attempts: int = Field(
        default=3,
        description="Preguntas guía antes de dar la respuesta"
    )
    # Palabras clave que disparan respuesta directa
    direct_answer_keywords: list[str] = Field(
        default=["muéstrame la solución", "dame la respuesta", "no entiendo nada"],
        description="Frases que saltan el método socrático"
    )


# =============================================================================
# INSTANCIAS GLOBALES DE CONFIGURACIÓN
# =============================================================================
paths = PathConfig()
chunking = ChunkingConfig()
model = ModelConfig()
chroma = ChromaConfig()
tutor = TutorConfig()

# Asegurar que los directorios existan al importar el módulo
paths.ensure_directories()


# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# =============================================================================
def validate_config() -> dict[str, bool]:
    """
    Valida que la configuración esencial esté presente.
    
    Returns:
        Dict con el estado de cada validación
    """
    validations = {
        "google_api_key": bool(model.google_api_key),
        "openai_api_key": bool(model.openai_api_key),
        "data_directories": all([
            paths.DATA_DIR.exists(),
            paths.PDF_DIR.exists(),
            paths.CHROMA_DB_DIR.exists()
        ]),
    }
    return validations


if __name__ == "__main__":
    # Test de configuración
    print("🔧 Validando configuración...")
    status = validate_config()
    for key, valid in status.items():
        icon = "✅" if valid else "❌"
        print(f"  {icon} {key}: {'OK' if valid else 'FALTA'}")
