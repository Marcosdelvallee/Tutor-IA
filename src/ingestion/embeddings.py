"""
=============================================================================
Tutor IA Socrático - Generador de Embeddings
=============================================================================
Generación de embeddings vectoriales para documentos usando:
- Google Generative AI (predeterminado)
- OpenAI Embeddings (fallback)

Los embeddings convierten texto en vectores numéricos que capturan
su significado semántico, permitiendo búsquedas por similitud.
=============================================================================
"""

import logging
from typing import Optional
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

from src.utils.exceptions import (
    EmbeddingAPIError,
    MissingAPIKeyError
)

# Configuración de logging
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interfaz abstracta para proveedores de embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de documentos."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Genera embedding para una consulta individual."""
        pass


class EmbeddingGenerator:
    """
    Generador de embeddings con soporte para múltiples proveedores.
    
    Orden de preferencia:
    1. HuggingFace local (gratuito, sin rate limits)
    2. Google Generative AI
    3. OpenAI Embeddings
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> 
        >>> # Para documentos (batch)
        >>> embeddings = generator.embed_documents(["texto 1", "texto 2"])
        >>> 
        >>> # Para queries (individual)
        >>> query_embedding = generator.embed_query("¿Qué es una integral?")
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        google_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Inicializa el generador de embeddings.
        
        Args:
            provider: "huggingface", "google" o "openai" (default: "huggingface")
            google_api_key: API key de Google (o desde env GOOGLE_API_KEY)
            openai_api_key: API key de OpenAI (o desde env OPENAI_API_KEY)
            model_name: Nombre del modelo de embeddings (opcional)
        
        Raises:
            MissingAPIKeyError: Si no se proporciona API key para el proveedor
        """
        self.provider = provider.lower()
        self._embeddings: Optional[Embeddings] = None
        
        # Intentar inicializar el proveedor principal
        if self.provider == "huggingface":
            self._init_huggingface(model_name)
        elif self.provider == "google":
            self._init_google(google_api_key, model_name)
        elif self.provider == "openai":
            self._init_openai(openai_api_key, model_name)
        else:
            logger.warning(f"Proveedor desconocido: {provider}, usando HuggingFace local")
            self.provider = "huggingface"
            self._init_huggingface(model_name)
        
        logger.info(f"EmbeddingGenerator inicializado con proveedor: {self.provider}")
    
    def _init_huggingface(self, model_name: Optional[str]) -> None:
        """Inicializa embeddings locales de HuggingFace."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.debug("HuggingFace Embeddings inicializado (local)")
            
        except ImportError as e:
            logger.error(f"langchain-huggingface no instalado: {e}")
            raise EmbeddingAPIError("huggingface", str(e))
    
    def _init_google(self, api_key: Optional[str], model_name: Optional[str]) -> None:
        """Inicializa embeddings de Google Generative AI."""
        import os
        
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise MissingAPIKeyError("GOOGLE_API_KEY")
        
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name or "models/embedding-001",
                google_api_key=key
            )
            logger.debug("Google Generative AI Embeddings inicializado")
            
        except ImportError as e:
            logger.error(f"langchain-google-genai no instalado: {e}")
            raise EmbeddingAPIError("google", str(e))
        except Exception as e:
            logger.error(f"Error inicializando Google embeddings: {e}")
            raise EmbeddingAPIError("google", str(e))
    
    def _init_openai(self, api_key: Optional[str], model_name: Optional[str]) -> None:
        """Inicializa embeddings de OpenAI."""
        import os
        
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise MissingAPIKeyError("OPENAI_API_KEY")
        
        try:
            from langchain_openai import OpenAIEmbeddings
            
            self._embeddings = OpenAIEmbeddings(
                model=model_name or "text-embedding-3-small",
                openai_api_key=key
            )
            logger.debug("OpenAI Embeddings inicializado")
            
        except ImportError as e:
            logger.error(f"langchain-openai no instalado: {e}")
            raise EmbeddingAPIError("openai", str(e))
        except Exception as e:
            logger.error(f"Error inicializando OpenAI embeddings: {e}")
            raise EmbeddingAPIError("openai", str(e))
    
    @property
    def embeddings(self) -> Embeddings:
        """
        Retorna el objeto Embeddings de LangChain.
        
        Útil para pasar directamente a ChromaDB o a chains de LangChain.
        """
        if self._embeddings is None:
            raise EmbeddingAPIError(self.provider, "Embeddings no inicializados")
        return self._embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para una lista de documentos.
        
        Optimizado para procesamiento en batch, más eficiente que
        llamar embed_query múltiples veces.
        
        Args:
            texts: Lista de textos a embeber
            
        Returns:
            Lista de vectores de embedding (cada uno es list[float])
            
        Raises:
            EmbeddingAPIError: Si hay error en la API
        """
        if not texts:
            return []
        
        # Filtrar textos vacíos
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        
        try:
            logger.debug(f"Generando embeddings para {len(valid_texts)} documentos")
            embeddings = self._embeddings.embed_documents(valid_texts)
            logger.debug(f"Embeddings generados: {len(embeddings)} vectores de dim {len(embeddings[0])}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generando embeddings de documentos: {e}")
            raise EmbeddingAPIError(self.provider, str(e))
    
    def embed_query(self, text: str) -> list[float]:
        """
        Genera embedding para una consulta individual.
        
        Algunos proveedores usan modelos diferentes para queries vs documentos
        para optimizar la búsqueda semántica.
        
        Args:
            text: Texto de la consulta
            
        Returns:
            Vector de embedding
            
        Raises:
            EmbeddingAPIError: Si hay error en la API
        """
        if not text or not text.strip():
            raise ValueError("El texto de query no puede estar vacío")
        
        try:
            logger.debug(f"Generando embedding para query: {text[:50]}...")
            embedding = self._embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding de query: {e}")
            raise EmbeddingAPIError(self.provider, str(e))
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna la dimensión de los embeddings generados.
        
        Útil para verificar compatibilidad con el vector store.
        """
        # Generar un embedding de prueba
        test_embedding = self.embed_query("test")
        return len(test_embedding)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_embedding_generator(
    prefer_local: bool = True,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> EmbeddingGenerator:
    """
    Factory para crear un EmbeddingGenerator con fallback automático.
    
    Intenta crear el generador en orden de preferencia:
    1. HuggingFace local (si prefer_local=True)
    2. Google Generative AI
    3. OpenAI
    
    Args:
        prefer_local: Si True, intenta HuggingFace local primero (default: True)
        google_api_key: API key de Google (opcional)
        openai_api_key: API key de OpenAI (opcional)
        
    Returns:
        EmbeddingGenerator configurado
    """
    providers = ["huggingface", "google", "openai"] if prefer_local else ["google", "openai", "huggingface"]
    
    for provider in providers:
        try:
            return EmbeddingGenerator(
                provider=provider,
                google_api_key=google_api_key,
                openai_api_key=openai_api_key
            )
        except (MissingAPIKeyError, EmbeddingAPIError) as e:
            logger.warning(f"No se pudo inicializar {provider}: {e}")
            continue
    
    raise MissingAPIKeyError("No se pudo inicializar ningún proveedor de embeddings")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    # Verificar que hay API key configurada
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Configura GOOGLE_API_KEY u OPENAI_API_KEY en tu .env")
        exit(1)
    
    try:
        generator = create_embedding_generator()
        
        # Test con documentos
        docs = [
            "La derivada mide la tasa de cambio instantánea.",
            "El integral es la operación inversa de la derivada.",
            "El teorema fundamental del cálculo conecta ambos conceptos."
        ]
        
        embeddings = generator.embed_documents(docs)
        print(f"✅ Generados {len(embeddings)} embeddings")
        print(f"   Dimensión: {len(embeddings[0])}")
        
        # Test con query
        query_emb = generator.embed_query("¿Qué es una integral?")
        print(f"✅ Query embedding generado (dim: {len(query_emb)})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
