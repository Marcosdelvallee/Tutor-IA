"""
=============================================================================
Tutor IA Socrático - Vector Store Manager (ChromaDB)
=============================================================================
Gestión de la base de datos vectorial ChromaDB para almacenamiento
y búsqueda semántica de documentos.

Características:
- Almacenamiento persistente local
- CRUD completo para documentos
- Búsqueda por similitud con filtros de metadatos
- Gestión de colecciones
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src.ingestion.chunker import DocumentChunk
from src.ingestion.embeddings import EmbeddingGenerator
from src.utils.exceptions import (
    VectorStoreConnectionError,
    CollectionNotFoundError,
    DocumentNotFoundError
)

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Resultado de una búsqueda en el vector store.
    
    Attributes:
        content: Texto del documento encontrado
        score: Score de similitud (0-1, mayor es mejor para cosine)
        metadata: Metadatos del documento
        chunk_id: ID único del chunk
    """
    content: str
    score: float
    metadata: dict
    chunk_id: str
    
    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"SearchResult(score={self.score:.3f}, '{preview}')"


class VectorStoreManager:
    """
    Manager para ChromaDB con soporte para almacenamiento persistente.
    
    Proporciona una interfaz simplificada para operaciones CRUD sobre
    documentos y búsqueda semántica.
    
    Example:
        >>> from src.ingestion import EmbeddingGenerator
        >>> 
        >>> embeddings = EmbeddingGenerator()
        >>> store = VectorStoreManager(
        ...     persist_directory="data/chroma_db",
        ...     embedding_generator=embeddings
        ... )
        >>> 
        >>> # Agregar documentos
        >>> store.add_chunks(chunks)
        >>> 
        >>> # Buscar
        >>> results = store.search("concepto de derivada", n_results=5)
    """
    
    def __init__(
        self,
        persist_directory: str | Path,
        embedding_generator: EmbeddingGenerator,
        collection_name: str = "tutor_documents"
    ):
        """
        Inicializa el manager de ChromaDB.
        
        Args:
            persist_directory: Directorio para almacenamiento persistente
            embedding_generator: Generador de embeddings
            collection_name: Nombre de la colección principal
        
        Raises:
            VectorStoreConnectionError: Si no se puede conectar a ChromaDB
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_generator = embedding_generator
        self.collection_name = collection_name
        
        # Asegurar que el directorio existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # Inicializar cliente ChromaDB con persistencia
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,  # Desactivar telemetría
                    allow_reset=True
                )
            )
            
            # Obtener o crear colección
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Usar distancia coseno
            )
            
            logger.info(
                f"VectorStoreManager inicializado: {self.persist_directory}, "
                f"colección: {collection_name}, "
                f"documentos existentes: {self._collection.count()}"
            )
            
        except Exception as e:
            logger.error(f"Error conectando a ChromaDB: {e}")
            raise VectorStoreConnectionError(str(self.persist_directory), str(e))
    
    @property
    def count(self) -> int:
        """Retorna el número de documentos en la colección."""
        return self._collection.count()
    
    def add_chunks(
        self, 
        chunks: list[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """
        Agrega chunks de documento al vector store.
        
        Procesa en batches para evitar timeouts en documentos grandes.
        
        Args:
            chunks: Lista de DocumentChunk a agregar
            batch_size: Tamaño del batch para procesamiento (default: 100)
            
        Returns:
            Número de chunks agregados exitosamente
        """
        if not chunks:
            logger.warning("Lista de chunks vacía, nada que agregar")
            return 0
        
        added = 0
        
        # Procesar en batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Preparar datos para ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "source_file": chunk.source_file,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "page_numbers": ",".join(map(str, chunk.page_numbers)),
                    **chunk.metadata
                }
                for chunk in batch
            ]
            
            try:
                # Generar embeddings para el batch
                embeddings = self.embedding_generator.embed_documents(documents)
                
                # Agregar a ChromaDB
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                added += len(batch)
                logger.debug(f"Batch {i // batch_size + 1}: {len(batch)} chunks agregados")
                
            except Exception as e:
                logger.error(f"Error agregando batch {i // batch_size + 1}: {e}")
                # Continuar con el siguiente batch
                continue
        
        logger.info(f"Total agregados: {added}/{len(chunks)} chunks")
        return added
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
        include_distances: bool = True
    ) -> list[SearchResult]:
        """
        Busca documentos similares a una consulta.
        
        Args:
            query: Texto de la consulta
            n_results: Número de resultados a retornar (default: 5)
            filter_metadata: Filtros de metadatos (opcional)
                Ejemplo: {"source_file": "capitulo1.pdf"}
            include_distances: Si incluir scores de similitud (default: True)
            
        Returns:
            Lista de SearchResult ordenados por similitud
        """
        if not query or not query.strip():
            logger.warning("Query vacía")
            return []
        
        try:
            # Generar embedding de la query
            query_embedding = self.embedding_generator.embed_query(query)
            
            # Preparar filtros
            where = filter_metadata if filter_metadata else None
            
            # Realizar búsqueda
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.count) if self.count > 0 else n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convertir a SearchResult
            search_results = []
            
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0] if results["distances"] else [0] * len(documents)
                ids = results["ids"][0]
                
                for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
                    # Convertir distancia coseno a score de similitud
                    # distancia coseno: 0 = idéntico, 2 = opuesto
                    # score: 1 = idéntico, 0 = opuesto
                    score = 1 - (dist / 2) if include_distances else 1.0
                    
                    search_results.append(SearchResult(
                        content=doc,
                        score=score,
                        metadata=meta,
                        chunk_id=chunk_id
                    ))
            
            logger.debug(f"Búsqueda: encontrados {len(search_results)} resultados")
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """
        Obtiene un documento por su ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            SearchResult si existe, None si no
        """
        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if result["documents"] and result["documents"]:
                return SearchResult(
                    content=result["documents"][0],
                    score=1.0,
                    metadata=result["metadatas"][0],
                    chunk_id=chunk_id
                )
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo documento {chunk_id}: {e}")
            return None
    
    def delete_by_source(self, source_file: str) -> int:
        """
        Elimina todos los chunks de un archivo fuente.
        
        Útil para re-procesar un documento.
        
        Args:
            source_file: Nombre del archivo fuente
            
        Returns:
            Número de documentos eliminados
        """
        try:
            # Buscar IDs de documentos del archivo
            results = self._collection.get(
                where={"source_file": source_file},
                include=[]
            )
            
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info(f"Eliminados {len(results['ids'])} chunks de {source_file}")
                return len(results["ids"])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error eliminando documentos de {source_file}: {e}")
            return 0
    
    def delete_all(self) -> None:
        """
        Elimina todos los documentos de la colección.
        
        ⚠️ Operación destructiva, usar con cuidado.
        """
        try:
            # Recrear la colección
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("Todos los documentos eliminados de la colección")
            
        except Exception as e:
            logger.error(f"Error eliminando todos los documentos: {e}")
    
    def get_sources(self) -> list[str]:
        """
        Retorna lista de archivos fuente únicos en la colección.
        
        Returns:
            Lista de nombres de archivos fuente
        """
        try:
            results = self._collection.get(include=["metadatas"])
            
            sources = set()
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    if "source_file" in meta:
                        sources.add(meta["source_file"])
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Error obteniendo fuentes: {e}")
            return []
    
    def get_stats(self) -> dict:
        """
        Retorna estadísticas de la colección.
        
        Returns:
            Dict con estadísticas
        """
        try:
            results = self._collection.get(include=["metadatas"])
            
            total_tokens = 0
            sources = set()
            
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    if "token_count" in meta:
                        total_tokens += meta["token_count"]
                    if "source_file" in meta:
                        sources.add(meta["source_file"])
            
            return {
                "total_documents": self.count,
                "total_tokens": total_tokens,
                "unique_sources": len(sources),
                "sources": sorted(list(sources)),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Configura GOOGLE_API_KEY u OPENAI_API_KEY en tu .env")
        exit(1)
    
    try:
        from src.ingestion.embeddings import create_embedding_generator
        from src.ingestion.chunker import DocumentChunker
        
        # Crear generador de embeddings
        embeddings = create_embedding_generator()
        
        # Crear vector store (en directorio temporal para test)
        store = VectorStoreManager(
            persist_directory="data/chroma_db",
            embedding_generator=embeddings
        )
        
        print(f"✅ Vector store inicializado")
        print(f"   Documentos existentes: {store.count}")
        print(f"   Fuentes: {store.get_sources()}")
        
        # Test de búsqueda si hay documentos
        if store.count > 0:
            results = store.search("concepto importante", n_results=3)
            print(f"\n🔍 Búsqueda de prueba: {len(results)} resultados")
            for r in results:
                print(f"   {r}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
