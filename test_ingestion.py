# -*- coding: utf-8 -*-
"""Script de ingesta usando embeddings locales (sin API)."""
import os
import sys

os.chdir(r'c:\Users\user\Desktop\Edtech')
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

print("=" * 50)
print("INGESTA CON EMBEDDINGS LOCALES")
print("=" * 50)

try:
    from config import paths, chunking, chroma
    from src.ingestion import PDFLoader, DocumentChunker, VectorStoreManager
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("\n1. Cargando PDF...")
    loader = PDFLoader()
    doc = loader.load('data/pdfs/Integracion_Numerica.pdf')
    print("   PDF cargado: %d paginas" % doc.total_pages)
    
    print("\n2. Fragmentando...")
    chunker = DocumentChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_document(doc)
    print("   Chunks: %d" % len(chunks))
    
    print("\n3. Inicializando embeddings locales...")
    print("   (Descargando modelo, puede tardar la primera vez)")
    
    # Usar modelo local de HuggingFace - no requiere API
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("   Embeddings locales OK")
    
    # Crear wrapper compatible
    class LocalEmbeddingWrapper:
        def __init__(self, hf_embeddings):
            self._embeddings = hf_embeddings
        
        @property
        def embeddings(self):
            return self._embeddings
        
        def embed_documents(self, texts):
            return self._embeddings.embed_documents(texts)
        
        def embed_query(self, text):
            return self._embeddings.embed_query(text)
    
    emb_wrapper = LocalEmbeddingWrapper(embeddings)
    
    print("\n4. Conectando a ChromaDB...")
    store = VectorStoreManager(
        persist_directory=str(paths.CHROMA_DB_DIR),
        embedding_generator=emb_wrapper,
        collection_name=chroma.collection_name
    )
    print("   DB conectada. Existentes: %d" % store.count)
    
    print("\n5. Almacenando chunks...")
    added = store.add_chunks(chunks)
    print("   Almacenados: %d/%d" % (added, len(chunks)))
    
    print("\n6. Verificando...")
    print("   Total en DB: %d" % store.count)
    
    if store.count > 0:
        print("\n7. Probando busqueda...")
        results = store.search("metodo de integracion", n_results=2)
        print("   Resultados: %d" % len(results))
        if results:
            print("   Score mejor resultado: %.3f" % results[0].score)
            preview = results[0].content[:100].replace('\n', ' ')
            print("   Preview: %s..." % preview)
    
    print("\n" + "=" * 50)
    print("[EXITO] INGESTA COMPLETADA!")
    print("=" * 50)
    
except Exception as e:
    print("\n[ERROR]: %s" % str(e))
    import traceback
    traceback.print_exc()
