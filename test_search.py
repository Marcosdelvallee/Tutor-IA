# -*- coding: utf-8 -*-
"""Prueba de busqueda en el vector store."""
import os
import sys

os.chdir(r'c:\Users\user\Desktop\Edtech')
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from config import paths, chroma
from src.ingestion import VectorStoreManager
from langchain_huggingface import HuggingFaceEmbeddings

print("Inicializando embeddings locales...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

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

print("Conectando a ChromaDB...")
store = VectorStoreManager(
    persist_directory=str(paths.CHROMA_DB_DIR),
    embedding_generator=emb_wrapper,
    collection_name=chroma.collection_name
)
print("Documentos en DB: %d" % store.count)

print("\n" + "=" * 50)
query = "metodo del trapecio"
print("Buscando: '%s'" % query)
print("=" * 50)

results = store.search(query, n_results=3)

for i, r in enumerate(results, 1):
    print("\n--- Resultado %d (score: %.3f) ---" % (i, r.score))
    preview = r.content[:300].replace('\n', ' ')
    print(preview[:300] + "...")
    print("Fuente: %s" % r.metadata.get('source_file', 'N/A'))
