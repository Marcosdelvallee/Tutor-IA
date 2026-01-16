# -*- coding: utf-8 -*-
"""
=============================================================================
Tutor IA - Interfaz Web con Streamlit
=============================================================================
Interfaz completa para:
- Chat Q&A basado en PDFs (respuestas directas)
- Subida e ingesta de PDFs
- Generacion de examenes/quizzes
- Correccion de soluciones con imagenes
=============================================================================
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Configurar path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configurar pagina
st.set_page_config(
    page_title="Tutor IA - Estudia con tus PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .exam-question {
        background-color: #1e1e2e;
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .exam-question * {
        color: #ffffff !important;
    }
    .correct-answer {
        background-color: #1a4d2e;
        color: #ffffff !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .wrong-answer {
        background-color: #4d1a1a;
        color: #ffffff !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INICIALIZACION DE COMPONENTES
# =============================================================================
@st.cache_resource
def init_embeddings():
    """Inicializa embeddings locales (cached)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


@st.cache_resource
def init_vector_store(_embeddings):
    """Inicializa vector store (cached)."""
    from config import paths, chroma
    from src.ingestion.vector_store import VectorStoreManager
    
    class EmbWrapper:
        def __init__(self, emb):
            self._embeddings = emb
        @property
        def embeddings(self):
            return self._embeddings
        def embed_documents(self, texts):
            return self._embeddings.embed_documents(texts)
        def embed_query(self, text):
            return self._embeddings.embed_query(text)
    
    return VectorStoreManager(
        persist_directory=str(paths.CHROMA_DB_DIR),
        embedding_generator=EmbWrapper(_embeddings),
        collection_name=chroma.collection_name
    )


@st.cache_resource
def init_groq_llm():
    """Inicializa Groq LLM (cached)."""
    from langchain_groq import ChatGroq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key
    )


# =============================================================================
# FUNCIONES DE CHAT (Q&A DIRECTO)
# =============================================================================
def search_context(query: str, store, n_results: int = 5) -> tuple[str, list]:
    """Busca contexto relevante en el vector store."""
    results = store.search(query, n_results=n_results)
    if not results:
        return "No hay material de referencia disponible.", []
    
    contexts = []
    sources = []
    for r in results:
        source = r.metadata.get('source_file', 'Desconocido')
        contexts.append(r.content)
        if source not in sources:
            sources.append(source)
    
    return "\n\n---\n\n".join(contexts), sources


def generate_direct_answer(question: str, context: str, llm) -> str:
    """Genera respuesta directa basada en el contenido del PDF."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    system_prompt = f"""Eres un asistente de estudio experto. Tu trabajo es responder preguntas 
basandote UNICAMENTE en el material de estudio proporcionado.

MATERIAL DE ESTUDIO:
{context}

INSTRUCCIONES:
1. Responde de forma clara y directa basandote en el material
2. Si la informacion no esta en el material, dilo claramente
3. Usa ejemplos del material cuando sea posible
4. Formatea bien la respuesta (usa bullets, negritas, etc.)
5. Si hay formulas matematicas, muestralas claramente"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generando respuesta: {str(e)}"


# =============================================================================
# FUNCIONES DE EXAMEN
# =============================================================================
def generate_exam(context: str, num_questions: int, question_type: str, llm) -> list:
    """Genera un examen basado en el contenido del PDF."""
    from langchain_core.messages import HumanMessage, SystemMessage
    import re
    
    if question_type == "multiple_choice":
        format_example = '''
{
  "questions": [
    {
      "question": "Pregunta aqui",
      "options": ["A) primera opcion", "B) segunda opcion", "C) tercera opcion", "D) cuarta opcion"],
      "correct": "B",
      "explanation": "Explicacion detallada aqui"
    }
  ]
}'''
    else:
        format_example = '''
{
  "questions": [
    {
      "question": "Pregunta aqui",
      "answer": "Respuesta detallada",
      "key_points": ["punto 1", "punto 2"]
    }
  ]
}'''
    
    system_prompt = f"""Eres un profesor universitario EXIGENTE. Genera {num_questions} preguntas de examen DIFICILES.

MATERIAL DE ESTUDIO:
{context}

FORMATO DE RESPUESTA (JSON):
{format_example}

REGLAS:
1. NIVEL UNIVERSITARIO AVANZADO - preguntas que requieran razonamiento profundo
2. PREGUNTAS TRAMPA - opciones que parecen correctas pero tienen errores sutiles  
3. Para formulas matematicas usa texto simple: x^2, h/2, integral de f(x), etc.
4. La explicacion debe enseñar POR QUE es correcta y por que las otras estan mal
5. Responde SOLO con JSON valido, sin comentarios ni texto adicional
6. NO uses caracteres especiales ni backslashes en el JSON"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Genera {num_questions} preguntas dificiles de nivel universitario")
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Limpiar markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        
        # Limpiar caracteres problematicos
        content = content.strip()
        
        # Intentar parsear
        data = json.loads(content)
        return data.get("questions", [])
        
    except json.JSONDecodeError as e:
        st.error(f"Error parseando JSON: {str(e)[:100]}")
        st.text("Respuesta del LLM (para debug):")
        st.code(response.content[:500] if response else "No response")
        return []
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return []


# =============================================================================
# FUNCIONES DE PDF
# =============================================================================
def process_uploaded_pdf(uploaded_file, store):
    """Procesa un PDF subido."""
    from src.ingestion import PDFLoader, DocumentChunker
    from config import chunking
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        loader = PDFLoader()
        doc = loader.load(tmp_path)
        
        chunker = DocumentChunker(
            chunk_size=chunking.chunk_size,
            overlap=chunking.chunk_overlap
        )
        chunks = chunker.chunk_document(doc)
        
        for chunk in chunks:
            chunk.source_file = uploaded_file.name
        
        added = store.add_chunks(chunks)
        
        return {
            "success": True,
            "pages": doc.total_pages,
            "chunks": len(chunks),
            "added": added
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    st.markdown('<h1 class="main-header">📚 Tutor IA - Estudia con tus PDFs</h1>', unsafe_allow_html=True)
    
    # Inicializar componentes
    with st.spinner("Cargando modelos..."):
        embeddings = init_embeddings()
        store = init_vector_store(embeddings)
        llm = init_groq_llm()
    
    if llm is None:
        st.error("❌ No se encontro GROQ_API_KEY en el archivo .env")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Estado del Sistema")
        st.metric("Documentos indexados", store.count)
        
        sources = store.get_sources()
        if sources:
            st.write("**PDFs cargados:**")
            for src in sources:
                st.write(f"📄 {src}")
        else:
            st.info("Sube PDFs para empezar")
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "💬 Pregunta sobre el PDF",
        "📝 Generar Examen", 
        "📚 Subir PDFs"
    ])
    
    # =================================
    # TAB 1: CHAT Q&A DIRECTO
    # =================================
    with tab1:
        st.subheader("Pregunta lo que quieras sobre tus documentos")
        
        if store.count == 0:
            st.warning("⚠️ No hay documentos indexados. Sube PDFs primero en la pestaña '📚 Subir PDFs'.")
        else:
            st.success(f"✅ Listo para responder preguntas sobre {store.count} fragmentos de texto")
        
        # Estado del chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mostrar historial
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    st.caption(f"📄 Fuentes: {', '.join(msg['sources'])}")
        
        # Input del usuario
        if prompt := st.chat_input("Escribe tu pregunta sobre el PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Buscando en el material..."):
                    context, sources = search_context(prompt, store)
                    response = generate_direct_answer(prompt, context, llm)
                    st.markdown(response)
                    if sources:
                        st.caption(f"📄 Fuentes: {', '.join(sources)}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
        
        # Boton limpiar
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # =================================
    # TAB 2: GENERAR EXAMEN
    # =================================
    with tab2:
        st.subheader("Genera un examen basado en tus PDFs")
        
        if store.count == 0:
            st.warning("⚠️ Sube PDFs primero para generar examenes.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                num_questions = st.slider("Numero de preguntas:", 3, 15, 5)
            
            with col2:
                question_type = st.selectbox(
                    "Tipo de preguntas:",
                    ["multiple_choice", "open_ended"],
                    format_func=lambda x: "Opcion Multiple" if x == "multiple_choice" else "Respuesta Abierta"
                )
            
            if st.button("📝 Generar Examen", type="primary"):
                with st.spinner("Generando examen..."):
                    # Obtener contexto amplio
                    context, _ = search_context("resumen conceptos principales", store, n_results=10)
                    questions = generate_exam(context, num_questions, question_type, llm)
                    
                    if questions:
                        st.session_state.exam_questions = questions
                        st.session_state.exam_type = question_type
                        st.session_state.exam_answers = {}
                        st.success(f"✅ Examen generado con {len(questions)} preguntas")
            
            # Mostrar examen si existe
            if "exam_questions" in st.session_state and st.session_state.exam_questions:
                st.divider()
                st.subheader("📋 Tu Examen")
                
                questions = st.session_state.exam_questions
                exam_type = st.session_state.exam_type
                
                for i, q in enumerate(questions):
                    # Usar st.markdown para renderizar LaTeX
                    st.markdown(f"**Pregunta {i+1}:**")
                    st.markdown(q["question"])
                    
                    if exam_type == "multiple_choice":
                        options = q.get("options", [])
                        answer = st.radio(
                            f"Selecciona tu respuesta:",
                            options,
                            key=f"q_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state.exam_answers[i] = answer
                    else:
                        answer = st.text_area(
                            "Tu respuesta:",
                            key=f"q_{i}",
                            height=100
                        )
                        st.session_state.exam_answers[i] = answer
                    
                    st.markdown("---")
                
                if st.button("✅ Calificar Examen", type="primary"):
                    st.divider()
                    st.subheader("📊 Resultados")
                    
                    correct = 0
                    total = len(questions)
                    
                    for i, q in enumerate(questions):
                        user_answer = st.session_state.exam_answers.get(i, "")
                        
                        if exam_type == "multiple_choice":
                            # Extraer letra de la respuesta
                            user_letter = user_answer[0] if user_answer else ""
                            correct_letter = q.get("correct", "")
                            is_correct = user_letter == correct_letter
                            
                            # Mostrar pregunta
                            st.markdown(f"### Pregunta {i+1}")
                            st.markdown(q["question"])
                            
                            if is_correct:
                                correct += 1
                                st.success(f"✅ **Correcto!** Tu respuesta: {user_answer}")
                            else:
                                st.error(f"❌ **Incorrecto.** Tu respuesta: {user_answer}")
                                st.warning(f"La respuesta correcta es: **{correct_letter}**")
                            
                            # Mostrar explicación detallada SIEMPRE
                            explanation = q.get('explanation', 'No hay explicación disponible.')
                            st.info(f"💡 **Explicación:** {explanation}")
                            st.markdown("---")
                        else:
                            # Para respuestas abiertas
                            st.markdown(f"### Pregunta {i+1}")
                            st.markdown(q["question"])
                            st.write(f"**Tu respuesta:** {user_answer}")
                            st.markdown(f"**Respuesta esperada:**")
                            st.markdown(q.get('answer', ''))
                            if q.get('key_points'):
                                st.write("**Puntos clave a incluir:**")
                                for point in q['key_points']:
                                    st.markdown(f"- {point}")
                            st.markdown("---")
                    
                    if exam_type == "multiple_choice":
                        score = (correct / total) * 100
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Puntuación", f"{score:.0f}%")
                        with col2:
                            st.metric("Correctas", f"{correct}/{total}")
                        with col3:
                            if score >= 70:
                                st.success("🎉 ¡Aprobado!")
                            else:
                                st.warning("📚 Sigue estudiando")
    
    # =================================
    # TAB 3: SUBIR PDFs
    # =================================
    with tab3:
        st.subheader("Sube tus documentos PDF")
        
        uploaded_files = st.file_uploader(
            "Arrastra tus PDFs aqui",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📥 Procesar PDFs", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status.write(f"Procesando: {file.name}...")
                    result = process_uploaded_pdf(file, store)
                    
                    if result["success"]:
                        st.success(f"✅ {file.name}: {result['pages']} paginas, {result['added']} fragmentos indexados")
                    else:
                        st.error(f"❌ {file.name}: {result['error']}")
                    
                    progress.progress((i + 1) / len(uploaded_files))
                
                status.write("✅ Proceso completado!")
                st.rerun()


if __name__ == "__main__":
    main()
