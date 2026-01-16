# -*- coding: utf-8 -*-
"""
Tutor IA - Flask Application with MathJax
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================
_embeddings = None
_store = None
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _embeddings

def get_store():
    global _store
    if _store is None:
        from config import paths, chroma
        from src.ingestion.vector_store import VectorStoreManager
        
        embeddings = get_embeddings()
        
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
        
        _store = VectorStoreManager(
            persist_directory=str(paths.CHROMA_DB_DIR),
            embedding_generator=EmbWrapper(embeddings),
            collection_name=chroma.collection_name
        )
    return _store

def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                groq_api_key=api_key
            )
    return _llm

# Modelos ordenados por tamaño de contexto (mayor a menor)
GROQ_MODELS = [
    {"name": "llama-3.1-8b-instant", "context": 131072},      # 128K - más contexto
    {"name": "llama-3.3-70b-versatile", "context": 32768},    # 32K - principal
    {"name": "mixtral-8x7b-32768", "context": 32768},         # 32K - alternativo
    {"name": "gemma2-9b-it", "context": 8192},                # 8K - backup
]

def invoke_with_fallback(messages, preferred_model=None):
    """
    Invoca un LLM con sistema de fallback automático.
    Si un modelo falla por límite de tokens, prueba el siguiente.
    """
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return None, "API key no configurada"
    
    # Ordenar modelos: el preferido primero, luego por contexto
    models_to_try = GROQ_MODELS.copy()
    if preferred_model:
        models_to_try = sorted(models_to_try, key=lambda m: m["name"] != preferred_model)
    
    last_error = None
    
    for model_info in models_to_try:
        try:
            print(f"🔄 Intentando con modelo: {model_info['name']} ({model_info['context']} tokens)")
            
            llm = ChatGroq(
                model=model_info["name"],
                temperature=0.7,
                groq_api_key=api_key
            )
            
            response = llm.invoke(messages)
            print(f"✅ Éxito con modelo: {model_info['name']}")
            return response, model_info["name"]
            
        except Exception as e:
            error_str = str(e).lower()
            last_error = str(e)
            
            # Si es error de tokens, intentar siguiente modelo
            if "token" in error_str or "context" in error_str or "limit" in error_str:
                print(f"⚠️ {model_info['name']} excedió límite de tokens, probando siguiente...")
                continue
            else:
                # Otro tipo de error, intentar siguiente
                print(f"❌ Error con {model_info['name']}: {e}")
                continue
    
    return None, f"Todos los modelos fallaron. Último error: {last_error}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def search_context(query: str, n_results: int = 5):
    store = get_store()
    results = store.search(query, n_results=n_results)
    if not results:
        return "", []
    
    contexts = []
    sources = []
    for r in results:
        source = r.metadata.get('source_file', 'Desconocido')
        contexts.append(r.content)
        if source not in sources:
            sources.append(source)
    
    return "\n\n---\n\n".join(contexts), sources


def generate_answer(question: str, context: str):
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = get_llm()
    if not llm:
        return "Error: No se encontró GROQ_API_KEY"
    
    system_prompt = f"""Eres un asistente de estudio experto. Responde basándote en el material.
Usa notación LaTeX para fórmulas: $...$ para inline, $$...$$ para display.

MATERIAL:
{context}

Responde de forma clara y usa LaTeX para todas las fórmulas matemáticas."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"


def generate_exam_questions(num_questions: int, question_type: str, difficulty: str = "dificil"):
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = get_llm()
    if not llm:
        return []
    
    context, _ = search_context("conceptos principales métodos fórmulas teoremas", n_results=12)
    
    # Prompts por nivel de dificultad
    difficulty_prompts = {
        "facil": """Genera preguntas BÁSICAS de comprensión.
- Definiciones directas
- Aplicación simple de fórmulas
- Una sola operación
- Sin trampas ni ambigüedades
- Respuestas claras y obvias""",
        
        "medio": """Genera preguntas de NIVEL INTERMEDIO.
- Requieren entender conceptos, no solo memorizar
- Aplicación de 2-3 pasos
- Algunas opciones pueden ser parcialmente correctas
- Incluir variaciones de fórmulas conocidas""",
        
        "dificil": """Genera preguntas de EXAMEN UNIVERSITARIO DIFÍCIL.
- Requieren razonamiento multi-paso
- Opciones engañosas basadas en errores comunes
- Combinación de 2+ conceptos
- Casos límite y condiciones especiales
- Trampas sutiles (signos, índices, límites)""",
        
        "extremo": """Genera preguntas de DIFICULTAD DOCTORAL/COMPETENCIA.
REGLAS EXTREMAS:
1. NUNCA preguntas directas - siempre requieren DERIVAR o DEMOSTRAR
2. TODAS las opciones deben parecer plausibles a primera vista
3. Las opciones incorrectas deben ser errores SUTILES:
   - Error de signo en un paso intermedio
   - Confundir límites de integración
   - Olvidar una condición de convergencia
   - Usar aproximación incorrecta del orden
4. COMBINACIÓN OBLIGATORIA de múltiples temas
5. CASOS PATOLÓGICOS: singularidades, discontinuidades, inestabilidades
6. Requiere conocer LIMITACIONES y CUANDO FALLA el método
7. Preguntas tipo "¿Cuál es la afirmación FALSA?" con todas pareciendo verdaderas
8. Errores comunes de estudiantes como opciones
9. JAMÁS preguntar "¿Cuál es la definición de X?"
10. El estudiante debe DETECTAR LA TRAMPA para responder correctamente"""
    }
    
    # Tipos de pregunta adicionales para extremo
    question_styles = {
        "facil": "opción múltiple con 4 opciones claras",
        "medio": "opción múltiple con opciones que requieren cálculo",
        "dificil": "opción múltiple con trampas y explicación detallada",
        "extremo": """VARÍA los tipos:
- "¿Cuál es la afirmación FALSA?"
- "¿En cuál caso FALLA el método?"
- "¿Cuál contiene un ERROR sutil?"
- "Ordene de mayor a menor precisión"
- "¿Cuál NO es una condición necesaria?"
- Casos donde la respuesta "correcta" depende de condiciones"""
    }
    
    format_example = '''[
  {
    "question": "Pregunta con fórmulas en LaTeX: $\\\\frac{h}{2}[f(a)+f(b)]$",
    "options": ["A) Primera", "B) Segunda", "C) Tercera", "D) Cuarta"],
    "correct": "B",
    "explanation": "Explicación detallada de por qué B es correcta y las demás son errores comunes..."
  }
]'''
    
    diff_prompt = difficulty_prompts.get(difficulty, difficulty_prompts["dificil"])
    style_prompt = question_styles.get(difficulty, question_styles["dificil"])
    
    system_prompt = f"""Eres un profesor universitario con PhD creando un examen.
NIVEL DE DIFICULTAD: {difficulty.upper()}

{diff_prompt}

ESTILO DE PREGUNTAS:
{style_prompt}

MATERIAL DEL CURSO:
{context}

FORMATO JSON (responde SOLO esto):
{format_example}

REGLAS TÉCNICAS:
1. USA LaTeX para fórmulas: $\\frac{{a}}{{b}}$, $\\int$, $\\sum$, $\\lim$
2. Escapa backslashes: usa \\\\ en lugar de \\
3. Explicación DETALLADA de por qué cada opción incorrecta está mal
4. Devuelve SOLO el JSON array válido, sin texto adicional"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Genera {num_questions} preguntas nivel {difficulty.upper()}")
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        content = content.strip()
        if content.startswith("{"):
            data = json.loads(content)
            return data.get("questions", [])
        else:
            return json.loads(content)
    except Exception as e:
        print(f"Error: {e}")
        return []


def generate_flashcards(num_cards: int):
    """Genera flashcards del contenido del PDF."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = get_llm()
    if not llm:
        print("Error: No LLM disponible")
        return []
    
    context, _ = search_context("conceptos definiciones formulas metodos", n_results=10)
    
    if not context:
        print("Error: No hay contexto disponible")
        return []
    
    system_prompt = f"""Genera {num_cards} flashcards de estudio.

MATERIAL:
{context[:3000]}

FORMATO JSON (responde SOLO esto):
[
  {{"front": "pregunta o concepto", "back": "respuesta", "category": "tema"}}
]

REGLAS:
1. Cada card = UN concepto
2. Front = pregunta corta
3. Back = respuesta concisa
4. Para formulas usa texto: x^2, h/2, integral(f(x))
5. NO uses backslashes ni caracteres especiales
6. SOLO devuelve el JSON array, nada mas"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Genera exactamente {num_cards} flashcards en JSON")
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        print(f"LLM Response (primeros 500 chars): {content[:500]}")
        
        # Limpiar markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        
        content = content.strip()
        
        # Asegurar que empieza con [
        if not content.startswith("["):
            # Buscar el inicio del array
            idx = content.find("[")
            if idx != -1:
                content = content[idx:]
        
        result = json.loads(content)
        print(f"Flashcards generadas: {len(result)}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print(f"Content that failed: {content[:300] if content else 'empty'}")
        return []
    except Exception as e:
        print(f"Error flashcards: {e}")
        return []


# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
def index():
    store = get_store()
    doc_count = store.count
    sources = store.get_sources()
    return render_template('index.html', doc_count=doc_count, sources=sources)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    context, sources = search_context(question)
    answer = generate_answer(question, context)
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })


@app.route('/exam', methods=['POST'])
def exam():
    data = request.json
    num_questions = data.get('num_questions', 5)
    question_type = data.get('type', 'multiple_choice')
    difficulty = data.get('difficulty', 'dificil')  # facil, medio, dificil, extremo
    
    questions = generate_exam_questions(num_questions, question_type, difficulty)
    
    return jsonify({'questions': questions, 'difficulty': difficulty})


@app.route('/flashcards', methods=['POST'])
def flashcards():
    data = request.json
    num_cards = data.get('num_cards', 10)
    
    cards = generate_flashcards(num_cards)
    
    return jsonify({'cards': cards})


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'})
    
    try:
        from src.ingestion import PDFLoader, DocumentChunker
        from src.ingestion.chunker import DocumentChunk
        from config import chunking
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Process text
        loader = PDFLoader()
        doc = loader.load(tmp_path)
        
        chunker = DocumentChunker(
            chunk_size=chunking.chunk_size,
            overlap=chunking.chunk_overlap
        )
        chunks = chunker.chunk_document(doc)
        
        for chunk in chunks:
            chunk.source_file = file.filename
        
        # Process images with Gemini Vision (if API key available)
        images_processed = 0
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from src.vision.gemini_vision import process_pdf_images
                
                image_results = process_pdf_images(tmp_path, max_images=5)
                
                for img_result in image_results:
                    # Create a chunk for each image description
                    image_chunk = DocumentChunk(
                        content=f"[IMAGEN - Página {img_result['page_number']}]\n{img_result['description']}",
                        source_file=file.filename,
                        source_path=tmp_path,
                        chunk_index=len(chunks) + images_processed,
                        page_numbers=[img_result['page_number']],
                        token_count=len(img_result['description'].split()),
                        metadata={"type": "image_description", "dimensions": f"{img_result['width']}x{img_result['height']}"}
                    )
                    chunks.append(image_chunk)
                    images_processed += 1
                    
            except Exception as e:
                print(f"⚠️ Error procesando imágenes: {e}")
        
        store = get_store()
        added = store.add_chunks(chunks)
        
        os.unlink(tmp_path)
        
        return jsonify({
            'success': True,
            'pages': doc.total_pages,
            'chunks': added,
            'images_analyzed': images_processed
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# =============================================================================
# IMAGE CORRECTION
# =============================================================================
@app.route('/correct-image', methods=['POST'])
def correct_image():
    """Corrige una imagen de solución escrita a mano."""
    import base64
    from langchain_core.messages import HumanMessage, SystemMessage
    
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió imagen'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No se seleccionó imagen'})
    
    # Leer imagen y convertir a base64
    image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Obtener contexto relevante
    data = request.form
    topic = data.get('topic', 'matemáticas')
    context, _ = search_context(f"{topic} formulas metodos", n_results=5)
    
    llm = get_llm()
    if not llm:
        return jsonify({'error': 'LLM no disponible'})
    
    # Nota: Groq no soporta imágenes directamente, usamos descripción
    system_prompt = f"""Eres un profesor experto corrigiendo soluciones de estudiantes.

CONTEXTO DEL MATERIAL:
{context[:2000]}

El estudiante ha subido una imagen de su solución escrita a mano.
Como no puedo ver la imagen directamente, por favor proporciona feedback general sobre:

1. Errores comunes en este tipo de problemas
2. Pasos correctos para resolver problemas de {topic}
3. Fórmulas clave a recordar
4. Consejos para evitar errores

Si el estudiante describe su solución en texto, corrige específicamente eso."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"El estudiante está trabajando en: {topic}. Proporciona feedback detallado sobre cómo resolver correctamente este tipo de problemas.")
    ]
    
    try:
        response = llm.invoke(messages)
        return jsonify({
            'feedback': response.content,
            'note': 'Nota: Para corrección precisa de imágenes se requiere un modelo multimodal (GPT-4V o Gemini Vision)'
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# =============================================================================
# PDF MANAGEMENT
# =============================================================================
@app.route('/pdfs', methods=['GET'])
def list_pdfs():
    """Lista todos los PDFs indexados."""
    store = get_store()
    sources = store.get_sources()
    
    # Obtener estadísticas por PDF
    pdf_stats = []
    for source in sources:
        # Contar chunks por fuente
        results = store.search(source, n_results=100)
        chunk_count = len([r for r in results if r.metadata.get('source_file') == source])
        pdf_stats.append({
            'name': source,
            'chunks': chunk_count
        })
    
    return jsonify({
        'pdfs': pdf_stats,
        'total_chunks': store.count
    })


@app.route('/pdfs/<path:pdf_name>', methods=['DELETE'])
def delete_pdf(pdf_name):
    """Elimina un PDF y todos sus chunks."""
    try:
        store = get_store()
        
        # Usar el método delete_by_source del VectorStoreManager
        deleted_count = store.delete_by_source(pdf_name)
        
        if deleted_count > 0:
            # Resetear store para refrescar count
            global _store
            _store = None
            
            return jsonify({
                'success': True,
                'deleted_chunks': deleted_count,
                'message': f'PDF "{pdf_name}" eliminado correctamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No se encontró el PDF "{pdf_name}"'
            })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("Inicializando modelos...")
    get_embeddings()
    get_store()
    print(f"Documentos en DB: {get_store().count}")
    print("\n📚 Tutor IA corriendo en: http://localhost:5000")
    app.run(debug=True, port=5000)
