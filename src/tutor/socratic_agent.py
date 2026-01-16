"""
=============================================================================
Tutor IA Socrático - Agente Principal
=============================================================================
Implementa el agente tutor que:
- Integra RAG para buscar contexto relevante
- Aplica el método socrático (3 intentos antes de solución)
- Genera respuestas adaptativas según el progreso del estudiante
- Soporta bypass para respuestas directas
=============================================================================
"""

import logging
from typing import Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.ingestion import VectorStoreManager
from src.ingestion.embeddings import create_embedding_generator, EmbeddingGenerator
from src.tutor.response_logic import (
    AttemptTracker, 
    ResponseGenerator, 
    ResponseType,
    AttemptSession
)

logger = logging.getLogger(__name__)


@dataclass
class TutorResponse:
    """
    Respuesta estructurada del tutor.
    
    Attributes:
        content: Texto de la respuesta
        response_type: Tipo de respuesta generada
        attempt_number: Número de intento actual
        context_used: Fragmentos de contexto utilizados
        sources: Archivos fuente consultados
    """
    content: str
    response_type: ResponseType
    attempt_number: int
    context_used: list[str]
    sources: list[str]
    
    def __str__(self) -> str:
        return self.content


class SocraticTutor:
    """
    Agente tutor que implementa el método socrático.
    
    Combina RAG (búsqueda en documentos) con un LLM para generar
    respuestas que guían al estudiante mediante preguntas reflexivas,
    revelando la solución solo después de 3 intentos o a pedido.
    
    Example:
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> 
        >>> llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        >>> tutor = SocraticTutor(
        ...     llm=llm,
        ...     vector_store=store,
        ...     max_attempts=3
        ... )
        >>> 
        >>> # Primera pregunta - recibe pregunta socrática
        >>> response = tutor.ask("¿Qué es una integral?")
        >>> print(response)
        >>> 
        >>> # Intentar responder
        >>> response = tutor.respond("Es una suma de áreas")
    """
    
    SYSTEM_PROMPT = """Eres un profesor universitario de una institución prestigiosa.
Tu método de enseñanza es el MÉTODO SOCRÁTICO: guías a los estudiantes hacia 
el conocimiento a través de preguntas reflexivas, en lugar de dar respuestas directas.

PRINCIPIOS:
1. Nunca des la respuesta directamente en los primeros intentos
2. Haz preguntas que lleven al estudiante a descubrir la respuesta
3. Celebra el progreso y los intentos del estudiante
4. Sé paciente, motivador y claro
5. Usa el contexto de los materiales de estudio cuando esté disponible
6. Si el estudiante pide explícitamente la respuesta, puedes darla

FORMATO:
- Usa lenguaje claro y accesible
- Para fórmulas matemáticas, usa notación clara
- Estructura las respuestas largas con viñetas o numeración
"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStoreManager,
        max_attempts: int = 3,
        n_context_docs: int = 5
    ):
        """
        Inicializa el tutor socrático.
        
        Args:
            llm: Modelo de lenguaje de LangChain
            vector_store: Manager del vector store con documentos
            max_attempts: Intentos antes de revelar solución (default: 3)
            n_context_docs: Documentos de contexto a recuperar (default: 5)
        """
        self.llm = llm
        self.vector_store = vector_store
        self.max_attempts = max_attempts
        self.n_context_docs = n_context_docs
        
        # Componentes internos
        self.tracker = AttemptTracker(max_attempts=max_attempts)
        self.response_generator = ResponseGenerator()
        
        # Estado de la conversación actual
        self._current_question: Optional[str] = None
        self._conversation_history: list[dict] = []
        
        logger.info(
            f"SocraticTutor inicializado: max_attempts={max_attempts}, "
            f"n_context_docs={n_context_docs}"
        )
    
    def _retrieve_context(self, query: str) -> tuple[str, list[str]]:
        """
        Recupera contexto relevante del vector store.
        
        Args:
            query: Consulta para buscar
            
        Returns:
            Tuple de (contexto concatenado, lista de fuentes)
        """
        if self.vector_store.count == 0:
            logger.warning("Vector store vacío, sin contexto disponible")
            return "No hay materiales de referencia disponibles.", []
        
        results = self.vector_store.search(query, n_results=self.n_context_docs)
        
        if not results:
            return "No se encontró información relevante en los materiales.", []
        
        # Concatenar contextos
        contexts = []
        sources = set()
        
        for result in results:
            contexts.append(f"[Fuente: {result.metadata.get('source_file', 'Desconocido')}]\n{result.content}")
            if 'source_file' in result.metadata:
                sources.add(result.metadata['source_file'])
        
        context_text = "\n\n---\n\n".join(contexts)
        return context_text, list(sources)
    
    def _generate_response(
        self,
        prompt: str,
        response_type: ResponseType
    ) -> str:
        """
        Genera respuesta usando el LLM.
        
        Args:
            prompt: Prompt formateado
            response_type: Tipo de respuesta esperada
            
        Returns:
            Texto de respuesta del LLM
        """
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Lo siento, tuve un problema procesando tu pregunta. ¿Podrías reformularla?"
    
    def ask(
        self, 
        question: str,
        force_direct: bool = False
    ) -> TutorResponse:
        """
        Procesa una nueva pregunta del estudiante.
        
        Args:
            question: Pregunta del estudiante
            force_direct: Si True, da respuesta directa sin método socrático
            
        Returns:
            TutorResponse con la respuesta estructurada
        """
        self._current_question = question
        
        # Verificar si debe hacer bypass del método socrático
        if force_direct or self.tracker.should_bypass_socratic(question):
            response_type = ResponseType.DIRECT_ANSWER
            logger.info("Bypass socrático activado - respuesta directa")
        else:
            response_type = self.tracker.get_response_type(question)
        
        # Recuperar contexto RAG
        context, sources = self._retrieve_context(question)
        
        # Registrar el intento
        session, attempt_num = self.tracker.record_attempt(question)
        
        # Construir prompt según el tipo de respuesta
        prompt = self.response_generator.build_prompt(
            response_type=response_type,
            question=question,
            context=context
        )
        
        # Generar respuesta
        intro = self.response_generator.get_response_intro(response_type, attempt_num)
        llm_response = self._generate_response(prompt, response_type)
        
        full_response = f"{intro}\n\n{llm_response}" if intro else llm_response
        
        # Registrar en historial
        self._conversation_history.append({
            "role": "user",
            "content": question,
            "type": "question"
        })
        self._conversation_history.append({
            "role": "assistant", 
            "content": full_response,
            "type": response_type.value
        })
        
        return TutorResponse(
            content=full_response,
            response_type=response_type,
            attempt_number=attempt_num,
            context_used=[r.content[:200] for r in self.vector_store.search(question, n_results=3)] if self.vector_store.count > 0 else [],
            sources=sources
        )
    
    def respond(
        self, 
        student_response: str,
        evaluate: bool = True
    ) -> TutorResponse:
        """
        Procesa una respuesta/intento del estudiante.
        
        Args:
            student_response: Respuesta del estudiante al intento anterior
            evaluate: Si True, evalúa la respuesta del estudiante
            
        Returns:
            TutorResponse con feedback y siguiente paso
        """
        if not self._current_question:
            # Si no hay pregunta activa, tratar como nueva pregunta
            return self.ask(student_response)
        
        question = self._current_question
        
        # Recuperar contexto
        context, sources = self._retrieve_context(question)
        
        # Determinar tipo de respuesta según intentos
        session = self.tracker.get_or_create_session(question)
        
        # Registrar el intento con la respuesta del estudiante
        session.add_attempt(student_response)
        
        # Obtener nuevo tipo de respuesta
        if session.should_reveal_solution:
            response_type = ResponseType.FULL_SOLUTION
        else:
            response_type = self.tracker.get_response_type(question)
        
        # Construir prompt con la respuesta del estudiante
        prompt = self.response_generator.build_prompt(
            response_type=response_type,
            question=question,
            context=context,
            previous_attempt=student_response
        )
        
        # Agregar evaluación si está habilitada
        if evaluate:
            eval_prefix = f"""
El estudiante respondió: "{student_response}"

Primero, reconoce brevemente su respuesta (qué estuvo bien o cerca).
Luego, continúa con la guía según el tipo de respuesta.

"""
            prompt = eval_prefix + prompt
        
        # Generar respuesta
        intro = self.response_generator.get_response_intro(response_type, session.attempt_count)
        llm_response = self._generate_response(prompt, response_type)
        
        full_response = f"{intro}\n\n{llm_response}" if intro else llm_response
        
        # Registrar en historial
        self._conversation_history.append({
            "role": "user",
            "content": student_response,
            "type": "response"
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": full_response,
            "type": response_type.value
        })
        
        return TutorResponse(
            content=full_response,
            response_type=response_type,
            attempt_number=session.attempt_count,
            context_used=[],
            sources=sources
        )
    
    def show_solution(self) -> TutorResponse:
        """
        Muestra la solución completa para la pregunta actual.
        
        Returns:
            TutorResponse con la solución
        """
        if not self._current_question:
            return TutorResponse(
                content="No hay ninguna pregunta activa. ¿Qué te gustaría aprender?",
                response_type=ResponseType.DIRECT_ANSWER,
                attempt_number=0,
                context_used=[],
                sources=[]
            )
        
        return self.ask(self._current_question, force_direct=True)
    
    def new_topic(self) -> None:
        """Reinicia para un nuevo tema."""
        if self._current_question:
            self.tracker.reset_session(self._current_question)
        self._current_question = None
        logger.info("Sesión reiniciada para nuevo tema")
    
    def get_conversation_history(self) -> list[dict]:
        """Retorna el historial de conversación."""
        return self._conversation_history.copy()
    
    def clear_history(self) -> None:
        """Limpia el historial de conversación."""
        self._conversation_history.clear()
        self.tracker.clear_all()
        self._current_question = None
        logger.info("Historial limpiado")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================
def create_tutor_with_google(
    persist_directory: str,
    collection_name: str = "tutor_documents",
    model_name: str = "gemini-1.5-pro",
    max_attempts: int = 3
) -> SocraticTutor:
    """
    Factory para crear un tutor con Google Gemini.
    
    Args:
        persist_directory: Directorio de ChromaDB
        collection_name: Nombre de la colección
        model_name: Modelo de Gemini a usar
        max_attempts: Intentos antes de solución
        
    Returns:
        SocraticTutor configurado
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Crear componentes
    embeddings = create_embedding_generator(prefer_google=True)
    vector_store = VectorStoreManager(
        persist_directory=persist_directory,
        embedding_generator=embeddings,
        collection_name=collection_name
    )
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7
    )
    
    return SocraticTutor(
        llm=llm,
        vector_store=vector_store,
        max_attempts=max_attempts
    )


def create_tutor_with_openai(
    persist_directory: str,
    collection_name: str = "tutor_documents",
    model_name: str = "gpt-4o",
    max_attempts: int = 3
) -> SocraticTutor:
    """
    Factory para crear un tutor con OpenAI GPT.
    
    Args:
        persist_directory: Directorio de ChromaDB
        collection_name: Nombre de la colección
        model_name: Modelo de OpenAI a usar
        max_attempts: Intentos antes de solución
        
    Returns:
        SocraticTutor configurado
    """
    from langchain_openai import ChatOpenAI
    
    # Crear componentes
    embeddings = create_embedding_generator(prefer_google=False)
    vector_store = VectorStoreManager(
        persist_directory=persist_directory,
        embedding_generator=embeddings,
        collection_name=collection_name
    )
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7
    )
    
    return SocraticTutor(
        llm=llm,
        vector_store=vector_store,
        max_attempts=max_attempts
    )


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Configura GOOGLE_API_KEY u OPENAI_API_KEY en tu .env")
        exit(1)
    
    try:
        print("🎓 Iniciando Tutor Socrático...\n")
        
        tutor = create_tutor_with_google(
            persist_directory="data/chroma_db",
            max_attempts=3
        )
        
        # Demo interactivo simple
        question = "¿Qué es una derivada en cálculo?"
        print(f"📝 Estudiante: {question}\n")
        
        response = tutor.ask(question)
        print(f"👨‍🏫 Tutor ({response.response_type.value}):\n{response}\n")
        print(f"   Fuentes: {response.sources}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
