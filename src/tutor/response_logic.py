"""
=============================================================================
Tutor IA Socrático - Lógica de Respuestas y Seguimiento de Intentos
=============================================================================
Gestiona el flujo de respuestas del tutor socrático:
- Seguimiento de intentos por pregunta
- Escalamiento progresivo de ayuda
- Generación de respuestas adaptativas
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Tipos de respuesta del tutor."""
    SOCRATIC_QUESTION = "socratic_question"      # Pregunta reflexiva
    GUIDED_HINT = "guided_hint"                   # Pista más directa
    PARTIAL_SOLUTION = "partial_solution"         # Solución parcial
    FULL_SOLUTION = "full_solution"               # Solución completa
    DIRECT_ANSWER = "direct_answer"               # Respuesta directa (bypass)


@dataclass
class Attempt:
    """Representa un intento del estudiante."""
    question: str
    student_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    hints_given: list[str] = field(default_factory=list)


@dataclass
class AttemptSession:
    """
    Sesión de intentos para una pregunta específica.
    
    Attributes:
        original_question: Pregunta o problema original del estudiante
        attempts: Lista de intentos realizados
        max_attempts: Máximo de intentos antes de dar solución
        solved: Si el estudiante resolvió correctamente
    """
    original_question: str
    attempts: list[Attempt] = field(default_factory=list)
    max_attempts: int = 3
    solved: bool = False
    solution_revealed: bool = False
    
    @property
    def attempt_count(self) -> int:
        """Número de intentos realizados."""
        return len(self.attempts)
    
    @property
    def should_reveal_solution(self) -> bool:
        """Determina si se debe revelar la solución."""
        return self.attempt_count >= self.max_attempts and not self.solved
    
    def add_attempt(self, response: Optional[str] = None) -> Attempt:
        """Registra un nuevo intento."""
        attempt = Attempt(
            question=self.original_question,
            student_response=response
        )
        self.attempts.append(attempt)
        return attempt
    
    def mark_solved(self) -> None:
        """Marca la sesión como resuelta."""
        self.solved = True


class AttemptTracker:
    """
    Rastreador de intentos por sesión de usuario.
    
    Mantiene el estado de los intentos para múltiples preguntas,
    permitiendo el flujo socrático de 3 intentos.
    
    Example:
        >>> tracker = AttemptTracker()
        >>> session = tracker.get_or_create_session("¿Qué es una integral?")
        >>> print(session.attempt_count)  # 0
        >>> session.add_attempt("Es una suma")
        >>> print(session.attempt_count)  # 1
    """
    
    def __init__(self, max_attempts: int = 3):
        """
        Inicializa el tracker.
        
        Args:
            max_attempts: Intentos máximos antes de revelar solución
        """
        self.max_attempts = max_attempts
        self._sessions: dict[str, AttemptSession] = {}
        logger.info(f"AttemptTracker inicializado (max_attempts={max_attempts})")
    
    def _normalize_question(self, question: str) -> str:
        """Normaliza una pregunta para usar como clave."""
        return question.lower().strip()[:200]
    
    def get_or_create_session(self, question: str) -> AttemptSession:
        """
        Obtiene o crea una sesión para una pregunta.
        
        Args:
            question: Pregunta del estudiante
            
        Returns:
            AttemptSession existente o nueva
        """
        key = self._normalize_question(question)
        
        if key not in self._sessions:
            self._sessions[key] = AttemptSession(
                original_question=question,
                max_attempts=self.max_attempts
            )
            logger.debug(f"Nueva sesión creada para: {question[:50]}...")
        
        return self._sessions[key]
    
    def record_attempt(
        self, 
        question: str, 
        student_response: Optional[str] = None
    ) -> tuple[AttemptSession, int]:
        """
        Registra un intento y retorna el estado actual.
        
        Args:
            question: Pregunta original
            student_response: Respuesta del estudiante (opcional)
            
        Returns:
            Tuple de (sesión, número de intento actual)
        """
        session = self.get_or_create_session(question)
        session.add_attempt(student_response)
        return session, session.attempt_count
    
    def get_response_type(self, question: str) -> ResponseType:
        """
        Determina el tipo de respuesta basado en el historial.
        
        Args:
            question: Pregunta del estudiante
            
        Returns:
            ResponseType apropiado para el estado actual
        """
        session = self.get_or_create_session(question)
        
        if session.solved:
            return ResponseType.DIRECT_ANSWER
        
        attempt_num = session.attempt_count
        
        if attempt_num == 0:
            return ResponseType.SOCRATIC_QUESTION
        elif attempt_num == 1:
            return ResponseType.GUIDED_HINT
        elif attempt_num == 2:
            return ResponseType.PARTIAL_SOLUTION
        else:
            return ResponseType.FULL_SOLUTION
    
    def should_bypass_socratic(self, message: str) -> bool:
        """
        Detecta si el usuario quiere saltar el método socrático.
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            True si debe dar respuesta directa
        """
        bypass_keywords = [
            "muéstrame la solución",
            "dame la respuesta",
            "no entiendo nada",
            "dime directamente",
            "solo responde",
            "respuesta directa",
            "no más preguntas"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in bypass_keywords)
    
    def reset_session(self, question: str) -> None:
        """Reinicia la sesión para una pregunta."""
        key = self._normalize_question(question)
        if key in self._sessions:
            del self._sessions[key]
            logger.debug(f"Sesión reiniciada para: {question[:50]}...")
    
    def clear_all(self) -> None:
        """Limpia todas las sesiones."""
        self._sessions.clear()
        logger.info("Todas las sesiones limpiadas")


class ResponseGenerator:
    """
    Genera las respuestas del tutor según el tipo requerido.
    
    Proporciona plantillas y lógica para generar respuestas
    adaptativas basadas en el contexto RAG y el estado del estudiante.
    """
    
    # Plantillas de prompts para cada tipo de respuesta
    PROMPTS = {
        ResponseType.SOCRATIC_QUESTION: """
Eres un profesor universitario que usa el método socrático.
El estudiante pregunta: "{question}"

Contexto relevante de los materiales:
{context}

INSTRUCCIONES:
- NO des la respuesta directa
- Haz UNA pregunta reflexiva que guíe al estudiante hacia la respuesta
- La pregunta debe ayudar a descubrir el concepto por sí mismo
- Sé amable y motivador
- Máximo 2-3 oraciones

Responde solo con la pregunta socrática:
""",
        
        ResponseType.GUIDED_HINT: """
Eres un profesor universitario. El estudiante ya intentó responder pero necesita más ayuda.
Pregunta original: "{question}"
Intento previo del estudiante: "{previous_attempt}"

Contexto relevante:
{context}

INSTRUCCIONES:
- Da una pista más específica que lo acerque a la respuesta
- Menciona un concepto clave que debería considerar
- Aún NO reveles la respuesta completa
- Incluye una pregunta guía más específica

Responde con la pista y pregunta:
""",

        ResponseType.PARTIAL_SOLUTION: """
Eres un profesor universitario. El estudiante ha intentado varias veces y necesita más ayuda.
Pregunta original: "{question}"

Contexto relevante:
{context}

INSTRUCCIONES:
- Muestra el PRIMER PASO de la solución
- Explica el razonamiento de ese paso
- Pregunta si puede continuar desde ahí
- Mantén un tono de apoyo

Responde con el primer paso:
""",

        ResponseType.FULL_SOLUTION: """
Eres un profesor universitario. El estudiante necesita ver la solución completa.
Pregunta original: "{question}"

Contexto relevante:
{context}

INSTRUCCIONES:
- Proporciona la solución paso a paso
- Explica el razonamiento detrás de cada paso
- Resalta los conceptos clave
- Al final, haz una pregunta para verificar comprensión
- Usa formato claro con numeración

Responde con la solución completa:
""",

        ResponseType.DIRECT_ANSWER: """
Eres un profesor universitario respondiendo directamente.
Pregunta: "{question}"

Contexto relevante:
{context}

INSTRUCCIONES:
- Responde de forma clara y directa
- Incluye la explicación necesaria
- Usa ejemplos si ayudan
- Sé conciso pero completo

Responde:
"""
    }
    
    def __init__(self):
        """Inicializa el generador de respuestas."""
        logger.info("ResponseGenerator inicializado")
    
    def build_prompt(
        self,
        response_type: ResponseType,
        question: str,
        context: str,
        previous_attempt: Optional[str] = None
    ) -> str:
        """
        Construye el prompt para el LLM según el tipo de respuesta.
        
        Args:
            response_type: Tipo de respuesta a generar
            question: Pregunta del estudiante
            context: Contexto RAG recuperado
            previous_attempt: Respuesta previa del estudiante (si aplica)
            
        Returns:
            Prompt formateado para el LLM
        """
        template = self.PROMPTS.get(response_type, self.PROMPTS[ResponseType.DIRECT_ANSWER])
        
        return template.format(
            question=question,
            context=context,
            previous_attempt=previous_attempt or "No disponible"
        )
    
    def get_response_intro(self, response_type: ResponseType, attempt_num: int) -> str:
        """
        Genera una introducción para la respuesta.
        
        Args:
            response_type: Tipo de respuesta
            attempt_num: Número de intento actual
            
        Returns:
            Texto de introducción
        """
        intros = {
            ResponseType.SOCRATIC_QUESTION: "🤔 Reflexionemos juntos...",
            ResponseType.GUIDED_HINT: f"💡 Pista #{attempt_num}:",
            ResponseType.PARTIAL_SOLUTION: "📝 Veamos el primer paso...",
            ResponseType.FULL_SOLUTION: "✅ Aquí está la solución completa:",
            ResponseType.DIRECT_ANSWER: "📚 Respuesta:"
        }
        return intros.get(response_type, "")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Demo del tracker
    tracker = AttemptTracker(max_attempts=3)
    
    question = "¿Cómo calculo la derivada de x²?"
    
    print(f"\n📝 Pregunta: {question}\n")
    
    for i in range(4):
        response_type = tracker.get_response_type(question)
        print(f"Intento {i}: {response_type.value}")
        
        session, attempt_num = tracker.record_attempt(question, f"Respuesta {i}")
        
        if session.should_reveal_solution:
            print("  → Revelando solución completa")
            break
    
    # Test bypass
    print(f"\n¿Bypass 'dame la respuesta'? {tracker.should_bypass_socratic('dame la respuesta')}")
    print(f"¿Bypass 'hola'? {tracker.should_bypass_socratic('hola')}")
