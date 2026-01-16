"""
=============================================================================
Tutor IA Socrático - Lógica de Corrección de Soluciones
=============================================================================
Analiza y corrige soluciones escritas a mano usando modelos multimodales.

Flujo de corrección:
1. Validar legibilidad de la imagen
2. Analizar desarrollo lógico paso a paso
3. Comparar con "solución maestra" del RAG
4. Clasificar errores: cálculo vs concepto
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.vision.image_processor import ProcessedImage, ImageQuality
from src.ingestion import VectorStoreManager

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Tipos de errores en la solución."""
    NONE = "none"                    # Sin errores
    CALCULATION = "calculation"      # Error de cálculo (aritmética)
    CONCEPTUAL = "conceptual"        # Error de concepto/metodología
    PROCEDURAL = "procedural"        # Error en el procedimiento
    NOTATION = "notation"            # Error de notación matemática
    INCOMPLETE = "incomplete"        # Solución incompleta
    ILLEGIBLE = "illegible"          # No se puede leer


@dataclass
class StepAnalysis:
    """Análisis de un paso individual de la solución."""
    step_number: int
    content: str
    is_correct: bool
    error_type: Optional[ErrorType] = None
    explanation: str = ""
    suggestion: str = ""


@dataclass
class CorrectionResult:
    """
    Resultado completo de la corrección de una solución.
    
    Attributes:
        is_legible: Si la imagen era legible
        is_correct: Si la solución es correcta
        overall_score: Puntuación de 0 a 100
        error_type: Tipo principal de error (si hay)
        step_by_step: Análisis paso a paso
        feedback: Feedback general para el estudiante
        correct_solution: Solución correcta del RAG (si disponible)
        concepts_to_review: Conceptos que el estudiante debe repasar
    """
    is_legible: bool
    is_correct: bool
    overall_score: int
    error_type: Optional[ErrorType]
    step_by_step: list[StepAnalysis] = field(default_factory=list)
    feedback: str = ""
    correct_solution: str = ""
    concepts_to_review: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if not self.is_legible:
            return "❌ No se pudo leer la imagen claramente."
        
        status = "✅ Correcto" if self.is_correct else "❌ Tiene errores"
        return f"{status} | Puntuación: {self.overall_score}/100\n\n{self.feedback}"


class SolutionCorrector:
    """
    Corrector de soluciones usando visión multimodal.
    
    Analiza imágenes de soluciones escritas a mano y proporciona
    feedback detallado, identificando tipos de errores y sugiriendo
    correcciones basadas en los materiales del curso.
    
    Example:
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> 
        >>> llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        >>> corrector = SolutionCorrector(llm=llm, vector_store=store)
        >>> 
        >>> # Procesar imagen
        >>> image = ImageProcessor().process_file("solucion.jpg")
        >>> 
        >>> # Corregir
        >>> result = corrector.correct(
        ...     image=image,
        ...     problem_statement="Calcula la derivada de f(x) = x³ + 2x"
        ... )
        >>> print(result)
    """
    
    SYSTEM_PROMPT = """Eres un profesor universitario experto en corrección de exámenes.
Tu tarea es analizar soluciones escritas a mano por estudiantes y proporcionar 
feedback constructivo y educativo.

PROCESO DE ANÁLISIS:
1. Primero, verifica si la escritura es legible
2. Identifica cada paso de la solución
3. Evalúa la corrección de cada paso
4. Clasifica los errores encontrados
5. Proporciona feedback constructivo

TIPOS DE ERRORES:
- CALCULATION: Errores aritméticos o de cálculo (ej: 2+3=6)
- CONCEPTUAL: Errores de comprensión del concepto (ej: usar derivada en vez de integral)
- PROCEDURAL: Errores en el método aunque el concepto sea correcto
- NOTATION: Errores de notación matemática que no afectan el resultado
- INCOMPLETE: La solución está incompleta

FORMATO DE RESPUESTA:
Siempre responde en formato estructurado que pueda ser parseado.
Sé específico, constructivo y educativo.
"""

    ANALYSIS_PROMPT = """Analiza la siguiente solución escrita a mano a este problema:

PROBLEMA:
{problem_statement}

SOLUCIÓN DE REFERENCIA (del material del curso):
{reference_solution}

INSTRUCCIONES:
1. Examina la imagen cuidadosamente
2. Transcribe lo que puedes leer de la solución del estudiante
3. Compara paso a paso con la solución de referencia
4. Identifica errores específicos y clasifícalos

Responde en el siguiente formato JSON:
{{
    "legible": true/false,
    "transcripcion": "lo que se lee en la imagen",
    "pasos": [
        {{
            "numero": 1,
            "contenido": "lo que escribió el estudiante",
            "correcto": true/false,
            "tipo_error": "NONE/CALCULATION/CONCEPTUAL/PROCEDURAL/NOTATION/INCOMPLETE",
            "explicacion": "por qué está bien/mal",
            "sugerencia": "cómo corregirlo"
        }}
    ],
    "puntuacion": 0-100,
    "feedback_general": "mensaje constructivo para el estudiante",
    "conceptos_revisar": ["concepto1", "concepto2"]
}}
"""

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: Optional[VectorStoreManager] = None
    ):
        """
        Inicializa el corrector.
        
        Args:
            llm: Modelo de lenguaje multimodal (debe soportar imágenes)
            vector_store: Vector store para buscar soluciones de referencia
        """
        self.llm = llm
        self.vector_store = vector_store
        logger.info("SolutionCorrector inicializado")
    
    def _get_reference_solution(self, problem: str) -> str:
        """
        Busca una solución de referencia en el vector store.
        
        Args:
            problem: Enunciado del problema
            
        Returns:
            Solución de referencia o mensaje indicando que no hay
        """
        if not self.vector_store or self.vector_store.count == 0:
            return "No hay solución de referencia disponible."
        
        # Buscar en el vector store
        query = f"solución ejemplo {problem}"
        results = self.vector_store.search(query, n_results=3)
        
        if not results:
            return "No se encontró solución de referencia en los materiales."
        
        # Concatenar resultados relevantes
        references = []
        for result in results:
            references.append(result.content)
        
        return "\n\n---\n\n".join(references)
    
    def correct(
        self,
        image: ProcessedImage,
        problem_statement: str,
        custom_reference: Optional[str] = None
    ) -> CorrectionResult:
        """
        Corrige una solución en imagen.
        
        Args:
            image: Imagen procesada de la solución
            problem_statement: Enunciado del problema
            custom_reference: Solución de referencia personalizada (opcional)
            
        Returns:
            CorrectionResult con el análisis completo
        """
        # Validar legibilidad
        if not image.is_readable:
            return CorrectionResult(
                is_legible=False,
                is_correct=False,
                overall_score=0,
                error_type=ErrorType.ILLEGIBLE,
                feedback="La imagen no es lo suficientemente clara. Por favor, toma otra foto con mejor iluminación y enfoque."
            )
        
        # Obtener solución de referencia
        reference = custom_reference or self._get_reference_solution(problem_statement)
        
        # Construir prompt
        prompt = self.ANALYSIS_PROMPT.format(
            problem_statement=problem_statement,
            reference_solution=reference
        )
        
        # Construir mensaje multimodal
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                image.to_langchain_format()
            ])
        ]
        
        try:
            # Invocar LLM
            response = self.llm.invoke(messages)
            
            # Parsear respuesta
            return self._parse_response(response.content, reference)
            
        except Exception as e:
            logger.error(f"Error en corrección: {e}")
            return CorrectionResult(
                is_legible=True,
                is_correct=False,
                overall_score=0,
                error_type=None,
                feedback=f"Hubo un error al analizar la solución: {str(e)}"
            )
    
    def _parse_response(self, response: str, reference: str) -> CorrectionResult:
        """
        Parsea la respuesta del LLM.
        
        Args:
            response: Respuesta del LLM
            reference: Solución de referencia usada
            
        Returns:
            CorrectionResult parseado
        """
        import json
        import re
        
        try:
            # Intentar extraer JSON de la respuesta
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No se encontró JSON en la respuesta")
            
            data = json.loads(json_match.group())
            
            # Parsear pasos
            steps = []
            for paso in data.get("pasos", []):
                error_type = None
                tipo = paso.get("tipo_error", "NONE")
                if tipo != "NONE" and tipo in ErrorType.__members__:
                    error_type = ErrorType[tipo]
                
                steps.append(StepAnalysis(
                    step_number=paso.get("numero", 0),
                    content=paso.get("contenido", ""),
                    is_correct=paso.get("correcto", False),
                    error_type=error_type,
                    explanation=paso.get("explicacion", ""),
                    suggestion=paso.get("sugerencia", "")
                ))
            
            # Determinar tipo de error principal
            main_error = None
            for step in steps:
                if step.error_type and step.error_type != ErrorType.NONE:
                    main_error = step.error_type
                    break
            
            # Determinar si es correcto
            is_correct = all(s.is_correct for s in steps) if steps else False
            
            return CorrectionResult(
                is_legible=data.get("legible", True),
                is_correct=is_correct,
                overall_score=data.get("puntuacion", 0),
                error_type=main_error,
                step_by_step=steps,
                feedback=data.get("feedback_general", ""),
                correct_solution=reference,
                concepts_to_review=data.get("conceptos_revisar", [])
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parseando respuesta JSON: {e}")
            
            # Fallback: respuesta como texto plano
            return CorrectionResult(
                is_legible=True,
                is_correct=False,
                overall_score=0,
                error_type=None,
                feedback=response,
                correct_solution=reference
            )
    
    def quick_check(
        self,
        image: ProcessedImage,
        expected_answer: str
    ) -> tuple[bool, str]:
        """
        Verificación rápida de si la respuesta es correcta.
        
        Útil para respuestas cortas o resultados finales.
        
        Args:
            image: Imagen de la respuesta
            expected_answer: Respuesta esperada
            
        Returns:
            Tuple de (es_correcta, explicación)
        """
        if not image.is_readable:
            return False, "No se puede leer la imagen."
        
        prompt = f"""Verifica si la respuesta en la imagen coincide con: {expected_answer}

Responde solo:
- "CORRECTO" si la respuesta es correcta
- "INCORRECTO: <razón>" si no lo es
- "ILEGIBLE" si no puedes leer la imagen
"""
        
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                image.to_langchain_format()
            ])
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            if content.startswith("CORRECTO"):
                return True, "✅ ¡Correcto!"
            elif content.startswith("ILEGIBLE"):
                return False, "No se pudo leer la respuesta."
            else:
                return False, content.replace("INCORRECTO:", "").strip()
                
        except Exception as e:
            logger.error(f"Error en quick_check: {e}")
            return False, f"Error al verificar: {e}"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_corrector_with_google(
    persist_directory: Optional[str] = None,
    collection_name: str = "tutor_documents",
    model_name: str = "gemini-1.5-pro"
) -> SolutionCorrector:
    """
    Factory para crear corrector con Google Gemini.
    
    Args:
        persist_directory: Directorio de ChromaDB (opcional)
        collection_name: Nombre de la colección
        model_name: Modelo de Gemini
        
    Returns:
        SolutionCorrector configurado
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3  # Baja temperatura para corrección precisa
    )
    
    vector_store = None
    if persist_directory:
        from src.ingestion.embeddings import create_embedding_generator
        embeddings = create_embedding_generator(prefer_google=True)
        vector_store = VectorStoreManager(
            persist_directory=persist_directory,
            embedding_generator=embeddings,
            collection_name=collection_name
        )
    
    return SolutionCorrector(llm=llm, vector_store=vector_store)


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Configura GOOGLE_API_KEY en tu .env")
        exit(1)
    
    if len(sys.argv) < 2:
        print("Uso: python correction_logic.py <imagen> [problema]")
        exit(1)
    
    from src.vision.image_processor import ImageProcessor
    
    try:
        # Procesar imagen
        processor = ImageProcessor()
        image = processor.process_file(sys.argv[1])
        
        problem = sys.argv[2] if len(sys.argv) > 2 else "Resolver la ecuación"
        
        print(f"\n📝 Problema: {problem}")
        print(f"📷 Imagen: {sys.argv[1]}")
        print("\n⏳ Analizando...\n")
        
        # Crear corrector
        corrector = create_corrector_with_google()
        
        # Corregir
        result = corrector.correct(image, problem)
        
        print(result)
        
        if result.concepts_to_review:
            print(f"\n📚 Conceptos a repasar: {', '.join(result.concepts_to_review)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
