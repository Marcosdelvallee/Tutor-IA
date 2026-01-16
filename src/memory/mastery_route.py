"""
=============================================================================
Tutor IA Socrático - Generador de Ruta de Maestría
=============================================================================
Genera rutas de aprendizaje personalizadas basadas en:
- Errores cometidos por el estudiante
- Temas débiles identificados
- Progreso actual
- Metas de aprendizaje
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum

from src.memory.profile_manager import StudentProfile, ProfileManager

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Estado de un tema en la ruta."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    MASTERED = "mastered"


class Priority(Enum):
    """Prioridad de un tema."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TopicProgress:
    """
    Progreso en un tema específico.
    
    Attributes:
        topic_name: Nombre del tema
        status: Estado actual
        priority: Prioridad de estudio
        accuracy: Porcentaje de aciertos
        error_count: Número de errores
        recommended_exercises: Ejercicios recomendados
        estimated_time_minutes: Tiempo estimado de estudio
        prerequisites: Temas prerequisito
    """
    topic_name: str
    status: TopicStatus = TopicStatus.NOT_STARTED
    priority: Priority = Priority.MEDIUM
    accuracy: float = 0.0
    error_count: int = 0
    recommended_exercises: list[str] = field(default_factory=list)
    estimated_time_minutes: int = 30
    prerequisites: list[str] = field(default_factory=list)
    
    @property
    def is_ready_to_study(self) -> bool:
        """Indica si el tema está listo para estudiar (prerequisitos cumplidos)."""
        return len(self.prerequisites) == 0
    
    def to_dict(self) -> dict:
        return {
            "topic_name": self.topic_name,
            "status": self.status.value,
            "priority": self.priority.value,
            "accuracy": self.accuracy,
            "error_count": self.error_count,
            "recommended_exercises": self.recommended_exercises,
            "estimated_time_minutes": self.estimated_time_minutes,
            "prerequisites": self.prerequisites
        }


@dataclass
class MasteryRoute:
    """
    Ruta de maestría personalizada para el estudiante.
    
    Attributes:
        student_id: ID del estudiante
        generated_at: Fecha de generación
        focus_areas: Áreas de enfoque prioritario
        topics: Lista ordenada de temas a estudiar
        estimated_total_hours: Horas totales estimadas
        summary: Resumen de la ruta
        recommendations: Recomendaciones generales
    """
    student_id: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    focus_areas: list[str] = field(default_factory=list)
    topics: list[TopicProgress] = field(default_factory=list)
    estimated_total_hours: float = 0.0
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    
    @property
    def topics_by_priority(self) -> list[TopicProgress]:
        """Temas ordenados por prioridad (mayor a menor)."""
        return sorted(self.topics, key=lambda t: t.priority.value, reverse=True)
    
    @property
    def pending_topics(self) -> list[TopicProgress]:
        """Temas pendientes de completar."""
        return [
            t for t in self.topics 
            if t.status not in [TopicStatus.COMPLETED, TopicStatus.MASTERED]
        ]
    
    @property
    def completion_percentage(self) -> float:
        """Porcentaje de completitud de la ruta."""
        if not self.topics:
            return 0.0
        completed = len([
            t for t in self.topics 
            if t.status in [TopicStatus.COMPLETED, TopicStatus.MASTERED]
        ])
        return (completed / len(self.topics)) * 100
    
    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "generated_at": self.generated_at,
            "focus_areas": self.focus_areas,
            "topics": [t.to_dict() for t in self.topics],
            "estimated_total_hours": self.estimated_total_hours,
            "summary": self.summary,
            "recommendations": self.recommendations
        }
    
    def to_markdown(self) -> str:
        """Genera una representación Markdown de la ruta."""
        md = f"""# 🎯 Ruta de Maestría

**Estudiante:** {self.student_id}  
**Generada:** {self.generated_at[:10]}  
**Progreso:** {self.completion_percentage:.0f}% completado

## 📌 Resumen

{self.summary}

## 🔥 Áreas de Enfoque Prioritario

"""
        for area in self.focus_areas:
            md += f"- **{area}**\n"
        
        md += "\n## 📚 Plan de Estudio\n\n"
        
        priority_icons = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠",
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢"
        }
        
        status_icons = {
            TopicStatus.NOT_STARTED: "⬜",
            TopicStatus.IN_PROGRESS: "🔄",
            TopicStatus.NEEDS_REVIEW: "📝",
            TopicStatus.COMPLETED: "✅",
            TopicStatus.MASTERED: "⭐"
        }
        
        for topic in self.topics_by_priority:
            priority_icon = priority_icons.get(topic.priority, "⚪")
            status_icon = status_icons.get(topic.status, "⬜")
            
            md += f"### {status_icon} {topic.topic_name} {priority_icon}\n\n"
            md += f"- **Estado:** {topic.status.value.replace('_', ' ').title()}\n"
            md += f"- **Accuracy:** {topic.accuracy:.0f}%\n"
            md += f"- **Errores registrados:** {topic.error_count}\n"
            md += f"- **Tiempo estimado:** {topic.estimated_time_minutes} min\n"
            
            if topic.recommended_exercises:
                md += f"- **Ejercicios recomendados:**\n"
                for ex in topic.recommended_exercises:
                    md += f"  - {ex}\n"
            
            md += "\n"
        
        md += "## 💡 Recomendaciones\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            md += f"{i}. {rec}\n"
        
        md += f"\n---\n*Tiempo total estimado: {self.estimated_total_hours:.1f} horas*\n"
        
        return md


class MasteryRouteGenerator:
    """
    Generador de rutas de maestría personalizadas.
    
    Analiza el perfil del estudiante y genera una ruta de aprendizaje
    optimizada basada en sus debilidades y patrones de error.
    
    Example:
        >>> manager = ProfileManager("data/profiles/")
        >>> profile = manager.load("estudiante_123")
        >>> 
        >>> generator = MasteryRouteGenerator(manager)
        >>> route = generator.generate(profile)
        >>> 
        >>> print(route.to_markdown())
    """
    
    # Mapeo de temas y sus prerrequisitos (ejemplo para cálculo)
    TOPIC_PREREQUISITES = {
        "Límites": [],
        "Continuidad": ["Límites"],
        "Derivadas": ["Límites", "Continuidad"],
        "Reglas de derivación": ["Derivadas"],
        "Aplicaciones de derivadas": ["Reglas de derivación"],
        "Integrales indefinidas": ["Derivadas"],
        "Integrales definidas": ["Integrales indefinidas"],
        "Técnicas de integración": ["Integrales indefinidas"],
        "Aplicaciones de integrales": ["Integrales definidas"],
        "Series": ["Integrales"],
        "Ecuaciones diferenciales": ["Integrales", "Derivadas"]
    }
    
    # Ejercicios recomendados por tipo de error
    ERROR_EXERCISES = {
        "calculation": [
            "Practicar operaciones básicas",
            "Ejercicios de verificación paso a paso",
            "Problemas con respuestas para auto-corrección"
        ],
        "conceptual": [
            "Revisar definiciones fundamentales",
            "Ejercicios teóricos con explicación",
            "Comparar conceptos similares"
        ],
        "procedural": [
            "Seguir algoritmos paso a paso",
            "Ejercicios guiados con feedback",
            "Diagramas de flujo del procedimiento"
        ],
        "notation": [
            "Practicar notación matemática",
            "Ejercicios de transcripción",
            "Comparar notaciones equivalentes"
        ]
    }
    
    def __init__(self, profile_manager: Optional[ProfileManager] = None):
        """
        Inicializa el generador.
        
        Args:
            profile_manager: Manager de perfiles (opcional)
        """
        self.profile_manager = profile_manager
        logger.info("MasteryRouteGenerator inicializado")
    
    def generate(
        self, 
        profile: StudentProfile,
        available_topics: Optional[list[str]] = None
    ) -> MasteryRoute:
        """
        Genera una ruta de maestría personalizada.
        
        Args:
            profile: Perfil del estudiante
            available_topics: Lista de temas disponibles (opcional)
            
        Returns:
            MasteryRoute personalizada
        """
        topics_to_use = available_topics or list(self.TOPIC_PREREQUISITES.keys())
        
        # Analizar errores del perfil
        error_counts_by_topic = self._count_errors_by_topic(profile)
        
        # Generar progreso por tema
        topic_progress_list = []
        
        for topic_name in topics_to_use:
            # Obtener datos del perfil
            topic_data = profile.topics.get(topic_name)
            
            accuracy = topic_data.accuracy if topic_data else 0.0
            error_count = error_counts_by_topic.get(topic_name, 0)
            
            # Determinar estado
            status = self._determine_status(topic_data, accuracy)
            
            # Determinar prioridad
            priority = self._determine_priority(accuracy, error_count)
            
            # Obtener ejercicios recomendados
            exercises = self._get_recommended_exercises(profile, topic_name)
            
            # Prerrequisitos
            prereqs = self.TOPIC_PREREQUISITES.get(topic_name, [])
            pending_prereqs = [
                p for p in prereqs 
                if profile.topics.get(p, None) is None or 
                   profile.topics[p].accuracy < 50
            ]
            
            topic_progress = TopicProgress(
                topic_name=topic_name,
                status=status,
                priority=priority,
                accuracy=accuracy,
                error_count=error_count,
                recommended_exercises=exercises,
                estimated_time_minutes=self._estimate_study_time(accuracy, error_count),
                prerequisites=pending_prereqs
            )
            
            topic_progress_list.append(topic_progress)
        
        # Ordenar por prioridad y prerrequisitos
        sorted_topics = self._sort_topics(topic_progress_list)
        
        # Calcular tiempo total
        total_minutes = sum(t.estimated_time_minutes for t in sorted_topics)
        total_hours = total_minutes / 60
        
        # Generar áreas de enfoque
        focus_areas = self._identify_focus_areas(profile, sorted_topics)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(profile, sorted_topics)
        
        # Generar resumen
        summary = self._generate_summary(profile, sorted_topics, focus_areas)
        
        route = MasteryRoute(
            student_id=profile.student_id,
            focus_areas=focus_areas,
            topics=sorted_topics,
            estimated_total_hours=total_hours,
            summary=summary,
            recommendations=recommendations
        )
        
        logger.info(f"Ruta generada para {profile.student_id}: {len(sorted_topics)} temas")
        
        return route
    
    def _count_errors_by_topic(self, profile: StudentProfile) -> dict[str, int]:
        """Cuenta errores por tema."""
        counts = {}
        for error in profile.errors:
            topic = error.topic
            counts[topic] = counts.get(topic, 0) + 1
        return counts
    
    def _determine_status(self, topic_data, accuracy: float) -> TopicStatus:
        """Determina el estado de un tema."""
        if topic_data is None:
            return TopicStatus.NOT_STARTED
        
        total_answers = topic_data.correct_answers + topic_data.incorrect_answers
        
        if total_answers == 0:
            return TopicStatus.NOT_STARTED
        elif accuracy >= 90:
            return TopicStatus.MASTERED
        elif accuracy >= 70:
            return TopicStatus.COMPLETED
        elif accuracy >= 40:
            return TopicStatus.IN_PROGRESS
        else:
            return TopicStatus.NEEDS_REVIEW
    
    def _determine_priority(self, accuracy: float, error_count: int) -> Priority:
        """Determina la prioridad de estudio."""
        if accuracy < 30 or error_count >= 5:
            return Priority.CRITICAL
        elif accuracy < 50 or error_count >= 3:
            return Priority.HIGH
        elif accuracy < 70 or error_count >= 1:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _get_recommended_exercises(
        self, 
        profile: StudentProfile, 
        topic: str
    ) -> list[str]:
        """Obtiene ejercicios recomendados según errores."""
        exercises = []
        
        # Buscar errores en este tema
        for error in profile.errors:
            if error.topic == topic:
                error_exercises = self.ERROR_EXERCISES.get(error.error_type, [])
                for ex in error_exercises:
                    if ex not in exercises:
                        exercises.append(ex)
        
        # Limitar a 3 ejercicios
        return exercises[:3]
    
    def _estimate_study_time(self, accuracy: float, error_count: int) -> int:
        """Estima tiempo de estudio en minutos."""
        base_time = 30
        
        if accuracy < 30:
            base_time = 60
        elif accuracy < 50:
            base_time = 45
        
        # Agregar tiempo por errores
        base_time += error_count * 10
        
        return min(base_time, 120)  # Máximo 2 horas
    
    def _sort_topics(self, topics: list[TopicProgress]) -> list[TopicProgress]:
        """Ordena temas por prioridad y prerrequisitos."""
        # Primero ordenar por prioridad
        topics.sort(key=lambda t: t.priority.value, reverse=True)
        
        # Luego mover temas con prerrequisitos pendientes al final
        ready = [t for t in topics if t.is_ready_to_study]
        not_ready = [t for t in topics if not t.is_ready_to_study]
        
        return ready + not_ready
    
    def _identify_focus_areas(
        self, 
        profile: StudentProfile,
        topics: list[TopicProgress]
    ) -> list[str]:
        """Identifica áreas de enfoque prioritario."""
        focus = []
        
        # Temas críticos
        critical = [t.topic_name for t in topics if t.priority == Priority.CRITICAL]
        focus.extend(critical[:2])
        
        # Agregar según patrones de error
        for error_type, count in profile.error_patterns.items():
            if count >= 2:
                focus.append(f"Errores de {error_type}")
        
        return focus[:5]  # Máximo 5 áreas
    
    def _generate_recommendations(
        self,
        profile: StudentProfile,
        topics: list[TopicProgress]
    ) -> list[str]:
        """Genera recomendaciones personalizadas."""
        recs = []
        
        # Recomendación según accuracy general
        if profile.overall_accuracy < 50:
            recs.append(
                "Tu accuracy general es baja. Enfócate en los fundamentos antes "
                "de avanzar a temas más complejos."
            )
        elif profile.overall_accuracy >= 80:
            recs.append(
                "¡Excelente progreso! Continúa con la práctica avanzada para "
                "alcanzar la maestría."
            )
        
        # Recomendación según patrones de error
        if "calculation" in profile.error_patterns:
            recs.append(
                "Tienes varios errores de cálculo. Practica las operaciones "
                "básicas y verifica cada paso."
            )
        
        if "conceptual" in profile.error_patterns:
            recs.append(
                "Revisa las definiciones y conceptos fundamentales. "
                "Entender el 'por qué' es clave."
            )
        
        # Recomendación de prerrequisitos
        pending_prereqs = set()
        for topic in topics:
            pending_prereqs.update(topic.prerequisites)
        
        if pending_prereqs:
            recs.append(
                f"Completa primero: {', '.join(list(pending_prereqs)[:3])}"
            )
        
        return recs[:5]
    
    def _generate_summary(
        self,
        profile: StudentProfile,
        topics: list[TopicProgress],
        focus_areas: list[str]
    ) -> str:
        """Genera resumen de la ruta."""
        pending = len([t for t in topics if t.status != TopicStatus.MASTERED])
        mastered = len([t for t in topics if t.status == TopicStatus.MASTERED])
        
        summary = f"""Basado en tu historial, he identificado {pending} temas que necesitan trabajo.
Ya dominas {mastered} tema(s).

Tu accuracy general es {profile.overall_accuracy:.0f}%. """
        
        if focus_areas:
            summary += f"Las áreas prioritarias son: {', '.join(focus_areas[:2])}."
        
        return summary


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProfileManager(tmpdir)
        
        # Crear perfil de prueba
        profile = manager.get_or_create("demo_student", "Demo")
        
        # Simular historial
        profile.record_error("Derivadas", "calculation", "Error en regla de la cadena", "Derivar sin(x²)")
        profile.record_error("Derivadas", "calculation", "Error aritmético", "Derivar 3x²")
        profile.record_error("Límites", "conceptual", "Confusión con indeterminación", "Límite de x/x")
        profile.record_success("Integrales indefinidas")
        profile.record_success("Integrales indefinidas")
        profile.record_success("Integrales indefinidas")
        
        manager.save(profile)
        
        # Generar ruta
        generator = MasteryRouteGenerator(manager)
        route = generator.generate(profile)
        
        # Mostrar ruta en Markdown
        print(route.to_markdown())
