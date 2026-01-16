"""
=============================================================================
Tutor IA Socrático - Gestor de Perfiles de Estudiante
=============================================================================
Persiste información del estudiante para personalizar la experiencia:
- Errores comunes y patrones
- Historial de temas cubiertos
- Progreso y estadísticas
- Preferencias de aprendizaje
=============================================================================
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """
    Registro de un error cometido por el estudiante.
    
    Attributes:
        topic: Tema relacionado
        error_type: Tipo de error (calculation, conceptual, etc.)
        description: Descripción del error
        question: Pregunta/problema donde ocurrió
        timestamp: Momento del error
        was_corrected: Si el estudiante corrigió después
    """
    topic: str
    error_type: str
    description: str
    question: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    was_corrected: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ErrorRecord":
        return cls(**data)


@dataclass
class TopicInteraction:
    """Registro de interacción con un tema."""
    topic: str
    questions_asked: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    last_interaction: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def accuracy(self) -> float:
        """Porcentaje de respuestas correctas."""
        total = self.correct_answers + self.incorrect_answers
        if total == 0:
            return 0.0
        return (self.correct_answers / total) * 100
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TopicInteraction":
        return cls(**data)


@dataclass
class StudentProfile:
    """
    Perfil completo del estudiante.
    
    Attributes:
        student_id: Identificador único
        name: Nombre del estudiante (opcional)
        created_at: Fecha de creación del perfil
        errors: Lista de errores registrados
        topics: Diccionario de temas e interacciones
        preferences: Preferencias de aprendizaje
        stats: Estadísticas generales
    """
    student_id: str
    name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: list[ErrorRecord] = field(default_factory=list)
    topics: dict[str, TopicInteraction] = field(default_factory=dict)
    preferences: dict = field(default_factory=lambda: {
        "prefer_socratic": True,
        "detail_level": "medium",  # low, medium, high
        "language": "es"
    })
    stats: dict = field(default_factory=lambda: {
        "total_sessions": 0,
        "total_questions": 0,
        "total_correct": 0,
        "streak_days": 0,
        "last_session": None
    })
    
    @property
    def error_patterns(self) -> dict[str, int]:
        """Obtiene los patrones de error más comunes."""
        error_types = [e.error_type for e in self.errors]
        return dict(Counter(error_types).most_common(5))
    
    @property
    def weak_topics(self) -> list[str]:
        """Temas con menor accuracy (< 50%)."""
        weak = []
        for topic_name, topic in self.topics.items():
            if topic.accuracy < 50 and (topic.correct_answers + topic.incorrect_answers) >= 2:
                weak.append(topic_name)
        return weak
    
    @property
    def strong_topics(self) -> list[str]:
        """Temas con alta accuracy (> 80%)."""
        strong = []
        for topic_name, topic in self.topics.items():
            if topic.accuracy > 80 and (topic.correct_answers + topic.incorrect_answers) >= 3:
                strong.append(topic_name)
        return strong
    
    @property
    def overall_accuracy(self) -> float:
        """Accuracy general del estudiante."""
        total_correct = sum(t.correct_answers for t in self.topics.values())
        total_incorrect = sum(t.incorrect_answers for t in self.topics.values())
        total = total_correct + total_incorrect
        if total == 0:
            return 0.0
        return (total_correct / total) * 100
    
    def record_error(
        self,
        topic: str,
        error_type: str,
        description: str,
        question: str
    ) -> None:
        """Registra un nuevo error."""
        error = ErrorRecord(
            topic=topic,
            error_type=error_type,
            description=description,
            question=question
        )
        self.errors.append(error)
        
        # Actualizar topic stats
        if topic not in self.topics:
            self.topics[topic] = TopicInteraction(topic=topic)
        self.topics[topic].incorrect_answers += 1
        self.topics[topic].last_interaction = datetime.now().isoformat()
    
    def record_success(self, topic: str) -> None:
        """Registra una respuesta correcta."""
        if topic not in self.topics:
            self.topics[topic] = TopicInteraction(topic=topic)
        self.topics[topic].correct_answers += 1
        self.topics[topic].last_interaction = datetime.now().isoformat()
    
    def record_question(self, topic: str) -> None:
        """Registra una pregunta hecha."""
        if topic not in self.topics:
            self.topics[topic] = TopicInteraction(topic=topic)
        self.topics[topic].questions_asked += 1
    
    def to_dict(self) -> dict:
        """Serializa el perfil a diccionario."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "created_at": self.created_at,
            "errors": [e.to_dict() for e in self.errors],
            "topics": {k: v.to_dict() for k, v in self.topics.items()},
            "preferences": self.preferences,
            "stats": self.stats
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        """Deserializa desde diccionario."""
        errors = [ErrorRecord.from_dict(e) for e in data.get("errors", [])]
        topics = {
            k: TopicInteraction.from_dict(v) 
            for k, v in data.get("topics", {}).items()
        }
        
        return cls(
            student_id=data["student_id"],
            name=data.get("name", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            errors=errors,
            topics=topics,
            preferences=data.get("preferences", {}),
            stats=data.get("stats", {})
        )


class ProfileManager:
    """
    Gestor de perfiles de estudiantes.
    
    Maneja la persistencia de perfiles en archivos JSON,
    permitiendo guardar y cargar información del estudiante
    entre sesiones.
    
    Example:
        >>> manager = ProfileManager("data/profiles/")
        >>> 
        >>> # Crear o cargar perfil
        >>> profile = manager.get_or_create("estudiante_123", "Juan")
        >>> 
        >>> # Registrar error
        >>> profile.record_error(
        ...     topic="derivadas",
        ...     error_type="calculation",
        ...     description="Error en regla de la cadena",
        ...     question="Derivar f(x) = sin(x²)"
        ... )
        >>> 
        >>> # Guardar
        >>> manager.save(profile)
    """
    
    def __init__(self, profiles_directory: str | Path):
        """
        Inicializa el gestor de perfiles.
        
        Args:
            profiles_directory: Directorio para almacenar perfiles JSON
        """
        self.profiles_dir = Path(profiles_directory)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: dict[str, StudentProfile] = {}
        logger.info(f"ProfileManager inicializado: {self.profiles_dir}")
    
    def _get_profile_path(self, student_id: str) -> Path:
        """Obtiene la ruta del archivo de perfil."""
        # Sanitizar ID para nombre de archivo
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in student_id)
        return self.profiles_dir / f"{safe_id}.json"
    
    def exists(self, student_id: str) -> bool:
        """Verifica si existe un perfil."""
        return self._get_profile_path(student_id).exists()
    
    def load(self, student_id: str) -> Optional[StudentProfile]:
        """
        Carga un perfil existente.
        
        Args:
            student_id: ID del estudiante
            
        Returns:
            StudentProfile si existe, None si no
        """
        # Verificar cache
        if student_id in self._cache:
            return self._cache[student_id]
        
        path = self._get_profile_path(student_id)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            profile = StudentProfile.from_dict(data)
            self._cache[student_id] = profile
            logger.debug(f"Perfil cargado: {student_id}")
            return profile
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error cargando perfil {student_id}: {e}")
            return None
    
    def save(self, profile: StudentProfile) -> bool:
        """
        Guarda un perfil.
        
        Args:
            profile: Perfil a guardar
            
        Returns:
            True si se guardó correctamente
        """
        path = self._get_profile_path(profile.student_id)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
            
            self._cache[profile.student_id] = profile
            logger.debug(f"Perfil guardado: {profile.student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando perfil {profile.student_id}: {e}")
            return False
    
    def get_or_create(
        self, 
        student_id: str, 
        name: str = ""
    ) -> StudentProfile:
        """
        Obtiene un perfil existente o crea uno nuevo.
        
        Args:
            student_id: ID del estudiante
            name: Nombre del estudiante (para nuevos perfiles)
            
        Returns:
            StudentProfile existente o nuevo
        """
        profile = self.load(student_id)
        
        if profile is None:
            profile = StudentProfile(student_id=student_id, name=name)
            self.save(profile)
            logger.info(f"Nuevo perfil creado: {student_id}")
        
        return profile
    
    def delete(self, student_id: str) -> bool:
        """
        Elimina un perfil.
        
        Args:
            student_id: ID del estudiante
            
        Returns:
            True si se eliminó correctamente
        """
        path = self._get_profile_path(student_id)
        
        if student_id in self._cache:
            del self._cache[student_id]
        
        if path.exists():
            try:
                path.unlink()
                logger.info(f"Perfil eliminado: {student_id}")
                return True
            except Exception as e:
                logger.error(f"Error eliminando perfil {student_id}: {e}")
                return False
        
        return False
    
    def list_profiles(self) -> list[str]:
        """
        Lista todos los IDs de perfiles existentes.
        
        Returns:
            Lista de student_ids
        """
        return [
            p.stem 
            for p in self.profiles_dir.glob("*.json")
        ]
    
    def get_common_errors(self, student_id: str) -> list[tuple[str, int]]:
        """
        Obtiene los errores más comunes del estudiante.
        
        Args:
            student_id: ID del estudiante
            
        Returns:
            Lista de tuplas (tipo_error, cantidad)
        """
        profile = self.load(student_id)
        if not profile:
            return []
        
        return list(profile.error_patterns.items())
    
    def update_session_stats(self, student_id: str) -> None:
        """
        Actualiza estadísticas de sesión.
        
        Args:
            student_id: ID del estudiante
        """
        profile = self.load(student_id)
        if not profile:
            return
        
        now = datetime.now()
        last_session = profile.stats.get("last_session")
        
        profile.stats["total_sessions"] += 1
        profile.stats["last_session"] = now.isoformat()
        
        # Actualizar racha
        if last_session:
            last_date = datetime.fromisoformat(last_session).date()
            if (now.date() - last_date).days == 1:
                profile.stats["streak_days"] += 1
            elif (now.date() - last_date).days > 1:
                profile.stats["streak_days"] = 1
        else:
            profile.stats["streak_days"] = 1
        
        self.save(profile)


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import tempfile
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Crear manager con directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProfileManager(tmpdir)
        
        # Crear perfil
        profile = manager.get_or_create("demo_student", "Juan Pérez")
        print(f"\n👤 Perfil creado: {profile.student_id}")
        
        # Registrar errores
        profile.record_error(
            topic="Derivadas",
            error_type="calculation",
            description="Error aplicando regla de la cadena",
            question="Derivar sin(x²)"
        )
        
        profile.record_error(
            topic="Derivadas",
            error_type="conceptual",
            description="Confusión entre derivada e integral",
            question="¿Qué es la derivada?"
        )
        
        profile.record_success("Integrales")
        profile.record_success("Integrales")
        profile.record_success("Integrales")
        
        # Guardar
        manager.save(profile)
        
        # Mostrar stats
        print(f"\n📊 Estadísticas:")
        print(f"   Accuracy general: {profile.overall_accuracy:.1f}%")
        print(f"   Patrones de error: {profile.error_patterns}")
        print(f"   Temas débiles: {profile.weak_topics}")
        print(f"   Temas fuertes: {profile.strong_topics}")
        
        # Verificar persistencia
        loaded = manager.load("demo_student")
        print(f"\n✅ Perfil recargado: {len(loaded.errors)} errores")
