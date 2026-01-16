---
title: Tutor IA
emoji: 🎓
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 📚 Tutor IA

Un tutor de inteligencia artificial que te ayuda a estudiar tus PDFs con preguntas, flashcards, exámenes y más.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Características

### 💬 Chat Inteligente
- Preguntas y respuestas sobre el contenido de tus PDFs
- Múltiples conversaciones con historial persistente
- Renderizado de fórmulas matemáticas con LaTeX/MathJax

### 📇 Flashcards
- Generación automática de tarjetas de estudio
- Interfaz interactiva con flip animation
- Seguimiento de progreso (sabías / no sabías)

### 📋 Exámenes
- 4 niveles de dificultad: Fácil, Medio, Difícil, **Extremo**
- Preguntas de opción múltiple generadas por IA
- Calificación automática con explicaciones detalladas
- Modo Extremo con trampas y casos límite

### 🖼️ Análisis de Imágenes (Gemini Vision)
- Extracción automática de imágenes de PDFs
- Análisis con Google Gemini Vision
- Descripciones indexadas para búsqueda semántica

### ✍️ Corrección de Ejercicios
- Sube fotos de ejercicios escritos a mano
- El tutor los analiza y da feedback

### 🎨 Interfaz Premium
- Modo claro/oscuro
- Diseño responsive (mobile-friendly)
- Animaciones suaves
- Sidebar con historial de chats

## 🛠️ Tecnologías

- **Backend**: Flask, Python 3.10+
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Vision**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace (sentence-transformers)
- **Vector DB**: ChromaDB
- **Frontend**: HTML, CSS, JavaScript, MathJax

## 📦 Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/TU_USUARIO/tutor-ia.git
cd tutor-ia
```

### 2. Crear entorno virtual
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto:
```env
GROQ_API_KEY=tu_api_key_de_groq
GOOGLE_API_KEY=tu_api_key_de_google  # Opcional, para análisis de imágenes
```

**Obtener API Keys:**
- Groq: https://console.groq.com/
- Google: https://aistudio.google.com/

### 5. Ejecutar
```bash
python webapp.py
```

Abre http://localhost:5000 en tu navegador.

## 📖 Uso

1. **Subir PDFs**: Ve a la pestaña "📤 Subir" y carga tus documentos
2. **Preguntar**: Usa el chat para hacer preguntas sobre el contenido
3. **Estudiar**: Genera flashcards o exámenes para practicar
4. **Revisar**: Sube fotos de ejercicios para corrección

## 🔧 Estructura del Proyecto

```
tutor-ia/
├── webapp.py              # App principal Flask
├── config.py              # Configuración
├── templates/
│   └── index.html         # Frontend
├── src/
│   ├── ingestion/         # Carga de PDFs, chunking, embeddings
│   └── vision/            # Análisis de imágenes con Gemini
├── data/
│   └── chroma_db/         # Base de datos vectorial
└── requirements.txt
```

## 🎚️ Niveles de Dificultad (Exámenes)

| Nivel | Descripción |
|-------|-------------|
| 😊 Fácil | Definiciones directas, aplicación simple |
| 🤔 Medio | Requiere comprensión, 2-3 pasos |
| 😰 Difícil | Razonamiento multi-paso, opciones engañosas |
| 💀 Extremo | Nivel doctoral, trampas sutiles, afirmaciones falsas |

## 📄 Licencia

MIT License - Libre para uso personal y comercial.

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/nueva-funcion`)
3. Commit tus cambios (`git commit -m 'Add: nueva función'`)
4. Push al branch (`git push origin feature/nueva-funcion`)
5. Abre un Pull Request

---

**Hecho con ❤️ para estudiantes**
