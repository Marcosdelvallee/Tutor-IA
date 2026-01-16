# Usar una imagen base de Python liviana
FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y habilitar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para PyMuPDF y ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto al contenedor
COPY . .

# Crear directorio para la base de datos vectorial y temporales si no existen
RUN mkdir -p data/chroma_db temp

# Exponer el puerto que usa Hugging Face Spaces (7860)
EXPOSE 7860

# Comando para ejecutar la aplicación usando Gunicorn
# Escuchando en 0.0.0.0:7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "webapp:app"]
