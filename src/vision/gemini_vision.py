"""
=============================================================================
Gemini Vision - Analizador de imágenes con Google Gemini
=============================================================================
Extrae imágenes de PDFs y las analiza con Gemini Vision para obtener
descripciones textuales que se pueden indexar junto al texto.
=============================================================================
"""

import os
import io
import base64
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: str, min_width: int = 100, min_height: int = 100) -> list[dict]:
    """
    Extrae imágenes de un PDF.
    
    Args:
        pdf_path: Ruta al archivo PDF
        min_width: Ancho mínimo para filtrar imágenes pequeñas (iconos, etc.)
        min_height: Alto mínimo para filtrar imágenes pequeñas
        
    Returns:
        Lista de diccionarios con: page_number, image_bytes, width, height
    """
    images = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    width = base_image["width"]
                    height = base_image["height"]
                    ext = base_image["ext"]
                    
                    # Filtrar imágenes muy pequeñas
                    if width >= min_width and height >= min_height:
                        images.append({
                            "page_number": page_num + 1,
                            "image_bytes": image_bytes,
                            "width": width,
                            "height": height,
                            "extension": ext,
                            "index": img_index
                        })
                        logger.debug(f"Imagen extraída: página {page_num + 1}, {width}x{height}")
                        
                except Exception as e:
                    logger.warning(f"Error extrayendo imagen {img_index} de página {page_num + 1}: {e}")
                    continue
        
        doc.close()
        logger.info(f"Extraídas {len(images)} imágenes de {Path(pdf_path).name}")
        
    except Exception as e:
        logger.error(f"Error procesando PDF para imágenes: {e}")
    
    return images


def analyze_image_with_gemini(image_bytes: bytes, context: str = "") -> Optional[str]:
    """
    Analiza una imagen usando Gemini Vision.
    
    Args:
        image_bytes: Bytes de la imagen
        context: Contexto adicional sobre el documento (opcional)
        
    Returns:
        Descripción textual de la imagen, o None si falla
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("GOOGLE_API_KEY no configurada")
        return None
    
    try:
        import google.generativeai as genai
        from PIL import Image
        
        # Configurar API
        genai.configure(api_key=api_key)
        
        # Crear modelo Gemini Vision
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Convertir bytes a imagen PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prompt para análisis educativo
        prompt = f"""Analiza esta imagen de un documento educativo/académico.

Describe detalladamente:
1. Qué muestra la imagen (diagrama, gráfico, anatomía, fórmula, etc.)
2. Todos los elementos importantes que contiene
3. Etiquetas, textos o anotaciones visibles
4. Conceptos que ilustra
5. Relaciones entre elementos (si aplica)

{"Contexto del documento: " + context if context else ""}

Sé específico y preciso. Esta descripción se usará para búsquedas semánticas."""

        # Generar contenido
        response = model.generate_content([prompt, image])
        
        if response.text:
            logger.info(f"Imagen analizada correctamente ({len(response.text)} chars)")
            return response.text
        
        return None
        
    except Exception as e:
        logger.error(f"Error analizando imagen con Gemini: {e}")
        return None


def process_pdf_images(pdf_path: str, max_images: int = 10) -> list[dict]:
    """
    Procesa todas las imágenes de un PDF y las analiza con Gemini Vision.
    
    Args:
        pdf_path: Ruta al PDF
        max_images: Máximo de imágenes a procesar (para evitar costos excesivos)
        
    Returns:
        Lista de diccionarios con: page_number, description, width, height
    """
    results = []
    
    # Extraer imágenes
    images = extract_images_from_pdf(pdf_path)
    
    if not images:
        logger.info(f"No se encontraron imágenes en {Path(pdf_path).name}")
        return results
    
    # Limitar cantidad de imágenes
    images_to_process = images[:max_images]
    
    if len(images) > max_images:
        logger.warning(f"Limitando a {max_images} de {len(images)} imágenes")
    
    # Analizar cada imagen
    for i, img_data in enumerate(images_to_process):
        logger.info(f"Analizando imagen {i + 1}/{len(images_to_process)} (página {img_data['page_number']})")
        
        description = analyze_image_with_gemini(img_data["image_bytes"])
        
        if description:
            results.append({
                "page_number": img_data["page_number"],
                "description": description,
                "width": img_data["width"],
                "height": img_data["height"]
            })
    
    logger.info(f"Procesadas {len(results)} imágenes de {Path(pdf_path).name}")
    return results


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        print(f"\n📄 Extrayendo imágenes de: {pdf_path}")
        images = extract_images_from_pdf(pdf_path)
        print(f"   Encontradas {len(images)} imágenes")
        
        if images and os.getenv("GOOGLE_API_KEY"):
            print(f"\n🔍 Analizando primera imagen con Gemini Vision...")
            desc = analyze_image_with_gemini(images[0]["image_bytes"])
            if desc:
                print(f"\n📝 Descripción:\n{desc[:500]}...")
        else:
            print("⚠️ Configura GOOGLE_API_KEY para analizar imágenes")
    else:
        print("Uso: python gemini_vision.py <ruta_al_pdf>")
