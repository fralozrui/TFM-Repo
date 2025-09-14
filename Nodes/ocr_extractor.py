"""
Nodo especializado en el reconocimiento óptico de caracteres (OCR) en imágenes.
Utiliza el modelo preentrenado de EasyOCR para extraer texto de manera eficiente.
"""

import easyocr
from Agent.orchestrator_keys import OrchestratorState
from Nodes.utils_models import read_img
import numpy as np

# 1. Carga del modelo
reader = easyocr.Reader(['es', 'en'])

def clean_ocr_with_gemini(ocr_text: str, user_input: str) -> str:
    """
    Limpia y optimiza el resultado de OCR usando Gemini, 
    generando un texto coherente y natural que se pueda usar como input en responder_node.

    Args:
        ocr_text (str): Texto crudo devuelto por EasyOCR.
        user_input (str): Consulta original del usuario (para mantener el idioma y contexto).
        language (str): Idioma de salida esperado (por defecto "es").
    
    Returns:
        str: Texto procesado y optimizado.
    """
    from typing import Any
    from Agent.orchestrator_keys import model_gemini 

    prompt = f"""
    Has recibido el siguiente texto extraído con OCR de una imagen:

    --- OCR RAW ---
    {ocr_text}
    ----------------

    Este texto puede contener errores de reconocimiento, saltos de línea innecesarios
    o caracteres extraños. 

    Tu tarea es corregir los errores evidentes y devolver un texto claro y natural,
    manteniendo la máxima fidelidad posible a la información original. 
    No inventes contenido que no esté presente.

    Requisitos:
    - Responde únicamente con el texto limpio y legible.
    - Usa el mismo idioma que usa el usuario en {user_input}.
    - Mantén la estructura natural (puede ser frase, título, lista corta o párrafo).
    - Evita incluir símbolos raros o saltos de línea innecesarios.

    El usuario preguntó:
    "{user_input}"
    """

    try:
        response = model_gemini.generate_content(prompt)
        texto_limpio = response.text.strip()
    except Exception as e:
        # fallback: devolvemos el texto original para no romper el flujo
        print(f"[ERROR en clean_ocr_with_gemini]: {e}")
        texto_limpio = ocr_text

    return texto_limpio

def tool_ocr(state: OrchestratorState) -> dict:
    """
    Función de nodo para LangGraph que extrae texto de una imagen.
    Recibe el estado, busca una imagen en base64 y devuelve el texto encontrado.
    """    
    # 2. Reutilizamos tu patrón para leer la imagen desde el estado
    img_base64 = state.get("img_base64", "")
    if img_base64:
        image = read_img(img_base64)  
    else:
        return "[ERROR en tool_ocr]: No se ha proporcionado ninguna imagen."
    if image is None:
        return "[ERROR en tool_ocr]: No se pudo cargar la imagen con el img_base64 proporcionado."

    if isinstance(image, tuple):
        # Es una tupla de error, por ej. (None, "mensaje...")
        error_message = image[1]
        return error_message
    else:
        # Es un objeto de imagen válido
        image = np.array(image)

    # 3. EasyOCR
    try:
        resultados = reader.readtext(image, detail=0, paragraph=True)
        texto_extraido = "\n".join(resultados)
        # Limpiamos con Gemini
        texto_final = clean_ocr_with_gemini(texto_extraido, state.get("user_input", ""))

        if not texto_final:
            texto_final = "[ERROR en node_extract_text_ocr] No se detectó texto en la imagen."
        return texto_final
    except Exception as e:
        error_msg = f"[ERROR en node_extract_text_ocr]: Ocurrió un error en EasyOCR: {e}"
        print(error_msg)
        return error_msg
