"""
Nodo especializado en el reconocimiento óptico de caracteres (OCR) en imágenes.
Utiliza el modelo preentrenado de EasyOCR para extraer texto de manera eficiente.
"""

import easyocr
from Agent.orchestrator_keys import OrchestratorState
from Nodes.utils_models import read_img

# 1. Carga del modelo
reader = easyocr.Reader(['es', 'en'])

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
        return "[ERROR en tool_describe_scene]: No se ha proporcionado ninguna imagen."
    if image is None:
        return "[ERROR en tool_describe_scene]: No se pudo cargar la imagen con el img_base64 proporcionado."

    if isinstance(image, tuple):
        # Es una tupla de error, por ej. (None, "mensaje...")
        error_message = image[1]
        return {"tool_outputs": {"extract_text_ocr": error_message}}
    else:
        # Es un objeto de imagen válido
        image = np.array(image)

    # 3. EasyOCR
    try:
        resultados = reader.readtext(image, detail=0, paragraph=True)
        texto_extraido = "\n".join(resultados)
        
        if not texto_extraido:
            texto_extraido = "[INFO] No se detectó texto en la imagen."

    except Exception as e:
        error_msg = f"[ERROR en node_extract_text_ocr]: Ocurrió un error en EasyOCR: {e}"
        print(error_msg)
        return {"tool_outputs": {"extract_text_ocr": error_msg}}

    # 4. Guardar el resultado en el estado
    tool_outputs = state.get("tool_outputs", {})
    tool_outputs["extract_text_ocr"] = texto_extraido
    
    
    # 5. Devolvemos el diccionario para actualizar el estado de LangGraph
    return {"tool_outputs": tool_outputs}
