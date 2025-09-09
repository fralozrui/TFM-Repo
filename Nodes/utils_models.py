"""
Define funciones comunes para cargar y ejecutar modelos compartidos entre nodos,
como abstracciones de inferencia, carga desde HuggingFace, cacheo y gestión de dispositivos (CPU/GPU).
"""

def read_img(img_id: int):
    import os
    import io
    import base64
    from PIL import Image
    img_path = os.path.join("Pruebas", "Database", "Images", f"{img_id}.jpg")
    if not os.path.exists(img_path):
        return None, f"[ERROR] La imagen con ID {img_id} no existe en la ruta {img_path}."
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        img_base64_str = base64.b64encode(img_bytes).decode("utf-8")
        img_bytes = base64.b64decode(img_base64_str)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return image
    
    except:
        try:
            image = Image.open(img_path).convert("RGB")
            return image
        except Exception as e:
            return None

# Diccionario TOOLS
from Nodes.ocr_extractor import tool_ocr
from Nodes.object_localizer import tool_object_detection
from Nodes.scene_description import tool_describe_scene

def tool_modelo(img_input, is_base64=False):
    print("=== tool_modelo ===")
    print("Tipo de img_input recibido:", type(img_input))
    print("is_base64:", is_base64)
    
    from PIL import Image
    from io import BytesIO
    import base64
    
    if is_base64:
        print("[DEBUG] Decodificando imagen base64...")
        try:
            img_bytes = base64.b64decode(img_input)
            print("Longitud de bytes decodificados:", len(img_bytes))
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print("[ERROR] No se pudo decodificar base64:", e)
            return f"Error decodificando base64: {e}"
    else:
        if isinstance(img_input, bytes):
            print("[DEBUG] Abrir imagen desde bytes...")
            try:
                img = Image.open(BytesIO(img_input)).convert("RGB")
            except Exception as e:
                print("[ERROR] No se pudo abrir imagen desde bytes:", e)
                return f"Error abriendo bytes: {e}"
        else:
            print("[DEBUG] Abrir imagen desde ruta o file-like...")
            print("Valor de img_input:", img_input)
            try:
                img = Image.open(img_input).convert("RGB")
            except Exception as e:
                print("[ERROR] No se pudo abrir imagen desde ruta/file-like:", e)
                return f"Error abriendo ruta/file-like: {e}"

    print("[DEBUG] Imagen cargada correctamente:", img)
    
    # aquí tu código de generación de caption
    try:
        caption = tool_describe_scene(img)
        print("[DEBUG] Caption generado:", caption)
        return caption
    except Exception as e:
        print("[ERROR] Error al generar caption:", e)
        return f"Error generando caption: {e}"
TOOLS = {
    "ocr": tool_ocr,
    "object_detection": tool_object_detection,
    "imagen": tool_describe_scene
}