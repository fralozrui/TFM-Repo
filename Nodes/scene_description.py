"""
Nodo especializado en la descripción automática de escenas a partir de imágenes.
Utiliza el modelo BLIP para generar descripciones semánticas útiles.
"""

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from Agent.orchestrator_keys import OrchestratorState  # Asegúrate de importar correctamente
from Nodes.utils_models import read_img

# Forzar uso de CPU
device = torch.device("cpu")

# Cargar modelo y procesador
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

def tool_describe_scene(state: OrchestratorState) -> str:
    """
    Función adaptada para recibir un objeto OrchestratorState.
    Extrae imagen (base64 o ruta) y genera una descripción de la escena.
    """
    print("[DEBUG] Ejecutando tool_describe_scene con BLIP...")

    img_base64 = state.get("img_base64", None)
    if img_base64:
        image = read_img(img_base64)  
    else:
        return "[ERROR en tool_describe_scene]: No se ha proporcionado ninguna imagen."
    if image is None:
        return "[ERROR en tool_describe_scene]: No se pudo cargar la imagen con el img_base64 proporcionado."
    # Procesar y generar descripción
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # Guardar en el estado
    if "tool_outputs" not in state:
        state["tool_outputs"] = {}
    state["tool_outputs"]["describe_scene"] = caption

    return caption