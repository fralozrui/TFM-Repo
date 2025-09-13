"""
Nodo especializado en la descripción automática de escenas a partir de imágenes.
Utiliza modelos preentrenados como BLIP, GIT o similares para generar descripciones 
semánticas útiles.
"""

from PIL import Image
import base64
import io
import torch
from transformers import AutoProcessor, GitForCausalLM
from Agent.orchestrator_keys import OrchestratorState  # Asegúrate de importar correctamente
from Nodes.utils_models import read_img

# Forzar uso de CPU
device = torch.device("cpu")

# Cargar modelo y procesador
model_name = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(model_name)
model = GitForCausalLM.from_pretrained(model_name).to(device)

def tool_describe_scene(state: OrchestratorState) -> str:
    """
    Función adaptada para recibir un objeto OrchestratorState.
    Extrae imagen (base64 o ruta) y genera una descripción de la escena.
    """
    print("[DEBUG] Ejecutando tool_describe_scene...")
    img_base64 = state.get("img_base64", "")
    if img_base64:
        image = read_img(img_base64)  
    else:
        return "[ERROR en tool_describe_scene]: No se ha proporcionado ninguna imagen."
    if image is None:
        return "[ERROR en tool_describe_scene]: No se pudo cargar la imagen con el img_base64 proporcionado."

    # Procesar y generar descripción
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.1,
            num_return_sequences=1,
            pad_token_id=model.config.pad_token_id or model.config.eos_token_id
        )

    # Decodificar resultado
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Guardar en el estado
    if "tool_outputs" not in state:
        state["tool_outputs"] = {}
    state["tool_outputs"]["describe_scene"] = caption

    return caption
