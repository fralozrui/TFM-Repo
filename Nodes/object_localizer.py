from typing import Dict, Any
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict, Literal, List
from langgraph.graph.message import add_messages

from Agent.orchestrator_keys import OrchestratorState
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline
from PIL import Image
import torch, io, base64
from Nodes.utils_models import read_img


processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


def translate(text: str, direction: str = "es-en") -> str:
    if not text.strip():
        return text
    if direction == "es-en":
        return translator_es_en(text)[0]["translation_text"]
    elif direction == "en-es":
        return translator_en_es(text)[0]["translation_text"]
    else:
        raise ValueError("Dirección de traducción no soportada. Usa 'es-en' o 'en-es'.")

def define_location(image: Image.Image, question_es: str) -> str:
    question_en = translate(question_es, "es-en")
    inputs = processor(images=image, text=question_en, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    answer_en = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return translate(answer_en, "en-es").strip()


def tool_object_detection(state: OrchestratorState) -> str:
    try:
        question_es = state.get("user_input", "")
        img_id = state.get("img_id", None)
        if img_id is not None:
            image = read_img(img_id)  
        else:
            return "[ERROR en tool_object_detection]: No se ha proporcionado img_id en el estado."
        if image is None:
            return "[ERROR en tool_object_detection]: No se pudo cargar la imagen con el img_id proporcionado."
        
        # img_input = state.get("img_base64") or state.get("img")

        # if not img_input:
        #     return "[ERROR]: No se ha proporcionado imagen."

        # Decodificar imagen
        # if state.get("img_base64"):
        #     img_bytes = base64.b64decode(img_input)
        #     image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # else:
        #     image = Image.open(img_input).convert("RGB")

        print("[DEBUG] Ejecutando object_localizer con pregunta:", question_es)
        respuesta = define_location(image, question_es)
        print("[DEBUG] Respuesta obtenida:", respuesta)

        # Guardar en el estado (opcional si se usa fuera del LangGraph también)
        if "tool_outputs" not in state:
            state["tool_outputs"] = {}
        state["tool_outputs"]["object_detection"] = respuesta

        return respuesta
    except Exception as e:
        return f"[ERROR en tool_object_detection]: {str(e)}"
