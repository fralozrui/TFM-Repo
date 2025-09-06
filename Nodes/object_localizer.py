"""
Nodo responsable de detectar y localizar objetos específicos en imágenes,
utilizando modelos como YOLOv8 o Grounding DINO. 
Permite responder preguntas como “¿dónde está el mando?” o “¿ves un enchufe?”.
"""
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline
from PIL import Image
import torch, io, base64

# Inicializar una sola vez (mejor rendimiento)
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

def define_location(image: Image.Image, question_es: str):
    """Recibe una imagen y pregunta en español, devuelve la respuesta en español"""
    question_en = translate(question_es, "es-en")
    inputs = processor(images=image, text=question_en, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)

    answer_en = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    answer_es = translate(answer_en, "en-es").strip()
    return answer_es

def tool_object_detection(state: dict):
    """Wrapper para integrarlo en utils_models"""
    img_input = state.get("img_base64") or state.get("img")
    question_es = state.get("user_input", "")

    if state.get("img_base64"):
        img_bytes = base64.b64decode(img_input)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        image = Image.open(img_input).convert("RGB")

    print("[DEBUG] Ejecutando object_localizer con pregunta:", question_es)
    answer_es = define_location(image, question_es)
    print("[DEBUG] Respuesta obtenida:", answer_es)
    return {
        "tool": "object_detection",
        "question": question_es,
        "answer": answer_es
    }