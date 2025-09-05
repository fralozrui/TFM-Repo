from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline
from PIL import Image
import torch
import sys


def define_location(image_path:str, question_es:str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image = Image.open(image_path).convert("RGB")
    question_en = translate(question_es, "es-en")
    inputs = processor(image, question_en, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=100)

    answer_en = processor.decode(out[0], skip_special_tokens=True).strip()
    answer_es = translate(answer_en, "en-es").strip()
    return answer_es

def translate(text: str, direction: str = "es-en") -> str:
    translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

    if not text.strip():
        return text
    if direction == "es-en":
        return translator_es_en(text)[0]["translation_text"]
    elif direction == "en-es":
        return translator_en_es(text)[0]["translation_text"]
    else:
        raise ValueError("Dirección de traducción no soportada. Usa 'es-en' o 'en-es'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python modelo_consulta_ubicacion.py ruta/a/imagen.jpg [pregunta_en_español]")
        sys.exit(0)

    image_path = sys.argv[1]
    question = sys.argv[2]
    if len(sys.argv) == 3:
        answer = define_location(image_path, question)
        print(question)
        print(answer)
