imagen =  r"C:\Users\maria\OneDrive\Documentos 1\Escritorio\Imágenes\Capturas de pantalla\Captura de pantalla 2025-01-13 171918.png"

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import sys

def define_location(image_path:str, question:str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=100)

    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python modelo_consulta_ubicacion.py ruta/a/imagen.jpg [preunta_en_ingles]")
        sys.exit(0)

    image_path = sys.argv[1]
    question = sys.argv[2]
    if len(sys.argv) == 3:
        answer = define_location(image_path, question)
        print(question)
        print(answer)


