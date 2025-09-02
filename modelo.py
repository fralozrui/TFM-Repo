from PIL import Image
import torch
import sys
from transformers import AutoProcessor, GitForCausalLM

# Forzar CPU
device = torch.device("cpu")

model_name = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(model_name)
model = GitForCausalLM.from_pretrained(model_name).to(device)

def generar_caption(path_imagen):
    image = Image.open(path_imagen).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.1, #Lo estoy dejando así para que sea más creativo pero se puede cambiar
            num_return_sequences=1,
            pad_token_id=model.config.pad_token_id
        )

    captions = [processor.tokenizer.decode(output, skip_special_tokens=True).strip()
                for output in outputs]
    return captions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python modelo.py ruta/a/imagen.jpg")
        exit()
    path = sys.argv[1]
    caption = generar_caption(path)
    print("\n--- Descripción generada ---")
    for cap in caption:
        print(cap)
    print("---------------------------")