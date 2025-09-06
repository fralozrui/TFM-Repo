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

# Forzar CPU
device = torch.device("cpu")

model_name = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(model_name)
model = GitForCausalLM.from_pretrained(model_name).to(device)

def tool_describe_scene(input_data, is_base64=False):
    """
    input_data: ruta, base64 o PIL.Image
    """
    # Si ya es PIL.Image
    if isinstance(input_data, Image.Image):
        image = input_data
    elif is_base64:
        img_bytes = base64.b64decode(input_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        image = Image.open(input_data).convert("RGB")
    
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
    
    captions = [processor.tokenizer.decode(output, skip_special_tokens=True).strip()
                for output in outputs]
    return captions[0]