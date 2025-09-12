# ----------------------------- Dockerfile -----------------------------
# Define un entorno reproducible para el despliegue del agente multimodal. 
# Instala dependencias necesarias y configura el entorno para ejecutar el 
# modelo de forma portable en HuggingFace Spaces o servidores compatibles.

# Usa Python base oficial
    FROM python:3.10-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    libssl-dev \
    libffi-dev \
    libstdc++6 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos requirements e instalamos dependencias
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

### !Aquí se guardan los modelos cargados en cpu, cambiar si se cambia el modelo
# Descargamos los modelos en la imagen para no depender de HuggingFace en tiempo de arranque
RUN python -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')" \
 && python -c "from transformers import BlipForQuestionAnswering; BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')"

# Copiamos la caché de HF a una ruta estable dentro de la imagen
RUN mkdir -p /app/hf_cache && cp -r /root/.cache/huggingface/* /app/hf_cache/
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Copiamos el resto del proyecto
COPY . .

# Exponemos el puerto usado por Cloud Run
EXPOSE 8080

# Arranque de la API
CMD exec uvicorn App.main:app --host 0.0.0.0 --port ${PORT:-8080}
