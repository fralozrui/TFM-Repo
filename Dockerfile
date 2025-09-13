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

# Directorio de trabajo
WORKDIR /app

# Copiamos requirements e instalamos dependencias
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Directorio para caché de modelos HF
RUN mkdir -p /app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

### !Predescarga de modelos necesarios para evitar descargas en tiempo de ejecución

# 1) Blip VQA
RUN python -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('Salesforce/blip-vqa-base', cache_dir='/app/hf_cache')"
RUN python -c "from transformers import BlipForQuestionAnswering; BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base', cache_dir='/app/hf_cache')"

# 2) Helsinki-NLP opus-mt-es-en (español → inglés)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en', cache_dir='/app/hf_cache')"
RUN python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-es-en', cache_dir='/app/hf_cache')"

# 3) Helsinki-NLP opus-mt-en-es (inglés → español) <-- este es el que faltaba
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='/app/hf_cache')"
RUN python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='/app/hf_cache')"

# 4) Microsoft git-base (image-to-text)
RUN python -c "from transformers import GitProcessor; GitProcessor.from_pretrained('microsoft/git-base', cache_dir='/app/hf_cache')"
RUN python -c "from transformers import GitForCausalLM; GitForCausalLM.from_pretrained('microsoft/git-base', cache_dir='/app/hf_cache')"

# Copiamos el resto del proyecto
COPY . .

# Exponemos el puerto usado por Cloud Run
EXPOSE 8080

# Arranque de la API usando la variable PORT de Cloud Run
CMD ["sh", "-c", "uvicorn App.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
