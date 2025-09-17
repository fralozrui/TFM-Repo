# ----------------------------- Dockerfile (Frontend) -----------------------------
# Imagen base ligera de Python
FROM python:3.10-slim

# Dependencias del sistema necesarias (ffmpeg para whisper, espeak para pyttsx3)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak \
    libespeak-ng1 \
    libsndfile1 \
    && apt-get clean \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiamos requirements e instalamos dependencias
COPY frontend_requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r frontend_requirements.txt

# Descargamos el modelo Whisper en el build para evitar descargas en runtime
RUN python -c "import whisper; whisper.load_model('base')"

# Copiamos el resto del proyecto (frontend con Gradio)
COPY . .

# Exponer el puerto que usa Gradio
EXPOSE 8080

# Arranque de la aplicación con Gradio en Cloud Run
CMD ["python", "gradio_interface.py"]
