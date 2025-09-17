"""
Define la interfaz de usuario usando Gradio: permite introducir texto, audio e imágenes,
y muestra las respuestas del agente. Actúa como capa de interacción visual accesible para pruebas y despliegue web.
"""
import gradio as gr
import requests
import base64
import pyttsx3
import tempfile
import os
import mimetypes
import whisper

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

class AppState:
    def __init__(self, conversation=None, last_text="", current_image=None):
        self.conversation = conversation or []
        self.last_text = last_text
        self.current_image = current_image  # foto fija que se envía siempre

# Whisper small
modelo_whisper = whisper.load_model("small")
def transcribir_audio(filepath):
    if not filepath:
        return ""
    return modelo_whisper.transcribe(filepath, language="es")["text"]

# TTS → genera un archivo temporal y devuelve su ruta
def hablar(texto, filename="respuesta.wav"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es_' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.save_to_file(texto, filename)
    engine.runAndWait()
    return filename
def hablar_cloud(texto):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        filename = f.name
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es_' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.save_to_file(texto, filename)
    engine.runAndWait()
    return filename

def imagen_a_base64(img_path):
    if not img_path:
        return None
    mime_type, _ = mimetypes.guess_type(img_path)
    if not mime_type:
        mime_type = "image/png"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def render_chat(conversation):
    html = "<div id='chat_box' style='height:400px; overflow-y:auto; padding:10px; font-family:Arial; background:black;'>"
    html += "<style>@keyframes fadein{from{opacity:0}to{opacity:1}}.msg{animation:fadein 0.3s;margin:6px 0;padding:10px 14px;border-radius:20px;max-width:70%;display:inline-block;word-wrap:break-word;font-size:15px;}.user{background-color:#ffa500;color:white;float:right;clear:both;text-align:right}.assistant{background-color:#7d7d7d;color:white;float:left;clear:both;text-align:left}img.chat-img{max-width:120px;border-radius:10px;margin:5px 0;display:block}</style>"
    for msg in conversation:
        role = msg.get('role','user')
        content = msg.get('content','')
        msg_type = msg.get('type','text')
        if msg_type == "image":
            html_content = f"<img src='{content}' class='chat-img'>"
        else:
            html_content = content
        if role == "user":
            html += f"<div class='msg user'>{html_content}</div>"
        elif role == "assistant":
            html += f"<div class='msg assistant'>{html_content}</div>"
    html += "<script>var objDiv=document.getElementById('chat_box');objDiv.scrollTop=objDiv.scrollHeight;</script></div>"
    return html

# -------------------------------
# Función de envío a API
# -------------------------------
def enviar_a_api(user_text, state: AppState):
    img_base64 = ""
    if state.current_image:
        img_base64 = base64.b64encode(open(state.current_image, "rb").read()).decode("utf-8")
        if not any(m.get('type') == 'image' and m.get('content') == imagen_a_base64(state.current_image) for m in state.conversation):
            state.conversation.append({"role": "user", "type": "image", "content": imagen_a_base64(state.current_image)})

    user_text_api = user_text if user_text.strip() else ("que ves en la foto" if img_base64 else "")

    if user_text.strip():
        state.conversation.append({"role": "user", "content": user_text})

    payload = {
        "api_key": API_KEY,
        "user_input": user_text_api,
        "session_id": None,
        "img_base64": img_base64,
        "messages": []
    }

    respuesta = "[Error: no se obtuvo respuesta de la API]"
    try:
        response = requests.post(API_URL, json=payload)
        if response and response.ok and response.content:
            data = response.json()
            if isinstance(data, dict):
                respuesta = data.get("final_response", "[Sin respuesta]")
    except Exception as e:
        respuesta = f"[Error al conectar con API: {e}]"

    state.conversation.append({"role": "assistant", "content": respuesta})

    # Generar archivo de audio
    temp_wav = hablar_cloud(respuesta)

    # 🔑 Convertir archivo a bytes para Gradio
    with open(temp_wav, "rb") as f:
        audio_bytes = f.read()

    # (Opcional) limpiar archivo temporal
    try:
        os.remove(temp_wav)
    except OSError:
        pass

    return render_chat(state.conversation), audio_bytes, state

# -------------------------------
# Interfaz Gradio
# -------------------------------
with gr.Blocks() as demo:
    state_gr = gr.State(value=AppState())
    gr.HTML("<div style='background-color:#ff4500;color:white;padding:10px;text-align:center;font-size:24px;font-weight:bold;'>ARGOOS</div>")
    chat_area = gr.HTML(render_chat(state_gr.value.conversation), elem_id="chat_area")

    with gr.Row(equal_height=True):
        record_button = gr.Audio(sources="microphone", type="filepath", label="🎤 Grabar", elem_id="record_button", interactive=True)
        upload_image = gr.Image(label="Subir foto", type="filepath", elem_id="upload_image", interactive=True)

    with gr.Row(equal_height=True):
        input_text = gr.Textbox(placeholder="Escribe un mensaje...", show_label=False, lines=1, max_lines=3, scale=5)
        send_button = gr.Button("Enviar", scale=1)

    # 🔑 El output de enviar_a_api será audio_bytes
    output_audio = gr.Audio(label="", autoplay=True, show_label=False, elem_id="output_audio", interactive=False)

    def upload_photo_callback(img_file, state):
        if img_file:
            state.current_image = img_file
            img_data_uri = imagen_a_base64(img_file)
            if not any(m.get('type') == 'image' and m.get('content') == img_data_uri for m in state.conversation):
                state.conversation.append({"role": "user", "type": "image", "content": img_data_uri})
        return render_chat(state.conversation), state

    upload_image.change(upload_photo_callback, [upload_image, state_gr], [chat_area, state_gr])

    def send_message(user_text, state: AppState):
        chat_html, audio_bytes, new_state = enviar_a_api(user_text, state)
        return chat_html, audio_bytes, new_state, ""  # limpiar textbox

    send_button.click(send_message, [input_text, state_gr], [chat_area, output_audio, state_gr, input_text])

    def audio_callback(audio_path, state):
        if audio_path:
            texto = transcribir_audio(audio_path)
            return enviar_a_api(texto, state)
        return render_chat(state.conversation), None, state

    record_button.change(audio_callback, [record_button, state_gr], [chat_area, output_audio, state_gr])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=False, show_api=False)
