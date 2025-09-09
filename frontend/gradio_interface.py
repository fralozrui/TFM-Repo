import gradio as gr
import whisper
import pyttsx3
import tempfile
import os

# Cargar modelo Whisper
modelo = whisper.load_model("base")

# Estado de la conversación
class AppState:
    def __init__(self, conversation=None):
        self.conversation = conversation or []

# TTS con pyttsx3
def hablar(texto, filename="respuesta.wav"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    voz_espanol = None
    for voice in voices:
        if 'spanish' in voice.name.lower() or 'es_' in voice.id.lower():
            voz_espanol = voice.id
            break
    if voz_espanol:
        engine.setProperty('voice', voz_espanol)
    engine.save_to_file(texto, filename)
    engine.runAndWait()
    return filename

# Renderizar chat tipo WhatsApp
def render_chat(conversation):
    html = """
    <div id='chat_box' style='height:400px; overflow-y:auto; padding:10px; font-family:Arial, sans-serif;'>
    <style>
    @keyframes fadein { from {opacity:0;} to {opacity:1;} }
    .msg { animation: fadein 0.3s; margin:6px 0; padding:10px 14px; border-radius:20px; max-width:70%; display:inline-block; word-wrap:break-word; box-shadow: 0 1px 3px rgba(0,0,0,0.2); font-size:15px;color:black; }
    .user { background-color:#3e88e8; float:right; clear:both; text-align:right; }
    .assistant { background-color:#7d7d7d; float:left; clear:both; text-align:left; }
    img { max-width:70%; border-radius:20px; margin:5px 0; }
    </style>
    """
    for msg in conversation:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        msg_type = msg.get('type', 'text')
        if msg_type == "image":
            html_content = f"<img src='{content}'>"
        else:
            html_content = content
        if role == "user":
            html += f"<div class='msg user'>{html_content}</div>"
        else:
            html += f"<div class='msg assistant'>{html_content}</div>"
    html += "<script>var objDiv = document.getElementById('chat_box'); objDiv.scrollTop = objDiv.scrollHeight;</script></div>"
    return html

# Procesar audio
def process_audio(filepath, state: AppState):
    if filepath is None:
        return render_chat(state.conversation), None, state
    resultado = modelo.transcribe(filepath, language="es")
    texto = resultado["text"]
    state.conversation.append({"role":"user","content":texto})
    respuesta = f"Tú dijiste: {texto}"
    state.conversation.append({"role":"assistant","content":respuesta})

    temp_wav = os.path.join(tempfile.gettempdir(), "respuesta.wav")
    hablar(respuesta, temp_wav)

    return render_chat(state.conversation), temp_wav, state

# Procesar texto
def process_text(texto, state: AppState):
    if texto.strip() == "":
        return render_chat(state.conversation), None, state
    state.conversation.append({"role":"user","content":texto})
    respuesta = f"Tú dijiste: {texto}"
    state.conversation.append({"role":"assistant","content":respuesta})

    temp_wav = os.path.join(tempfile.gettempdir(), "respuesta.wav")
    hablar(respuesta, temp_wav)

    return render_chat(state.conversation), temp_wav, state

# Procesar imagen
def process_image(filepath, state: AppState):
    if filepath is None:
        return render_chat(state.conversation), state
    state.conversation.append({"role":"user", "type":"image", "content":filepath})
    return render_chat(state.conversation), state

# Interfaz
with gr.Blocks() as demo:
    state = gr.State(value=AppState())

    # CSS botones y altura fila inferior
    gr.HTML("""
    <style>
    #send_button button {
        background-color: #28a745;
        color: white;
        width: 80px;
        height: 50px;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    #record_button {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        width: 50px;
    }
    #upload_image {
        height: 100px !important;
        width: auto;
    }
    .gr-row > div {
        max-height: 60px;
        display: flex;
        align-items: center;
    }
    </style>
    """)

    # Header azul
    gr.HTML("<div style='background-color:#1E90FF; color:white; padding:10px; text-align:center; font-size:24px; font-weight:bold;'>ARGOOS</div>")

    # Chat
    chat_area = gr.HTML(render_chat(state.value.conversation), elem_id="chat_area")

    # Fila inferior: texto + enviar (más ancho)
    with gr.Row(equal_height=True):
        input_text = gr.Textbox(
            placeholder="Escribe un mensaje...", 
            show_label=False, 
            lines=1, 
            max_lines=3, 
            scale=5
        )
        send_button = gr.Button("Enviar", elem_id="send_button", scale=1)

    # Fila inferior: grabar + subir imagen (menos alto y ancho)
    with gr.Row(equal_height=True):
        record_button = gr.Audio(
            sources="microphone",
            type="filepath",
            label="🎤",
            elem_id="record_button",
            scale=1
        )
        upload_image = gr.Image(
            label="Subir imagen",
            type="filepath",
            elem_id="upload_image",
            scale=1
        )
        output_audio = gr.Audio(
            label="🔊 Respuesta TTS",
            autoplay=True,
            visible=False
        )

    # Conexiones
    upload_image.change(process_image, [upload_image, state], [chat_area, state])
    send_button.click(process_text, [input_text, state], [chat_area, output_audio, state])
    record_button.change(process_audio, [record_button, state], [chat_area, output_audio, state])

if __name__ == "__main__":
    demo.launch(share=True)

