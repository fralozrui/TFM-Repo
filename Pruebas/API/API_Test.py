import sys
import os
import requests
import json
import base64

# ----------------------------
# Añadir la carpeta raíz al path
# ----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ----------------------------
# Configuración de la imagen
# ----------------------------
img_id = 2  # Cambia el ID según la imagen que quieras usar (en el frontend debería ser dinámico y crear el id)
IMAGE_PATH = fr"Pruebas\Database\Images\{img_id}.jpg"  # Cambia por tu ruta
with open(IMAGE_PATH, "rb") as f:
    img_bytes = f.read()

img_base64_str = base64.b64encode(img_bytes).decode("utf-8")

# ----------------------------
# Payload de la API
# ----------------------------
init_req = {
    "session_id": None,
    "img_id": img_id,
    "user_input": "¿Qué se ve en la foto?",
    "messages": []
}

# ----------------------------
# Hacer petición POST
# ----------------------------
try:
    response = requests.post(url="http://127.0.0.1:8000/orchestrate", json=init_req)
    response.raise_for_status()
    json_response = response.json()
    
    print("\n=== RESPUESTA DE LA API ===")
    print(json.dumps(json_response, indent=2, ensure_ascii=False))
    print("\n=== TOOLS OUTPUT ===")
    tools_output = json_response.get("state", {}).get("tools_output", {})
    if tools_output:
        for tool, output in tools_output.items():
            print(f"{tool}: {output}")

except requests.exceptions.RequestException as e:
    print("Error en la petición:", str(e))
except json.JSONDecodeError:
    print("No se pudo parsear la respuesta como JSON")
    print("Respuesta:", response.text)