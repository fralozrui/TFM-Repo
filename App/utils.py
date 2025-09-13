"""
Funciones auxiliares reutilizables en distintos módulos del proyecto: manejo de archivos, logging,
validaciones comunes y operaciones utilitarias generales.
"""
from typing import Optional, Dict, Any
from typing_extensions import TypedDict

class OrchestratorRequest(TypedDict, total=False):
    api_key: str
    session_id: Optional[str] = None
    user_input: str
    img_base64: Optional[str] = None
    messages: Optional[list] = []
        
def serialize_messages(messages):
        serialized = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                serialized.append({"role": msg.type, "content": msg.content})
            else:
                serialized.append(str(msg))
        return serialized

def create_session_cloud(initial_payload: Dict[str, Any], fs_client, FIRESTORE_COLLECTION) -> str:
    import uuid
    from datetime import datetime
    session_id = str(uuid.uuid4())
    doc_ref = fs_client.collection(FIRESTORE_COLLECTION).document(session_id)
    data = {
        "session_id": session_id,
        "user_input": initial_payload.get("user_input", ""),
        "img": True if initial_payload.get("img_base64", "") != "" else False,
        "messages": initial_payload.get("messages", []),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    doc_ref.set(data)
    return session_id, data

def load_session_cloud(session_id: str, fs_client, FIRESTORE_COLLECTION) -> Dict[str, Any]:
        doc = fs_client.collection(FIRESTORE_COLLECTION).document(session_id).get()
        if not doc.exists:
            return {}
        return doc.to_dict()

def save_session_cloud(session_id: str, state: Dict[str, Any], fs_client, FIRESTORE_COLLECTION):
    doc_ref = fs_client.collection(FIRESTORE_COLLECTION).document(session_id)
    doc = doc_ref.get() 
    state_copy = dict(state)
    if "messages" in state_copy:
        state_copy["messages"] = serialize_messages(state_copy["messages"])
    if "error_history" in state_copy:
        state_copy["error_history"] = serialize_messages(state_copy["error_history"])
    if not doc.exists:
        doc_dict = {}
    else:
        doc_dict = doc.to_dict()
    doc_dict.update(state_copy)
    doc_ref.set(doc_dict, merge=False)

    
def save_image_to_gcs(img_base64: str, session_id: str) -> str:
    """Guarda imagen en Cloud Storage y devuelve la URL pública."""
    import base64, uuid, os
    from datetime import datetime
    from google.cloud import storage

    storage_client = storage.Client()
    BUCKET_NAME = os.environ.get("GCP_BUCKET", "argoos-images")
    # Decodificar base64
    img_bytes = base64.b64decode(img_base64)
    
    # Nombre único
    img_name = f"{session_id}/{uuid.uuid4().hex}.jpg"

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(img_name)
    blob.upload_from_string(img_bytes, content_type="image/jpeg")

    # Opcional: URL firmada temporal (24h)
    url = blob.generate_signed_url(expiration=3600 * 24)

    return url