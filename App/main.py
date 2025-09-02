"""
Archivo principal de ejecución de la aplicación. 
Se encarga de inicializar el orquestador LangGraph, la interfaz de usuario (Gradio) 
y establecer el flujo de entrada y salida entre el usuario y el agente. 
Este archivo sirve como punto de entrada del sistema completo.
"""
# App/main.py
import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from google.cloud import secretmanager, firestore
from starlette.middleware.cors import CORSMiddleware
from Agent.orchestrator import agent 

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator-api")

# Config
PROJECT_ID = os.environ.get("GCP_PROJECT")  # set by Cloud Run env
API_KEY_SECRET_NAME = os.environ.get("API_KEY_SECRET_NAME")  # secret name in Secret Manager
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "sessions")
DEFAULT_SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", 3600 * 2))  # 2h default

# Initialize clients
sm_client = secretmanager.SecretManagerServiceClient()
fs_client = firestore.Client()

# Retrieve the API key value from Secret Manager (we will cache it)
_cached_api_key = None
def get_api_key_from_secret():
    global _cached_api_key
    if _cached_api_key:
        return _cached_api_key
    if not API_KEY_SECRET_NAME:
        logger.warning("API_KEY_SECRET_NAME not set")
        return None
    name = f"projects/{PROJECT_ID}/secrets/{API_KEY_SECRET_NAME}/versions/latest"
    resp = sm_client.access_secret_version(name=name)
    _cached_api_key = resp.payload.data.decode("utf-8")
    return _cached_api_key

# FastAPI app
app = FastAPI(title="Argos")

# CORS — ajusta origen si tu frontend tiene dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia a tu dominio en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrchestratorRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str
    img: Optional[bool] = False
    img_base64: Optional[str] = None

# Session helpers
def create_session(initial_payload: Dict[str, Any]) -> str:
    session_id = str(uuid.uuid4())
    doc_ref = fs_client.collection(FIRESTORE_COLLECTION).document(session_id)
    data = {
        "session_id": session_id,
        "messages": initial_payload.get("messages", []),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    }
    doc_ref.set(data)
    return session_id

def load_session(session_id: str) -> Dict[str, Any]:
    doc = fs_client.collection(FIRESTORE_COLLECTION).document(session_id).get()
    if not doc.exists:
        return {}
    return doc.to_dict()

def save_session(session_id: str, state: Dict[str, Any]):
    doc_ref = fs_client.collection(FIRESTORE_COLLECTION).document(session_id)
    # put a TTL timestamp (expires_at)
    from datetime import datetime, timedelta
    state_copy = dict(state)
    state_copy["updated_at"] = firestore.SERVER_TIMESTAMP
    # if you want TTL in seconds
    state_copy["expires_at"] = firestore.SERVER_TIMESTAMP
    doc_ref.set(state_copy, merge=True)

# API endpoint
@app.post("/orchestrate")
async def orchestrate(req: OrchestratorRequest, x_api_key: Optional[str] = Header(None)):
    # 1) Basic auth via API key
    expected_key = get_api_key_from_secret()
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2) Load or create session
    session_id = req.session_id
    if session_id:
        session = load_session(session_id)
        if not session:
            # start fresh if not found
            session = {}
    else:
        session = {}
        session_id = str(uuid.uuid4())

    # Build initial state for langgraph app
    init_state = {
        "user_input": req.user_input,
        "img": bool(req.img),
        "malprompt": False,
        "attempts": session.get("attempts", 0),
        "tools": session.get("tools", []),
        "justification": session.get("justification", ""),
        "tool_outputs": session.get("tool_outputs", {}),
        "final_response": "",
        "pending_error": session.get("pending_error", False),
        "validated": session.get("validated", False),
        "val_just": session.get("val_just", ""),
        "error_history": session.get("error_history", []),
        "messages": session.get("messages", []),
        "created_at": result.get("created_at", firestore.SERVER_TIMESTAMP),
        "expires_at": result.get("expires_at", firestore.SERVER_TIMESTAMP)
    }

    # Optionally append the user message to messages (so prompts can see history)
    from langchain_core.messages import HumanMessage
    init_state["messages"] = init_state.get("messages", []) + [HumanMessage(content=req.user_input)]

    # 3) Call the langgraph app synchronously
    try:
        result = agent.invoke(init_state)
    except Exception as e:
        logger.exception("LangGraph invocation failed")
        raise HTTPException(status_code=500, detail=str(e))

    # 4) Persist session state (only the fields you want)
    save_session(session_id, {
        "attempts": result.get("attempts", 0),
        "tools": result.get("tools", []),
        "justification": result.get("justification", ""),
        "tool_outputs": result.get("tool_outputs", {}),
        "final_response": result.get("final_response", ""),
        "pending_error": result.get("pending_error", False),
        "validated": result.get("validated", False),
        "val_just": result.get("val_just", ""),
        "error_history": result.get("error_history", []),
        "messages": result.get("messages", []),
        "created_at": result.get("created_at", firestore.SERVER_TIMESTAMP),
        "expires_at": result.get("expires_at", firestore.SERVER_TIMESTAMP)
    })

    return {
        "session_id": session_id,
        "final_response": result.get("final_response", ""),
        "state": {
            "validated": result.get("validated", False),
            "val_just": result.get("val_just", "")
        }
    }
