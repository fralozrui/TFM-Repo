"""
Archivo principal de ejecución de la aplicación. 
Se encarga de inicializar el orquestador LangGraph, la interfaz de usuario (Gradio) 
y establecer el flujo de entrada y salida entre el usuario y el agente. 
Este archivo sirve como punto de entrada del sistema completo.
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, APIRouter
from pydantic import BaseModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Agent.orchestrator import agent
from App.utils import save_image_to_gcs, OrchestratorRequest, serialize_messages, create_session_cloud, load_session_cloud, save_session_cloud

# Detectar entorno
entorno = 'cloud'

# --------------------------
# Rama CLOUD (GCP / no local)
# --------------------------
if entorno != "local":
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("orchestrator-api")

    # Config
    from google.cloud import secretmanager, firestore
    PROJECT_ID = os.environ.get("GCP_PROJECT")  
    ORCHESTRATOR_API_KEY = os.environ.get("ORCHESTRATOR_API_KEY")  
    FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "sessions")

    # Initialize clients
    sm_client = secretmanager.SecretManagerServiceClient()
    fs_client = firestore.Client()

    # --- API definition ---
    app = FastAPI()
    router = APIRouter()

    @router.post("/orchestrate")
    async def orchestrate(req: OrchestratorRequest):
        # 1) Auth
        x_api_key = req.get("api_key","")
        if ORCHESTRATOR_API_KEY and x_api_key != ORCHESTRATOR_API_KEY:
            return {"status_code":401, "message":"Invalid Key"}

        # 2) Load or create session
        session = {}
        if req.get("session_id",""):
            session = load_session_cloud(req.get("session_id",""), fs_client, FIRESTORE_COLLECTION)
            if not session:
                session_id, session = create_session_cloud(req, fs_client, FIRESTORE_COLLECTION)
            else:
                session.update(req)
                session_id = req.get("session_id","")
        else:
            session_id, session = create_session_cloud(req, fs_client, FIRESTORE_COLLECTION)

        # 3) Build init_state for LangGraph
        img_url = ''
        img_base64 = req.get("img_base64","")
        if img_base64:
            try:
                img_url = save_image_to_gcs(img_base64, session_id)
            except Exception as e:
                logger.error(f"Error al guardar imagen en GCS: {e}")
                img_url = ''

        init_state = {
            "user_input": session.get("user_input", req["user_input"]),
            "img_base64": img_base64,
            "img": session.get("img",False),
            "img_url": img_url,
            "malprompt": False,
            "attempts": session.get("attempts", 0),
            "run_tools": session.get("run_tools", []),
            "justification": session.get("justification", ""),
            "tool_outputs": session.get("tool_outputs", {}),
            "final_response": "",
            "pending_error": session.get("pending_error", False),
            "validated": session.get("validated", False),
            "val_just": session.get("val_just", ""),
            "error_history": session.get("error_history", []),
            "messages": session.get("messages", []),
            "created_at": session.get("created_at", datetime.utcnow().isoformat() + "Z"),
        }


        # --- Depuración ---
        print("DEBUG INIT STATE img:", init_state.get("img"))
        print("DEBUG INIT STATE img_base64:", "sí" if init_state.get("img_base64") else "no")
        print("DEBUG INIT STATE user_input:", init_state.get("user_input"))

        if init_state["user_input"]:
            from langchain_core.messages import HumanMessage
            init_state["messages"] = init_state.get("messages", []) + [HumanMessage(content=init_state["user_input"])]

            try:
                result = agent.invoke(init_state)
            except Exception as e:
                logger.exception("LangGraph invocation failed")
                raise HTTPException(status_code=500, detail=str(e))

            # 4) Persist updated state
            save_session_cloud(session_id, {
                "user_input": init_state["user_input"],
                "img_url": result.get("img_url",""),
                "img": result.get("img", False),
                "malprompt": result.get("malprompt", False),
                "attempts": result.get("attempts", 0),
                "run_tools": result.get("run_tools", []),
                "justification": result.get("justification", ""),
                "tool_outputs": result.get("tool_outputs", {}),
                "final_response": result.get("final_response", ""),
                "pending_error": result.get("pending_error", False),
                "validated": result.get("validated", False),
                "val_just": result.get("val_just", ""),
                "error_history": result.get("error_history", []),
                "messages": result.get("messages", []),
                "created_at": result.get("created_at", datetime.utcnow().isoformat() + "Z"),
            }, fs_client, FIRESTORE_COLLECTION)

            return {
                "session_id": session_id,
                "final_response": result.get("final_response", ""),
                "state": {
                    "validated": result.get("validated", False),
                    "val_just": result.get("val_just", "")
                }
            }
    app.include_router(router)


# --------------------------
# Rama LOCAL
# --------------------------
else:
    logger = logging.getLogger("orchestrator-local")

    # --- API definition ---
    app = FastAPI()
    router = APIRouter()

    # Cargar base de datos desde archivo JSON
    DATABASE_PATH = r"Pruebas\Database\Database_v1.json"
    try:
        with open(DATABASE_PATH, "r", encoding="utf-8") as f:
            Database_v1 = json.load(f)
    except FileNotFoundError:
        Database_v1 = {}

    # Función para guardar la base de datos en disco
    def save_database():
        db_copy = Database_v1.copy()
        # conviertes solo la parte que contiene mensajes
        for session_id, session_data in db_copy.items():
            if "messages" in session_data:
                session_data["messages"] = serialize_messages(session_data["messages"])
            if "error_history" in session_data:
                session_data["error_history"] = serialize_messages(session_data["error_history"])

        with open(DATABASE_PATH, "w", encoding="utf-8") as f:
            json.dump(db_copy, f, ensure_ascii=False, indent=2)
            
    # Session helpers
    def create_session(initial_payload: Dict[str, Any]) -> str:
        # Genera un session_id único que no esté en la base de datos
        while True:
            session_id = str(uuid.uuid4())
            if session_id not in Database_v1:
                break
        img_id = initial_payload.get("img_id", None)
        img_bool = False if img_id is None else True
        
        data = {
            session_id: {
                "user_input": initial_payload.get("user_input", ""),
                "img": img_bool,
                "img_id": img_id,
                "messages": initial_payload.get("messages", []),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        }
        return session_id, data

    def load_session(session_id: str) -> Dict[str, Any]:
        if session_id not in Database_v1.keys():
            return {}
        return Database_v1[session_id]

    def save_session(session_id: str, state: Dict[str, Any]):
        if session_id in Database_v1.keys():
            Database_v1[session_id].update(state)
        else:
            Database_v1[session_id] = state

    @router.post("/orchestrate")
    async def orchestrate(req: OrchestratorRequest, x_api_key: Optional[str] = Header(None)):
        try:
            session_id = req.get("session_id")
            if session_id:
                session = load_session(session_id)
                session.update(req) 
                if not session:
                    # start fresh if not found
                    session_id, created_session = create_session(req)
                    session = created_session[session_id]
                else:
                    session.update(req)
            else:
                session_id, created_session = create_session(req)
                session = created_session[session_id]

            init_state = {
                "session_id": session_id,
                "user_input": session.get("user_input", req.get("user_input", "")),
                "img": session.get("img", False),
                "img_id": session.get("img_id", None),
                "malprompt": False,
                "attempts": session.get("attempts", 0),
                "run_tools": session.get("run_tools", []),
                "justification": session.get("justification", ""),
                "tool_outputs": session.get("tool_outputs", {}),
                "final_response": "",
                "pending_error": session.get("pending_error", False),
                "validated": session.get("validated", False),
                "val_just": session.get("val_just", ""),
                "error_history": session.get("error_history", []),
                "messages": session.get("messages", []),
                "created_at": session.get("created_at", datetime.utcnow().isoformat() + "Z"),
            }
            # --- Depuración ---
            print("DEBUG INIT STATE session_id:", init_state.get("session_id"))
            print("DEBUG INIT STATE img:", init_state.get("img"))
            print("DEBUG INIT STATE user_input:", init_state.get("user_input"))

            if init_state["user_input"]:

                # Añadir el input del usuario al historial de mensajes
                from langchain_core.messages import HumanMessage
                init_state["messages"] = init_state.get("messages", []) + [HumanMessage(content=init_state["user_input"])]

                # Llamada al agente de manera asíncrona 
                try:
                    result = agent.invoke(init_state)
                except Exception as e:
                    logger.exception("LangGraph invocation failed")
                    raise HTTPException(status_code=500, detail=str(e))

                save_session(session_id, {
                    "user_input": init_state["user_input"],
                    "img": result.get("img", False),
                    "img_id": result.get("img_id", None),
                    "malprompt": result.get("malprompt", False),
                    "attempts": result.get("attempts", 0),
                    "run_tools": result.get("run_tools", []),
                    "justification": result.get("justification", ""),
                    "tool_outputs": result.get("tool_outputs", {}),
                    "final_response": result.get("final_response", ""),
                    "pending_error": result.get("pending_error", False),
                    "validated": result.get("validated", False),
                    "val_just": result.get("val_just", ""),
                    "error_history": result.get("error_history", []),
                    "messages": result.get("messages", []),
                    "created_at": result.get("created_at", datetime.utcnow().isoformat() + "Z"),
                })

                try:
                    save_database()
                except Exception as e:
                    print(f"Error saving database: {str(e)}")

                return {
                    "status": 200,
                    "session_id": session_id,
                    "final_response": result.get("final_response", ""),             
                        }
            else:
                return {
                    "status": 400,
                    "error": "An input is required"
                }
        except Exception as e:
            return{
                "status": 500,
                "error":str(e)
            }
    app.include_router(router)
