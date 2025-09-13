from typing import Dict, Any
from langchain_core.messages import  BaseMessage
from typing_extensions import TypedDict, Annotated, Dict, Any, List
from langgraph.graph.message import add_messages
import google.generativeai as genai
import os

class OrchestratorState(TypedDict, total=False):
    user_input: str
    img_base64: str
    img: bool
    malprompt: bool
    attempts: int
    run_tools: List[str]
    justification: str
    tool_outputs: Dict[str, Any]
    final_response: str
    pending_error: bool
    validated: bool
    val_just: str
    error_history: Annotated[List[str], add_messages]
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    created_at: str
    img_id: int


# Configuración Gemini
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")  
genai.configure(api_key = GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")