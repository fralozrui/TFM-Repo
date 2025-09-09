from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated, Literal, Dict, Any, List
from pydantic import BaseModel
from langgraph.graph.message import add_messages
import google.generativeai as genai
import pandas as pd

class OrchestratorState(TypedDict, total=False):
    user_input: str
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
genai.configure(api_key = 'AIzaSyDYfdOJUzU1WZJnIDYXkEQF5NmdP1OwnyQ')
# genai.configure(api_key = userdata.get('GOOGLE_API_KEY'))
model_gemini = genai.GenerativeModel("gemini-2.5-flash")