"""
Implementa el grafo de nodos LangGraph que define la lógica de flujo del agente multimodal. 
Contiene la estructura de decisión basada en el LLM orquestador que dirige cada tarea al nodo especializado correspondiente.
"""
import json
import re
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from typing import TypedDict, Annotated, Literal, Dict, Any, List
from pydantic import BaseModel
from langgraph.graph.message import add_messages
import google.generativeai as genai
from google.colab import userdata
from Nodes.ocr_extractor import tool_ocr
from Nodes.object_localizer import tool_object_detection
from Nodes.scene_description import tool_describe_scene
from Agent.nodes import security_node, orchestrator_node, tools_node, response_node, validator_node, error_handler_node
# Configuración Gemini
genai.configure(api_key = userdata.get('GOOGLE_API_KEY'))
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

class OrchestratorState(TypedDict, total=False):
    user_input: str         
    img: bool                
    malprompt: bool         
    attempts: int           
    tools: List[str]         
    justification: str       
    tool_outputs: Dict[str, Any]  
    final_response: str      
    pending_error: bool
    validated: bool
    val_just: str
    error_history: Annotated[List[str], add_messages]
    messages: Annotated[List[BaseMessage], add_messages]
    
TOOLS = {
    "ocr": tool_ocr,
    "object_detection": tool_object_detection,
    "describe_scene": tool_describe_scene
}

# --- Routing ---
def route_from_orchestrator(state: OrchestratorState) -> str:
    if state.get("pending_error") and state.get("attempts", 0) < 2:
        return "error_handler"
    return "tools"

def route_from_tools(state: OrchestratorState) -> str:
    if state.get("pending_error") and state.get("attempts", 0) < 2:
        return "error_handler"
    return "response"

def route_from_response(state: OrchestratorState) -> str:
    if state.get("pending_error") and state.get("attempts", 0) < 2:
        return "error_handler"
    return "validator"

def route_from_validator(state: OrchestratorState) -> str:
    if state.get("pending_error") and state.get("attempts", 0) < 2:
        return "error_handler"
    return END

def route_from_error_handler(state: OrchestratorState) -> str:
    attempts = state.get("attempts", 0)
    max_attempts = 2

    if attempts < max_attempts:
        # Volver al último nodo que falló
        return "orchestrator"
    return END

# --- Grafo ---
workflow = StateGraph(OrchestratorState)

workflow.add_node("security", security_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tools", tools_node)
workflow.add_node("response", response_node)
workflow.add_node("validator", validator_node)
workflow.add_node("error_handler", error_handler_node)

workflow.set_entry_point("security")
workflow.add_edge("security", "orchestrator")

workflow.add_conditional_edges("orchestrator", route_from_orchestrator,
    {"error_handler": "error_handler", "tools": "tools"})
workflow.add_conditional_edges("tools", route_from_tools,
    {"error_handler": "error_handler", "response": "response"})
workflow.add_conditional_edges("response", route_from_response,
    {"error_handler": "error_handler", "validator": "validator"})
workflow.add_conditional_edges("validator", route_from_validator,
    {"error_handler": "error_handler", END: END})

workflow.add_conditional_edges("error_handler", route_from_error_handler,
    {"orchestrator": "orchestrator", END: END})

agent = workflow.compile()
