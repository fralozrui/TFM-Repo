"""
Implementa el grafo de nodos LangGraph que define la lógica de flujo del agente multimodal. 
Contiene la estructura de decisión basada en el LLM orquestador que dirige cada tarea al nodo especializado correspondiente.
"""
import json
import re
import os
import sys
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from typing import TypedDict, Annotated, Literal, Dict, Any, List
from pydantic import BaseModel
from langgraph.graph.message import add_messages
# from google.colab import userdata
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Agent.orchestrator_keys import OrchestratorState, model_gemini
from Agent.nodes import security_node, orchestrator_node, tools_node, response_node, validator_node, error_handler_node

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

# Estado inicial
init_state = {
    "user_input": "¿Qué pone en el cartel?",
    "img": True,
    "malprompt": False,
    "attempts": 0,
    "tools": [],
    "justification": "",
    "tool_outputs": {},
    "final_response": "",
    "error_history": [],
    "messages": []
}

# Ejecutamos el grafo
result = agent.invoke(init_state)

print("\n--- RESULTADO FINAL ---")
print(result["final_response"])

print("\n--- HISTORIAL DE MENSAJES ---")
for msg in result["messages"]:
    print(f"[{msg.type.upper()}] {msg.content}")