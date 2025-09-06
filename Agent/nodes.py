from Agent.orchestrator_keys import OrchestratorState, model_gemini
from Nodes.utils_models import TOOLS
import json
from Agent.prompt_templates import (
                                orchestrator_prompt,
                                responder_prompt,
                                security_prompt,
                                validator_prompt,
                                fallback_prompt
                                )
from Agent.safe_parse import safe_orchestrator_parse, safe_validator_parse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END

# --- Security Node ---
def security_node(state: OrchestratorState) -> OrchestratorState:
    print('-) Executing security node')
    safety_prompt = security_prompt(state) 
    response = model_gemini.generate_content(safety_prompt)
    verdict = response.text.strip().lower()

    if "inseguro" in verdict:
        state["malprompt"] = True
        return state

    return state

# --- Orchestrator ---
def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    print('-) Executing orchestrator node')
    user_input = state["user_input"]

    prompt = orchestrator_prompt(state)
    response = model_gemini.generate_content(prompt)

    text = response.candidates[0].content.parts[0].text.strip()
    try:
        parsed = safe_orchestrator_parse(text, True)
        tools = parsed.get("tools", [])

        if state.get("img") or state.get("img_base64"):
            if any(w in user_input.lower() for w in ["descripción", "qué hay", "escena", "foto"]):
                if "imagen" not in tools:
                    tools.append("imagen")
            elif any(w in user_input.lower() for w in ["objetos", "objetos en la foto"]):
                if "object_detection" not in tools:
                    tools.append("object_detection")
            elif any(w in user_input.lower() for w in ["texto", "cartel", "letrero"]):
                if "ocr" not in tools:
                    tools.append("ocr")

        state['tools'] = tools
        state['justification'] = parsed.get("justification", "")
        state['messages'] = state.get("messages", []) + [
            AIMessage(content=f"[Orchestrator] Tools: {state['tools']}, Justification: {state['justification']}")
        ]

    except Exception as e:
        state['tools'] = []
        state['justification'] = f"Error parsing JSON {str(e)}"
        state['last_failed_node'] = 'orchestrator'
        state['pending_error'] = True
        state['error_history'].append(f"Error parsing JSON: {str(e)}")
        state['messages'] = state.get("messages", []) + [HumanMessage(content=user_input), AIMessage(content=f'[Orchestrator] {state["justification"]}')]
    
    print("User input:", state["user_input"])
    print("Tools selected:", state['tools'])
    print("User input:", user_input)
    print("img:", state.get("img"))
    print("img_base64:", "sí" if state.get("img_base64") else "no")
    print("Tools seleccionadas:", tools)
    return state

# --- Tools Executor ---
def tools_node(state: OrchestratorState) -> OrchestratorState:
    print('-) Executing tools node')
    print("State recibido en tools_node:", state)
    try:
      outputs = {}
      for tool in state.get("tools", []):
          if tool in TOOLS:
                result = TOOLS[tool](state)
                print(f"[DEBUG] Resultado de la herramienta {tool}: {result}")
                if isinstance(result, list):
                    result = " ".join(result)
                print(f"[TOOL] {tool} ejecutada -> {result}")
                outputs[tool] = result
      state['tools_output'] = outputs
      state['messages'] = state.get("messages", []) + [AIMessage(content=f"[Tools] Herramientas a ejecutar: {json.dumps(outputs, ensure_ascii=False)}")]
    except Exception as e:
        print(f"[ERROR] Error al ejecutar las herramientas: {str(e)}")
        state['tools_output'] = {}
        state['last_failed_node'] = 'tools'
        state['pending_error'] = True
        state['error_history'].append(f"Error al ejecutar las herramientas: {str(e)}")
        state['messages'] = state.get("messages", []) + [AIMessage(content=f"[Tools] Error al ejecutar las herramientas: {str(e)}")]
    return state

# --- Response Generator ---
def response_node(state: OrchestratorState) -> OrchestratorState:
    print('-) Executing response node')
    try:
      prompt = responder_prompt(state)     
      response = model_gemini.generate_content(prompt)
      text = response.candidates[0].content.parts[0].text.strip()
      state['final_response'] = text
      state['messages'] = state.get("messages", []) + [AIMessage(content=text)]
    except Exception as e:
        state['final_response'] = f"Error al generar la respuesta: {str(e)}"
        state['last_failed_node'] = 'response'
        state['error_history'].append(f"Error al generar la respuesta: {str(e)}")
        state['messages'] = state.get("messages", []) + [AIMessage(content=state["final_response"])]
    return state


# --- Validator ---
def validator_node(state: OrchestratorState) -> OrchestratorState:    
    print('-) Executing validator node')
    try:  
        validation_prompt = validator_prompt(state)
        response = model_gemini.generate_content(validation_prompt)
        verdict = response.text.strip()
        try:
            verdict_json = safe_validator_parse(verdict)
        except:
            verdict_json = {"validated": False, "val_just": "Error al parsear la validación"}

        state["validated"] = verdict_json.get("validated", False)
        state["val_just"] = verdict_json.get("val_just", "Sin justificación proporcionada")
        if not state["validated"]:
            state["last_failed_node"] = "orchestrator"
            state['pending_error'] = True
            state["error_history"] = state.get("error_history", []) + [state["val_just"]]
            state["messages"] = state.get("messages", []) + [AIMessage(content=f"[Validator] {state['val_just']}")]

        else: 
          state['validated'] = True
    except Exception as e:
        state['validated'] = False
        state['last_failed_node'] = 'validator'
        state['error_history'].append(f"Error al validar la respuesta: {str(e)}")
        state['messages'] = state.get("messages", []) + [AIMessage(content=f"[Validator] Error al validar la respuesta: {str(e)}")]
    return state

# --- Error Handler ---
def error_handler_node(state: OrchestratorState) -> OrchestratorState:
    print('-) Executing error handler node')
    max_attempts = 2
    attempts = state.get("attempts", 0)

    if state.get("pending_error"):
        if attempts < max_attempts:
            # Reintento → solo incrementa contador, no ejecuta nada aquí
            state["attempts"] = attempts + 1
            state["pending_error"] = False
            print(f"[Error Handler] Reintentando generar respuesta (intento {attempts+1})")
        else:
            # Se agotaron los intentos → fallback final
            fallback_prompt_str = fallback_prompt(state)
            try:
                response = model_gemini.generate_content(fallback_prompt_str)
                text = response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                text = "Lo siento, no he podido generar una respuesta. ¿Podrías reformular tu consulta?"

            state["final_response"] = text
            state["messages"] = state.get("messages", []) + [AIMessage(content=text)]
            state["pending_error"] = False
    return state
