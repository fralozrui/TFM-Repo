import json
from Agent.orchestrator_keys import model_gemini
from typing import Any, Dict
def safe_orchestrator_parse(text: str, retry: bool = True) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + "}"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Intento de corrección trivial
        if cleaned.endswith(","):
            cleaned = cleaned[:-1] + "}"
        try:
            return json.loads(cleaned)
        except Exception:
            if retry:
                # reintento con prompt de corrección
                fix_prompt = f"""Corrige el siguiente texto para que sea un JSON válido:
                ---
                {cleaned}
                ---
                Devuelve solo el JSON válido, sin explicaciones."""
                response = model_gemini.generate_content(fix_prompt)
                return safe_orchestrator_parse(response.text, retry=False)
            else:
                return {"tools": [], "justification": "Error parsing JSON"}
            
def safe_validator_parse(text: str, retry: bool = True) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + "}"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Intento de corrección trivial
        if cleaned.endswith(","):
            cleaned = cleaned[:-1] + "}"
        try:
            return json.loads(cleaned)
        except Exception:
            if retry:
                # reintento con prompt de corrección
                fix_prompt = f"""Corrige el siguiente texto para que sea un JSON válido:
                ---
                {cleaned}
                ---
                Devuelve solo el JSON válido, sin explicaciones."""
                response = model_gemini.generate_content(fix_prompt)
                return safe_validator_parse(response.text, retry=False)
            else:
                return {"validated": [], "val_just": "Error parsing validation JSON"}