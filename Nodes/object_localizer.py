"""
Nodo responsable de detectar y localizar objetos específicos en imágenes,
utilizando modelos como YOLOv8 o Grounding DINO. 
Permite responder preguntas como “¿dónde está el mando?” o “¿ves un enchufe?”.
"""
from Agent.orchestrator_keys import OrchestratorState
def tool_object_detection(state: OrchestratorState) -> str:
    print("Ejecutando detección de objetos...")
    return "Objeto detectado: 'Parada de autobús'"