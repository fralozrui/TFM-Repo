"""
Nodo especializado en la descripción automática de escenas a partir de imágenes.
Utiliza modelos preentrenados como BLIP, GIT o similares para generar descripciones 
semánticas útiles.
"""
from Agent.orchestrator import OrchestratorState
def tool_describe_scene(state: OrchestratorState) -> str:
    print("Ejecutando descripción de escena...")
    return "La imagen muestra una calle con varios edificios."