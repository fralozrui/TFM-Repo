"""
Nodo especializado en el reconocimiento de texto en imágenes.
Implementa OCR utilizando modelos como PaddleOCR o Donut para extraer información textual
de documentos, etiquetas, señales, etc.
"""
from Agent.orchestrator import OrchestratorState
def tool_ocr(state: OrchestratorState) -> str:
    print("Ejecutando OCR...")
    return "Texto detectado: 'Ejemplo en cartel'"