"""
Define funciones comunes para cargar y ejecutar modelos compartidos entre nodos,
como abstracciones de inferencia, carga desde HuggingFace, cacheo y gestión de dispositivos (CPU/GPU).
"""
from Nodes.ocr_extractor import tool_ocr
from Nodes.object_localizer import tool_object_detection
from Nodes.scene_description import tool_describe_scene
TOOLS = {
    "ocr": tool_ocr,
    "object_detection": tool_object_detection,
    "describe_scene": tool_describe_scene
}