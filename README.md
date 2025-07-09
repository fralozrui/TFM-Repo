# 🧿 Asistente Multimodal para Personas con Discapacidad Visual

> Proyecto de fin de máster en Inteligencia Artificial, Data Science y Big Data
> Aplicación accesible e inteligente capaz de asistir a personas con discapacidad visual mediante entrada por voz e imagen.

---

## 🌍 Objetivo del proyecto

El objetivo principal es **desarrollar un asistente multimodal accesible** que pueda ayudar a personas con discapacidad visual a **comprender y realizar tareas del entorno físico**, como:

- 📸 Describir escenas visibles desde una imagen.
- 🔍 Localizar objetos específicos.
- 📃 Leer textos en documentos o pantallas (OCR).
- 🗣️ Conversar por voz y obtener respuestas también por voz.

Este asistente toma **una imagen (o vídeo) y un mensaje de audio** del usuario, detecta la intención y activa el modelo especializado que mejor puede resolverla.

---

## 🧠 ¿Cómo funciona?

El sistema está basado en una arquitectura de agente **LangGraph**, donde:

- Un **LLM orquestador** decide **qué modelo usar** según la consulta del usuario.
- Cada **nodo especializado** resuelve una tarea concreta de visión o comprensión.
- Se gestionan las entradas/salidas mediante una interfaz web accesible con Gradio.


[Entrada Usuario] --> (Audio + Imagen)
|
v
[Transcripción del audio (STT)]
|
v
[LLM Orquestador (LangGraph)]
|
v
[Nodo especializado en cierta tarea]
|
v
[Respuesta en texto] 
|
v
[Conversión a audio (TTS)] 
|
v
[Salida al usuario]

## 📁 Estructura del Repositorio
├── app/
│ ├── main.py # Inicialización del agente e interfaz Gradio
│ ├── router.py # Endpoints API opcionales
│ ├── config.py # Configuración general del sistema
│ └── utils.py # Funciones auxiliares
│
├── agent/
│ ├── langgraph_orchestrator.py # Orquestador con LangGraph
│ ├── prompt_templates.py # Prompts del LLM orquestador
│ └── memory_manager.py # Gestión opcional de memoria conversacional
│
├── nodes/
│ ├── scene_description.py # Nodo de descripción de escenas
│ ├── ocr_extractor.py # Nodo de reconocimiento de texto
│ ├── object_localizer.py # Nodo de detección de objetos
│ └── common_models.py # Carga compartida de modelos
│
├── preprocessing/
│ ├── speech_to_text.py # Conversión de audio a texto
│ ├── image_utils.py # Preprocesado de imágenes
│ └── text_to_speech.py # Conversión de texto a voz
│
├── frontend/
│ └── gradio_interface.py # Interfaz visual de usuario
│
├── models/ # (Opcional) Modelos fine-tuned específicos
│
├── tests/
│ ├── test_nodes.py # Unit tests de nodos
│ ├── test_agent.py # Test del orquestador
│ └── test_integration.py # Tests end-to-end del pipeline
│
├── Dockerfile # Entorno de ejecución portátil
└── README.md # Documentación general

## 🚀 Cómo ejecutar el proyecto
