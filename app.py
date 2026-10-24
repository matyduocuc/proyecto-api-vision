"""
API de reconocimiento de rostros del curso (EP2 - Deep Learning AVY1103).
Adaptación del template proyecto-api-vision: CNN Keras 128x128x3 con FastAPI.
Si la confianza (prob. máxima del softmax) es menor al umbral, o el modelo
predice la clase "desconocido", responde que la persona NO está en la base.

Ejecutar:  uvicorn app:app --host 0.0.0.0 --port 8000
Probar:    http://127.0.0.1:8000/docs
"""
import io
import json
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

RUTA_MODELO = os.getenv("RUTA_MODELO", "modelo_cnn_rostros_curso.keras")
RUTA_CLASES = os.getenv("RUTA_CLASES", "clases_cnn_rostros.json")
TAM = 128
UMBRAL = float(os.getenv("UMBRAL", "0.60"))

app = FastAPI(title="API Reconocimiento de Rostros - EP2", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

modelo = None
clases = []


def _cargar_clases(ruta):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    if isinstance(datos, dict):
        try:
            return [datos[str(i)] for i in range(len(datos))]
        except KeyError:
            return [k for k, _ in sorted(datos.items(), key=lambda kv: kv[1])]
    return list(datos)


@app.on_event("startup")
def cargar_recursos():
    global modelo, clases
    import tensorflow as tf
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    clases = _cargar_clases(RUTA_CLASES)
    print(f"Modelo cargado. {len(clases)} clases. Umbral = {UMBRAL}")


def preprocesar(contenido: bytes) -> np.ndarray:
    """Igual que en el entrenamiento: RGB, 128x128 (LANCZOS), /255."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(contenido))).convert("RGB")
    img = img.resize((TAM, TAM), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/")
def raiz():
    return {"mensaje": "API de reconocimiento de rostros activa",
            "clases": len(clases), "umbral": UMBRAL}


@app.post("/predecir/")
async def predecir(archivo: UploadFile = File(...)):
    if modelo is None:
        raise HTTPException(503, "El modelo aún no está cargado")
    try:
        x = preprocesar(await archivo.read())
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida")

    probs = modelo.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    confianza = float(probs[idx])
    top3 = [{"nombre": clases[i].replace("_", " ").title(),
             "probabilidad": round(float(probs[i]), 4)}
            for i in np.argsort(probs)[::-1][:3]]

    # La clase "desconocido" (caras que no son del curso) también cuenta como "no está"
    es_desconocido = clases[idx].lower().startswith("desconocido")

    if confianza < UMBRAL or es_desconocido:
        return {"nombre_archivo": archivo.filename, "encontrado": False,
                "clase_predicha": None, "confianza": round(confianza, 4),
                "mensaje": "La persona no está en la base de datos",
                "top3": top3}

    nombre = clases[idx].replace("_", " ").title()
    return {"nombre_archivo": archivo.filename, "encontrado": True,
            "clase_predicha": nombre, "confianza": round(confianza, 4),
            "mensaje": f"Persona identificada: {nombre}", "top3": top3}




