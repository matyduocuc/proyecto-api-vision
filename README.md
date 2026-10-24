# API de reconocimiento de rostros del curso (EP2 - Deep Learning AVY1103)

API hecha con FastAPI que recibe una foto de 128x128 pixeles (RGB) y responde con el nombre de la persona del curso, o con "La persona no está en la base de datos".

La app movil que la consume esta en https://github.com/matyduocuc/app-rostros

## Modelo

- CNN de arquitectura `A_basica` (3 bloques Conv2D + MaxPooling, 4.29 M de parametros), ganadora entre 3 arquitecturas comparadas en el notebook.
- Entrenada con **19 personas del curso (200 fotos)** mas la clase `desconocido` (45 rostros del dataset publico LFW): **20 clases**.
- Aumento de datos durante el entrenamiento: volteo horizontal, rotacion (hasta ±18°), zoom, desplazamiento, brillo y contraste.
- Division 60/20/20 (train/val/test). Test accuracy: **0.776** sobre 49 imagenes.
- Las fotos vienen del repositorio DataOps del curso (DVC).

## Contenido del repositorio

| Archivo | Para que sirve |
|---|---|
| `app.py` | API (FastAPI): carga el modelo y expone `/predecir/` |
| `modelo_cnn_rostros_curso.keras` | Modelo CNN entrenado (20 clases) |
| `clases_cnn_rostros.json` | Lista de clases, en el orden de salida del modelo |
| `requirements.txt` | Librerias necesarias |
| `notebook/` | Notebook de Colab: entrenamiento, comparacion A/B/C, evaluacion y calibracion del umbral |

## Como ejecutarla (Windows, PowerShell)

1. Clonar y entrar a la carpeta:

```powershell
git clone https://github.com/matyduocuc/proyecto-api-vision.git
cd proyecto-api-vision
git checkout matyduocuc
```

2. Crear el entorno virtual e instalar librerias (Python 3.10 a 3.12):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Levantar la API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Debe aparecer `Modelo cargado. 20 clases. Umbral = 0.5`.

4. Probar en el navegador: `http://127.0.0.1:8000/docs` y usar `POST /predecir/` con una foto en el campo `archivo`. Tambien con curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predecir/" -F "archivo=@foto.jpg"
```

Para que la app del celular llegue a la API, el PC y el celular deben estar en la misma red (WiFi o hotspot del celular) y el firewall de Windows debe permitir el puerto 8000:

```powershell
New-NetFirewallRule -DisplayName "API Rostros 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## Endpoints

- `GET /` : estado de la API (`clases` y `umbral`). La app lo usa para comprobar la conexion.
- `POST /predecir/` : recibe una imagen (campo `archivo`) y responde:

```json
{
  "nombre_archivo": "foto.jpg",
  "encontrado": true,
  "clase_predicha": "Matias Gaete",
  "confianza": 0.82,
  "mensaje": "Persona identificada: Matias Gaete",
  "top3": [
    {"nombre": "Matias Gaete", "probabilidad": 0.82},
    {"nombre": "Benjamin Palma", "probabilidad": 0.054},
    {"nombre": "Desconocido", "probabilidad": 0.049}
  ]
}
```

Si la persona no esta en la base: `"encontrado": false`, `"clase_predicha": null` y `"mensaje": "La persona no está en la base de datos"`.

Errores: `400` si el archivo no es una imagen valida y `503` si el modelo aun no esta cargado.

## Umbral de confianza

Una prediccion se rechaza si su confianza es menor al umbral, o si el modelo predice la clase `desconocido`. El valor por defecto es **0.50** y se puede cambiar sin tocar el codigo:

```powershell
$env:UMBRAL = "0.60"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Calibracion sobre el conjunto de test (alumnos reconocidos correctamente segun el umbral):

| Umbral | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|
| Alumnos aceptados | 65 % | 60 % | 55 % | 42 % | 32 % |

Subir el umbral rechaza mas desconocidos, pero tambien deja fuera a mas alumnos reales.

## Limitaciones

- Hay pocas fotos por persona (unas 10), por lo que la confianza baja con angulos, distancias o luz distintos a los de entrenamiento.
- La clase `desconocido` se construyo con retratos de LFW, asi que algunos rostros ajenos al curso aun podrian aceptarse.

## Autor

Nicolas Cuevas - DuocUC, Deep Learning (AVY1103).