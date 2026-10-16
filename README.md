@'
# API de reconocimiento de rostros del curso (EP2 - Deep Learning AVY1103)

API hecha con FastAPI que recibe una foto de 128x128 pixeles (RGB) y responde con el nombre de la persona del curso, o con "La persona no está en la base de datos".

El modelo es una CNN (arquitectura `A_basica`) entrenada con 15 personas del curso mas una clase `desconocido` (rostros de LFW), que sirve para rechazar a quien no esta en la base.

## Contenido del repositorio

| Archivo | Para que sirve |
|---|---|
| `app.py` | API (FastAPI): carga el modelo y expone `/predecir/` |
| `modelo_cnn_rostros_curso.keras` | Modelo CNN entrenado (16 clases) |
| `clases_cnn_rostros.json` | Lista de clases, en el orden de salida del modelo |
| `requirements.txt` | Librerias necesarias |
| `notebook/` | Notebook de Colab con el entrenamiento, la comparacion A/B/C y la calibracion del umbral |

## Como ejecutarla (Windows, PowerShell)

1. Clonar el repositorio y entrar a la carpeta:

```powershell
git clone https://github.com/matyduocuc/proyecto-api-vision.git
cd proyecto-api-vision
git checkout matyduocuc
```

2. Crear el entorno virtual e instalar las librerias (Python 3.10 a 3.12):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Levantar la API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Debe aparecer `Modelo cargado. 16 clases. Umbral = 0.7`.

4. Probarla en el navegador, abriendo `http://127.0.0.1:8000/docs`, y usando `POST /predecir/` con una foto en el campo `archivo`.

Tambien se puede probar con curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predecir/" -F "archivo=@foto.jpg"
```

## Endpoints

- `GET /` : estado de la API (numero de clases y umbral).
- `POST /predecir/` : recibe una imagen (campo `archivo`) y responde con:

```json
{
  "nombre_archivo": "foto.jpg",
  "encontrado": true,
  "clase_predicha": "Nombre Apellido",
  "confianza": 0.93,
  "mensaje": "...",
  "top3": []
}
```

Si la persona no esta en la base: `"encontrado": false`, `"clase_predicha": null` y `"mensaje": "La persona no está en la base de datos"`.

## Umbral de confianza

Una prediccion se rechaza si su confianza es menor al umbral, o si el modelo predice la clase `desconocido`. El valor por defecto es **0.70**. Se puede cambiar sin tocar el codigo, antes de levantar la API:

```powershell
$env:UMBRAL = "0.60"
uvicorn app:app --host 0.0.0.0 --port 8000
```

En el set de prueba, subir el umbral rechaza mas desconocidos pero tambien deja fuera a mas alumnos reales (0.50: 80 % de alumnos aceptados; 0.70: 63 %).

## Limitaciones

- Hay pocas fotos por persona, por lo que algunas personas se reconocen peor que otras.
- Fotos muy distintas a las de entrenamiento (angulo, luz, fondo) pueden bajar la confianza.
- La clase `desconocido` esta hecha con retratos de LFW, asi que algunos rostros ajenos al curso aun podrian aceptarse.

## Autor

Nicolas Cuevas - DuocUC, Deep Learning (AVY1103).
'@ | Set-Content README.md -Encoding UTF8
Get-Content README.md -TotalCount 5
