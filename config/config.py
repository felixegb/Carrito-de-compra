"""
Configuración centralizada del proyecto
"""
import os

# ========== RUTAS ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_DIR = os.path.join(BASE_DIR, "modelo")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Rutas de modelo
MODELO_PATH = os.path.join(MODELO_DIR, "best.pt")

# Rutas de datos
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
IMAGEN_DIR = os.path.join(DATA_DIR, "imagenes")
VIDEO_PATH = os.path.join(VIDEO_DIR, "video.mp4")

# Rutas de salida
JSON_SALIDA = os.path.join(OUTPUT_DIR, "detecciones.json")
RESULTADO_IMG = os.path.join(OUTPUT_DIR, "resultado.jpg")

# ========== DETECCIÓN ==========
CONFIANZA_MIN = 0.65
IOU_THRESHOLD = 0.45
MIN_FRAMES_ESTABLE = 5
SKIP_FRAMES = 2
IMGSZ = 640

# ========== INTERVALOS ==========
INTERVALO_GUARDADO = 2  # segundos
INTERVALO_LIMPIEZA_GPU = 500  # frames
INTERVALO_PROGRESO = 60  # frames

# ========== MQTT ==========
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "mercado/detecciones"

# ========== BASE DE DATOS ==========
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "mydb",
    "charset": "utf8mb4"
}

# ========== VISUALIZACIÓN ==========
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
FONT_SCALE = 0.6
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
