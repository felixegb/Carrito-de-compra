# Sistema de Detección de Productos con YOLO + MQTT

Detecta productos en vídeo usando YOLO11 con tracking ByteTrack y publica los resultados por MQTT con datos enriquecidos desde MySQL.

## 📁 Estructura del Proyecto

```
mercadona/
├── prueba_video.py     # ⭐ Detección en vídeo (PRINCIPAL)
├── mqtt.py             # ⭐ Publicación MQTT de detecciones
├── sort.py             # Algoritmo de tracking SORT (Kalman)
│
├── modelo/
│   └── best.pt         # Modelo YOLO entrenado
│
├── data/
│   └── videos/
│       └── video.mp4   # Vídeo de entrada
│
├── output/
│   └── detecciones.json  # Detecciones generadas en tiempo real
│
├── config/
│   └── config.py       # Configuración centralizada (rutas, MQTT, DB)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Uso

### 1. Detección en Vídeo
```bash
python prueba_video.py
```
Lee `data/videos/video.mp4`, detecta productos con YOLO + ByteTrack y escribe `output/detecciones.json` cada 2 segundos.

### 2. Publicación MQTT
```bash
python mqtt.py
```
Lee `output/detecciones.json`, enriquece los datos con MySQL y publica cada producto en el topic `mercado/datos` cada segundo.

## 📋 Requisitos

- Python 3.11+
- CUDA 11.8+ (opcional, para GPU)
- MySQL con base de datos `mydb` y tabla `productos`
- Broker MQTT accesible (por defecto `test.mosquitto.org:1883`)

## 🔧 Instalación

```bash
# Activar entorno virtual
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
# Para GPU (reemplaza la línea de torch en requirements.txt):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ⚙️ Configuración

Edita `config/config.py` para ajustar:
- Rutas de modelo, vídeo y salida
- Parámetros de detección (confianza, IOU, frames estables)
- Conexión MySQL (`DB_CONFIG`)
- Broker MQTT (`MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC`)

## 📊 Flujo del Sistema

```
video.mp4 ──► prueba_video.py ──► detecciones.json ──► mqtt.py ──► broker MQTT
                (YOLO + ByteTrack)      (output/)         (MySQL enrichment)
```

## 📝 Notas

- Las detecciones se guardan cada 2 segundos y se limpian al terminar el vídeo
- El modelo reconoce productos específicos de Mercadona
- Soporta GPU automáticamente si CUDA está disponible
