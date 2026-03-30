import json
import pymysql
import unidecode
import paho.mqtt.client as mqtt
import time

# --- CONFIGURACIÓN ---
json_detecciones = "output/detecciones.json"
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883

# Topic único con toda la info
TOPIC_MERCADO = "mercado/datos"

# --- CONEXIÓN MYSQL ---
def conectar_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="mydb",
        charset='utf8mb4'
    )

# --- OBTENER INFO PRODUCTO ---
def obtener_info_producto(nombre_buscar):
    conexion = conectar_db()
    try:
        with conexion.cursor() as cursor:
            sql = """
                SELECT nombre, producto, unidades, precio, categoria
                FROM productos
                WHERE LOWER(nombre) LIKE LOWER(%s)
            """
            cursor.execute(sql, (f"%{nombre_buscar}%",))
            resultado = cursor.fetchone()
            if resultado:
                nombre, producto, unidades, precio, categoria = resultado
                return {
                    "producto": producto.strip(),
                    "unidades": int(unidades),
                    "precio": float(precio),
                    "categoria": categoria.strip()
                }
    except Exception as e:
        print(f"Error al buscar producto: {e}")
    finally:
        conexion.close()
    return None

# --- LEER JSON DETECCIONES ---
def leer_detecciones_json():
    try:
        with open(json_detecciones, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            if not contenido:
                return []
            data = json.loads(contenido)
            if isinstance(data, dict):
                data = list(data.values())
            return data
    except Exception as e:
        print(f"Error al leer JSON: {e}")
        return []

# --- CREAR ESTRUCTURA COMPLETA ---
def crear_datos_completos():
    detecciones = leer_detecciones_json()
    productos, precios, unidades, categorias = [], [], [], []
    total = 0.0

    for obj in detecciones:
        clase = obj.get("clase", "")
        info = obtener_info_producto(clase)
        if info:
            productos.append(info["producto"])
            precios.append(info["precio"])
            unidades.append(info["unidades"])
            categorias.append(info["categoria"])
            total += info["precio"] 

    # Estructura completa
    return {
        "productos": productos,
        "precios": precios,
        "unidades": unidades,
        "categorias": categorias,
        "total": round(total, 2)
    }

# --- ENVIAR POR MQTT ---
def enviar_mqtt(topic, datos):
    cliente = mqtt.Client()
    cliente.connect(MQTT_BROKER, MQTT_PORT, 60)
    mensaje = json.dumps(datos, ensure_ascii=False)
    cliente.publish(topic, mensaje)
    cliente.disconnect()
    print(f"📤 Enviado: {mensaje}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("Enviando datos MQTT estructurados...")
    try:
        while True:
            datos_completos = crear_datos_completos()
            enviar_mqtt(TOPIC_MERCADO, datos_completos)
            
            print(f"✅ Enviados {len(datos_completos['productos'])} productos")
            print(f"💰 Total: {datos_completos['total']}€\n")
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")