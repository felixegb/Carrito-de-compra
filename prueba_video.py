from ultralytics import YOLO
import cv2
import torch
import time
import json
import numpy as np

# ========== CONFIGURACIÓN ==========
# Rutas
MODELO_PATH = "modelo/best.pt"
VIDEO_PATH = "data/videos/video.mp4"
JSON_SALIDA = "output/detecciones.json"

# Parámetros de detección
CONFIANZA_MIN = 0.65
IOU_THRESHOLD = 0.45
MIN_FRAMES_ESTABLE = 5
SKIP_FRAMES = 2

# Parámetros de guardado y limpieza
INTERVALO_GUARDADO = 2  # segundos
INTERVALO_LIMPIEZA_GPU = 500  # frames
INTERVALO_PROGRESO = 60  # frames

# GPU / CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== CARGAR MODELO ==========
print("\nCargando modelo...")
modelo = YOLO(MODELO_PATH)
modelo.to(device)
modelo.fuse()
print(f"Modelo cargado en {device}")

# ========== ESTRUCTURAS DE DATOS ==========
objetos_unicos = {}
historial_detecciones = {}
ultimo_guardado = time.time()

# ========== ABRIR VIDEO ==========
cap = cv2.VideoCapture(VIDEO_PATH)
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {ancho}x{alto} @ {fps}fps ({total_frames} frames)")

# ========== PRE-CALENTAMIENTO DEL MODELO ==========
modelo.predict(source=torch.zeros((1, 3, 640, 640)).to(device))

# ========== FUNCIÓN PARA ELIMINAR DUPLICADOS ==========
def eliminar_duplicados_temporales(objetos):
    """
    Elimina objetos duplicados de la misma clase que se solapan temporalmente.
    Mantiene el que tiene más detecciones (veces_visto).
    """
    # Agrupar por clase
    por_clase = {}
    for obj_id, obj in objetos.items():
        clase = obj['clase']
        if clase not in por_clase:
            por_clase[clase] = []
        por_clase[clase].append((obj_id, obj))
    
    objetos_limpios = {}
    ids_eliminados = set()
    
    for clase, lista_objetos in por_clase.items():
        # Ordenar por veces_visto (de mayor a menor)
        lista_objetos.sort(key=lambda x: x[1]['veces_visto'], reverse=True)
        
        for i, (id1, obj1) in enumerate(lista_objetos):
            if id1 in ids_eliminados:
                continue
            
            # Comprobar solapamiento con objetos restantes
            for j, (id2, obj2) in enumerate(lista_objetos[i+1:], start=i+1):
                if id2 in ids_eliminados:
                    continue
                
                # Verificar si hay solapamiento temporal
                start1, end1 = obj1['primer_frame'], obj1['ultimo_frame']
                start2, end2 = obj2['primer_frame'], obj2['ultimo_frame']
                
                # Si se solapan, eliminar el de menos detecciones
                if not (end1 < start2 or end2 < start1):
                    ids_eliminados.add(id2)
                    print(f"   🔄 Fusionando ID {id2} en ID {id1} ({clase})")
            
            # Mantener este objeto
            objetos_limpios[id1] = obj1
    
    return objetos_limpios

# ========== INICIALIZACIÓN ==========
frame_count = 0
tiempo_inicio = time.time()

print("\nProcesando video con ByteTrack...")
print("-" * 50)

# ========== PROCESAMIENTO DEL VIDEO ==========
try:
    resultados = modelo.track(
        source=VIDEO_PATH,
        conf=CONFIANZA_MIN,
        iou=IOU_THRESHOLD,
        device=device,
        half=True if device == 'cuda' else False,
        imgsz=640,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        vid_stride=SKIP_FRAMES
    )
    
    for r in resultados:
        frame_count += 1
        tiempo_actual = time.time()
        
        for box in r.boxes:
            if box.id is None:
                continue
            
            obj_id = int(box.id[0])
            clase_id = int(box.cls[0])
            clase_nombre = r.names[clase_id]
            conf = float(box.conf[0])
            
            historial_detecciones[obj_id] = historial_detecciones.get(obj_id, 0) + 1
            
            if historial_detecciones[obj_id] >= MIN_FRAMES_ESTABLE:
                if obj_id not in objetos_unicos:
                    objetos_unicos[obj_id] = {
                        'id': obj_id,
                        'clase': clase_nombre,
                        'primer_frame': frame_count,
                        'ultimo_frame': frame_count,
                        'veces_visto': 1,
                        'confianza_promedio': conf
                    }
                else:
                    obj = objetos_unicos[obj_id]
                    obj['ultimo_frame'] = frame_count
                    obj['veces_visto'] += 1
                    obj['confianza_promedio'] = (
                        (obj['confianza_promedio'] * (obj['veces_visto'] - 1) + conf)
                        / obj['veces_visto']
                    )
        
        # ========== GUARDADO PERIÓDICO ==========
        if tiempo_actual - ultimo_guardado >= INTERVALO_GUARDADO:
            # Eliminar duplicados
            objetos_filtrados = eliminar_duplicados_temporales(objetos_unicos)
            
            with open(JSON_SALIDA, "w", encoding="utf-8") as f:
                json.dump(objetos_filtrados, f, indent=4, ensure_ascii=False)
            
            conteo_clases = {}
            for obj in objetos_filtrados.values():
                clase = obj['clase']
                conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
            
            for clase, cantidad in conteo_clases.items():
                print(f"   - {clase}: {cantidad}")
            
            ultimo_guardado = tiempo_actual
        
        # ========== VISUALIZACIÓN ==========
        frame_anotado = r.plot(conf=True, line_width=2, font_size=10, labels=True)
        
        tiempo_transcurrido = tiempo_actual - tiempo_inicio
        fps_real = frame_count / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
        
        cv2.putText(frame_anotado, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_anotado, f"FPS: {fps_real:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Detección", cv2.resize(frame_anotado, (960, 540)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # ========== PROGRESO ==========
        if frame_count % INTERVALO_PROGRESO == 0:
            progreso = (frame_count / total_frames) * 100
            print(f"Progreso: {progreso:.1f}% | FPS: {fps_real:.1f}")
        
        # ========== LIMPIEZA DE MEMORIA GPU ==========
        if device == 'cuda' and frame_count % INTERVALO_LIMPIEZA_GPU == 0:
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido")

finally:
    cv2.destroyAllWindows()
    if device == 'cuda':
        torch.cuda.empty_cache()

# ========== RESUMEN FINAL ==========
tiempo_total = time.time() - tiempo_inicio
fps_promedio = frame_count / tiempo_total if tiempo_total > 0 else 0

# Aplicar filtro final
objetos_finales = eliminar_duplicados_temporales(objetos_unicos)

# Mostrar conteo final por clase
conteo_final = {}
for obj in objetos_finales.values():
    clase = obj['clase']
    conteo_final[clase] = conteo_final.get(clase, 0) + 1

print("\n" + "=" * 60)
print("CONTEO FINAL POR CLASE")
print("=" * 60)
for clase, cantidad in sorted(conteo_final.items(), key=lambda x: x[1], reverse=True):
    print(f"  {clase}: {cantidad} objeto(s)")

# ========== LIMPIEZA FINAL ==========
# Limpiar el JSON dejándolo vacío
with open(JSON_SALIDA, "w", encoding="utf-8") as f:
    json.dump({}, f, indent=4, ensure_ascii=False)

print("\n✓ Resumen final mostrado")
print("✓ JSON limpiado correctamente")
print("✓ Proceso completado ✅")