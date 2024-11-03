import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os

# Función para cargar datos de rutas y calificaciones
def cargar_datos(ruta_csv, calificaciones_csv):
    if not os.path.exists(ruta_csv) or not os.path.exists(calificaciones_csv):
        print(f"Uno o ambos archivos no existen: {ruta_csv}, {calificaciones_csv}")
        return None

    try:
        datos_rutas = pd.read_csv(ruta_csv)
        datos_calificaciones = pd.read_csv(calificaciones_csv)
    except Exception as e:
        print(f"Error al cargar los archivos: {e}")
        return None
    
    # Validación dinámica de columnas
    required_columns_rutas = {'Título', 'Dificultad técnica', 'Mejor época', 'URL_Imagen'}
    if not required_columns_rutas.issubset(datos_rutas.columns):
        print("Faltan columnas requeridas en el archivo de rutas.")
        return None

    # Generación dinámica de columnas para calificaciones basadas en títulos
    required_columns_calificaciones = ['usuario'] + datos_rutas['Título'].tolist()
    if not all(col in datos_calificaciones.columns for col in required_columns_calificaciones):
        print("Faltan columnas requeridas en el archivo de calificaciones.")
        return None

    # Mapeo de dificultad
    dificultad_mapping = {
        'Fácil': 1,
        'Moderado': 2,
        'Difícil': 3,
        'Muy difícil': 4,
        'Solo expertos': 5
    }

    rutas = datos_rutas['Título'].tolist()
    dificultades_rutas = dict(zip(datos_rutas['Título'], datos_rutas['Dificultad técnica'].map(dificultad_mapping)))
    mejor_epoca_rutas = dict(zip(datos_rutas['Título'], datos_rutas['Mejor época'].astype(int)))
    urls_imagenes_rutas = dict(zip(datos_rutas['Título'], datos_rutas['URL_Imagen']))
    matriz_calificaciones = datos_calificaciones.drop(columns=['usuario']).values
    
    return rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones

# Funciones auxiliares para calcular la similitud de dificultad y época
def calcular_similitud_dificultad(dificultad, preferencia_dificultad):
    diferencia = abs(dificultad - preferencia_dificultad)
    return max(0.1, 1 - 0.25 * diferencia)

def calcular_similitud_epoca(epoca, preferencia_mes):
    diferencia = abs(epoca - preferencia_mes)
    return max(0.1, 1 - 0.1 * diferencia)

# Función para recomendar rutas con enfoque híbrido
def recomendar_rutas_por_contenido(rutas, dificultades_rutas, mejor_epoca_rutas, calificaciones_rutas, preferencia_dificultad, preferencia_mes, top_n=3):
    recomendaciones_contenido = []
    max_calificacion = max(calificaciones_rutas.values())  # Normalizar las calificaciones

    for ruta in rutas:
        dificultad = dificultades_rutas[ruta]
        epoca = mejor_epoca_rutas[ruta]
        calificacion_promedio = calificaciones_rutas.get(ruta, 0)  # Calificación promedio de la ruta

        # Filtrar rutas con tolerancia en dificultad y época
        if abs(dificultad - preferencia_dificultad) <= 1 and abs(epoca - preferencia_mes) <= 1:
            similitud_dificultad = calcular_similitud_dificultad(dificultad, preferencia_dificultad)
            similitud_epoca = calcular_similitud_epoca(epoca, preferencia_mes)

            # Calcular el score ponderado con calificación promedio
            score = 0.3 * similitud_dificultad + 0.3 * similitud_epoca + 0.4 * (calificacion_promedio / max_calificacion)
            recomendaciones_contenido.append((ruta, score))

    # Ordenar y seleccionar las mejores recomendaciones
    recomendaciones_contenido = sorted(recomendaciones_contenido, key=lambda x: x[1], reverse=True)[:top_n]
    return recomendaciones_contenido

# Función para pedir parámetros de entrada al usuario
def obtener_parametros_entrada():
    opciones_dificultad = {
        1: 'Fácil',
        2: 'Moderado',
        3: 'Difícil',
        4: 'Muy difícil',
        5: 'Solo expertos'
    }
    
    print("Seleccione la preferencia de dificultad:")
    for k, v in opciones_dificultad.items():
        print(f"{k}. {v}")
    
    while True:
        try:
            preferencia_dificultad = int(input("Ingrese un número entre 1 y 5: "))
            if preferencia_dificultad in opciones_dificultad:
                break
            else:
                print("Entrada no válida. Intente de nuevo.")
        except ValueError:
            print("Entrada no válida. Intente de nuevo.")

    print("\nSeleccione el mes de preferencia (1 para Enero, 12 para Diciembre):")
    while True:
        try:
            preferencia_mes = int(input("Ingrese un número entre 1 y 12: "))
            if 1 <= preferencia_mes <= 12:
                break
            else:
                print("Entrada no válida. Intente de nuevo.")
        except ValueError:
            print("Entrada no válida. Intente de nuevo.")

    return preferencia_dificultad, preferencia_mes

# Función para encontrar rutas relevantes (calificación promedio >= 3 y sin calificaciones de 0)
def obtener_rutas_relevantes(matriz_calificaciones, rutas):
    rutas_relevantes = [
        rutas[i] for i in range(len(rutas))
        if np.mean([cal for cal in matriz_calificaciones[:, i] if cal > 0]) >= 3
    ]
    return rutas_relevantes

# Función para calcular Precision@k
def calcular_precision(recomendaciones, rutas_relevantes, k):
    k = min(k, len(recomendaciones))
    rutas_recomendadas = [r[0] for r in recomendaciones[:k]]
    rutas_relevantes_en_recomendadas = len(set(rutas_recomendadas).intersection(rutas_relevantes))
    precision = rutas_relevantes_en_recomendadas / k
    return precision

# Función para calcular Recall@k
def calcular_recall(recomendaciones, rutas_relevantes, k):
    rutas_recomendadas = [r[0] for r in recomendaciones[:k]]
    rutas_relevantes_en_recomendadas = len(set(rutas_recomendadas).intersection(rutas_relevantes))
    recall = rutas_relevantes_en_recomendadas / len(rutas_relevantes) if rutas_relevantes else 0
    return recall

# Función para combinar recomendaciones basada solo en contenido para usuarios nuevos
def combinar_recomendaciones(rutas, dificultades_rutas, mejor_epoca_rutas, calificaciones_rutas, preferencia_dificultad, preferencia_mes, top_n=3):
    recomendaciones_contenido = recomendar_rutas_por_contenido(rutas, dificultades_rutas, mejor_epoca_rutas, calificaciones_rutas, preferencia_dificultad, preferencia_mes, top_n)
    return recomendaciones_contenido

# Cargar datos iniciales
ruta_csv = 'datos-rutas-Procesado-6-csv-csv.csv'
calificaciones_csv = 'matriz_calificaciones_1.csv'
rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones = cargar_datos(ruta_csv, calificaciones_csv)

# Generar el diccionario calificaciones_rutas con promedios de calificación
calificaciones_rutas = {rutas[i]: np.mean([cal for cal in matriz_calificaciones[:, i] if cal > 0]) for i in range(len(rutas))}

# Pedir parámetros de entrada al usuario
preferencia_dificultad, preferencia_mes = obtener_parametros_entrada()

# Obtener rutas relevantes a partir del promedio de calificación (>= 3 y sin calificaciones de 0)
rutas_relevantes = obtener_rutas_relevantes(matriz_calificaciones, rutas)
print(f"Cantidad de rutas relevantes: {len(rutas_relevantes)}")

# Obtener recomendaciones basadas en contenido
recomendaciones = combinar_recomendaciones(rutas, dificultades_rutas, mejor_epoca_rutas, 
                                           calificaciones_rutas, preferencia_dificultad, preferencia_mes, top_n=3)
print("Rutas recomendadas:", [r[0] for r in recomendaciones])

# Calcular y mostrar Precision@k y Recall@k
k = 3
precision = calcular_precision(recomendaciones, rutas_relevantes, k)
recall = calcular_recall(recomendaciones, rutas_relevantes, k)

print(f"Precision@{k}: {precision:.2f}")
print(f"Recall@{k}: {recall:.2f}")

# Mostrar recomendaciones e imágenes
if recomendaciones:
    plt.figure(figsize=(10, 5))
    for i, (ruta, score) in enumerate(recomendaciones):
        print(f"- Ruta {ruta}: Calificación ponderada {score:.2f}")
        
        url_imagen = urls_imagenes_rutas[ruta]
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url_imagen, headers=headers)
            img = Image.open(BytesIO(response.content))
            
            plt.subplot(1, len(recomendaciones), i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Ruta: {ruta}")
        except Exception as e:
            print(f"No se pudo cargar la imagen de la ruta {ruta}: {e}")

    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron rutas que cumplan con la dificultad y mes preferidos.")
