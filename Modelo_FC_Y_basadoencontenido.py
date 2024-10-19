import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os

# Función para cargar datos de rutas y calificaciones
def cargar_datos(ruta_csv, calificaciones_csv):
    if not os.path.exists(ruta_csv) or not os.path.exists(calificaciones_csv):
        print(f"Uno o ambos archivos no existen: {ruta_csv}, {calificaciones_csv}")
        return None, None, None, None, None, None

    try:
        datos_rutas = pd.read_csv(ruta_csv)
        datos_calificaciones = pd.read_csv(calificaciones_csv)
    except Exception as e:
        print(f"Error al cargar los archivos: {e}")
        return None, None, None, None, None, None
    
    # Verificar columnas requeridas
    required_columns_rutas = ['Título', 'Dificultad técnica', 'Mejor época', 'URL_Imagen']
    required_columns_calificaciones = ['usuario'] + datos_rutas['Título'].tolist()

    if not all(col in datos_rutas.columns for col in required_columns_rutas):
        print("Faltan columnas requeridas en el archivo de rutas.")
        return None, None, None, None, None, None

    if not all(col in datos_calificaciones.columns for col in required_columns_calificaciones):
        print("Faltan columnas requeridas en el archivo de calificaciones.")
        return None, None, None, None, None, None

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
    mejor_epoca_rutas = dict(zip(datos_rutas['Título'], datos_rutas['Mejor época'].fillna('Desconocido')))
    urls_imagenes_rutas = dict(zip(datos_rutas['Título'], datos_rutas['URL_Imagen']))
    
    matriz_calificaciones = datos_calificaciones.drop(columns=['usuario']).values

    return rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones, datos_calificaciones

# Funciones auxiliares para calcular la similitud de dificultad y época
def calcular_similitud_dificultad(dificultad, preferencia_dificultad):
    if dificultad == preferencia_dificultad:
        return 1.0
    elif abs(dificultad - preferencia_dificultad) == 1:
        return 0.5
    return 0.1

def calcular_similitud_epoca(epoca, preferencia_mes):
    if epoca == 'Desconocido' or epoca is None:
        return 0.1
    if epoca == preferencia_mes:
        return 1.0
    elif abs(epoca - preferencia_mes) <= 1:
        return 0.5
    return 0.1

# Función para recomendar rutas basadas en contenido
def recomendar_rutas_por_contenido(rutas, dificultades_rutas, mejor_epoca_rutas, preferencia_dificultad, preferencia_mes, top_n=3):
    recomendaciones_contenido = []
    for ruta in rutas:
        dificultad = dificultades_rutas[ruta]
        epoca = mejor_epoca_rutas[ruta]

        similitud_dificultad = calcular_similitud_dificultad(dificultad, preferencia_dificultad)
        similitud_epoca = calcular_similitud_epoca(epoca, preferencia_mes)

        score = (similitud_dificultad + similitud_epoca) / 2
        recomendaciones_contenido.append((ruta, score))

    recomendaciones_contenido = sorted(recomendaciones_contenido, key=lambda x: x[1], reverse=True)[:top_n]
    return recomendaciones_contenido

# Función para encontrar usuarios similares usando similitud de coseno
def encontrar_usuarios_similares(matriz_calificaciones, usuario_id, top_n=5):
    similitudes_usuarios = cosine_similarity(matriz_calificaciones)
    usuarios_similares = np.argsort(similitudes_usuarios[usuario_id])[::-1][1:top_n+1]
    return usuarios_similares

# Función para combinar recomendaciones
def combinar_recomendaciones(matriz_calificaciones, usuario_id, rutas, dificultades_rutas, mejor_epoca_rutas, preferencia_dificultad, preferencia_mes, top_n=3, peso_contenido=0.5, peso_colaborativo=0.5):
    recomendaciones_contenido = recomendar_rutas_por_contenido(rutas, dificultades_rutas, mejor_epoca_rutas, preferencia_dificultad, preferencia_mes, top_n)
    
    usuarios_similares = encontrar_usuarios_similares(matriz_calificaciones, usuario_id, top_n=5)
    recomendaciones_colaborativas = np.mean(matriz_calificaciones[usuarios_similares], axis=0)

    rutas_no_calificadas = np.where(matriz_calificaciones[usuario_id] == 0)[0]
    recomendaciones_filtradas = [(rutas[i], recomendaciones_colaborativas[i]) for i in rutas_no_calificadas]
    recomendaciones_filtradas = sorted(recomendaciones_filtradas, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Normalizar los puntajes
    recomendaciones_contenido = [(ruta, score * peso_contenido) for ruta, score in recomendaciones_contenido]
    recomendaciones_filtradas = [(ruta, score * peso_colaborativo) for ruta, score in recomendaciones_filtradas]

    recomendaciones_finales = recomendaciones_contenido + recomendaciones_filtradas
    recomendaciones_finales = sorted(recomendaciones_finales, key=lambda x: x[1], reverse=True)[:top_n]
    
    return recomendaciones_finales

# Función para pedir parámetros de entrada al usuario
def obtener_parametros_entrada():
    # Opciones de dificultad
    opciones_dificultad = {
        1: 'Fácil',
        2: 'Moderado',
        3: 'Difícil',
        4: 'Muy difícil',
        5: 'Solo expertos'
    }
    
    # Solicitar preferencia de dificultad
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

    # Solicitar preferencia de mes
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

# Función para extraer rutas relevantes de un usuario
# Las rutas relevantes son aquellas que tienen una calificación de 4 o 5
def obtener_rutas_relevantes(usuario_id, matriz_calificaciones, rutas):
    rutas_relevantes = [rutas[i] for i in range(len(rutas)) if matriz_calificaciones[usuario_id, i] >= 4]
    return rutas_relevantes

# Función para calcular Precision@k
def calcular_precision(recomendaciones, rutas_relevantes, k):
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

# Cargar datos iniciales
ruta_csv = 'datos-rutas-Procesado-6-csv-csv.csv'
calificaciones_csv = 'tu_archivo_actualizado.csv'
rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones, datos_calificaciones = cargar_datos(ruta_csv, calificaciones_csv)

# Pedir parámetros de entrada al usuario
preferencia_dificultad, preferencia_mes = obtener_parametros_entrada()

# Definir usuario_id como 0 (esto puede cambiar)
usuario_id = 0  # Usuario para el que se harán recomendaciones

# Obtener recomendaciones combinadas (limitadas a 3)
recomendaciones = combinar_recomendaciones(matriz_calificaciones, usuario_id, rutas, dificultades_rutas, mejor_epoca_rutas, preferencia_dificultad, preferencia_mes, top_n=3)

# Obtener rutas relevantes a partir de la matriz de calificaciones (calificación de 4 o 5)
rutas_relevantes = obtener_rutas_relevantes(usuario_id, matriz_calificaciones, rutas)

# Calcular y mostrar Precision@k y Recall@k
k = 3 # Cambiamos a 3 para que coincida con las recomendaciones limitadas
precision = calcular_precision(recomendaciones, rutas_relevantes, k)
recall = calcular_recall(recomendaciones, rutas_relevantes, k)

print(f"Precision@{k}: {precision:.2f}")
print(f"Recall@{k}: {recall:.2f}")

# Mostrar recomendaciones e imágenes
plt.figure(figsize=(10, 5))  # Crear una figura para todas las imágenes
for i, (ruta, score) in enumerate(recomendaciones):
    print(f"- Ruta {ruta}: Calificación ponderada {score:.2f}")
    
    # Cargar y mostrar la imagen de la ruta
    url_imagen = urls_imagenes_rutas[ruta]
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url_imagen, headers=headers)
        img = Image.open(BytesIO(response.content))
        
        plt.subplot(1, len(recomendaciones), i + 1)  # Crear subgráfica
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Ruta: {ruta}")
    except Exception as e:
        print(f"No se pudo cargar la imagen de la ruta {ruta}: {e}")

plt.tight_layout()
plt.show()
