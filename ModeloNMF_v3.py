import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import random
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Función para cargar datos de rutas y calificaciones
def cargar_datos(ruta_csv, calificaciones_csv):
    datos_rutas = pd.read_csv(ruta_csv)

    # Mapear los niveles de dificultad a números
    dificultad_mapping = {
        'Fácil': 1,
        'Moderado': 2,
        'Difícil': 3,
        'Muy difícil': 4,
        'Solo expertos': 5
    }

    rutas = datos_rutas['Título'].tolist()
    dificultades_rutas = dict(zip(datos_rutas['Título'], datos_rutas['Dificultad técnica'].map(dificultad_mapping)))
    mejor_epoca_rutas = dict(zip(datos_rutas['Título'], datos_rutas.get('Mejor época', [None]*len(rutas))))
    urls_imagenes_rutas = dict(zip(datos_rutas['Título'], datos_rutas['URL_Imagen']))

    # Cargar las calificaciones y nombres de usuarios
    datos_calificaciones = pd.read_csv(calificaciones_csv)
    nombres_usuarios = datos_calificaciones['usuario'].tolist()  # Obtener los nombres de los usuarios
    
    matriz_calificaciones = datos_calificaciones.drop(columns=['usuario']).values

    return rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones, nombres_usuarios

# Función para actualizar el modelo NMF
def actualizar_modelo_nmf(matriz_calificaciones, n_components=30, max_iter=200):
    modelo_nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=max_iter)
    W = modelo_nmf.fit_transform(matriz_calificaciones)
    H = modelo_nmf.components_
    matriz_predicciones = np.dot(W, H)
    return modelo_nmf, W, H, matriz_predicciones

# Función para calcular Precision@k
def calcular_precision_a_k(matriz_real, matriz_predicciones, k=3):
    precisiones = []
    
    for usuario in range(matriz_real.shape[0]):
        # Obtener índices de las rutas con las predicciones más altas
        rutas_predichas = np.argsort(matriz_predicciones[usuario])[::-1][:k]
        
        # Obtener rutas relevantes (con calificación >= 4 en la matriz real)
        rutas_relevantes = np.where(matriz_real[usuario] >= 4)[0]
        
        # Contar cuántas de las rutas predichas están en las rutas relevantes
        rutas_relevantes_en_top_k = len(set(rutas_predichas).intersection(rutas_relevantes))
        
        # Calcular precision@k para este usuario
        precision = rutas_relevantes_en_top_k / k
        precisiones.append(precision)
    
    return np.mean(precisiones)

# Función para calcular Recall@k
def calcular_recall_a_k(matriz_real, matriz_predicciones, k=3):
    recalls = []
    
    for usuario in range(matriz_real.shape[0]):
        # Obtener índices de las rutas con las predicciones más altas
        rutas_predichas = np.argsort(matriz_predicciones[usuario])[::-1][:k]
        
        # Obtener rutas relevantes (con calificación >= 4 en la matriz real)
        rutas_relevantes = np.where(matriz_real[usuario] >= 4)[0]
        
        # Contar cuántas de las rutas predichas están en las rutas relevantes
        rutas_relevantes_en_top_k = len(set(rutas_predichas).intersection(rutas_relevantes))
        
        # Calcular recall@k para este usuario
        if len(rutas_relevantes) > 0:
            recall = rutas_relevantes_en_top_k / len(rutas_relevantes)
            recalls.append(recall)
        else:
            # Si no hay rutas relevantes para este usuario, omitimos el cálculo
            recalls.append(0)
    
    return np.mean(recalls)

# Función para recomendar rutas para un usuario existente con imágenes (excluyendo rutas ya calificadas)
def recomendar_rutas_para_usuario_existente_nmf_con_imagenes(matriz_predicciones, usuario_id, rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, matriz_real_calificaciones, top_n=3):
    calificaciones_usuario = matriz_predicciones[usuario_id]
    
    calificaciones_ponderadas = []
    for i, ruta in enumerate(rutas):
        # Excluir las rutas ya calificadas (calificación distinta de 0 en la matriz original)
        if matriz_real_calificaciones[usuario_id, i] == 0:
            dificultad = dificultades_rutas[ruta]
            epoca = mejor_epoca_rutas[ruta]

            peso_dificultad = 1.0 if dificultad == preferencia_dificultad else 0.5 if abs(dificultad - preferencia_dificultad) == 1 else 0.1
            peso_epoca = 1.0 if epoca == preferencia_mes else 0.5 if epoca and abs(epoca - preferencia_mes) <= 1 else 0.1

            peso_total = (peso_dificultad + peso_epoca) / 2
            calificaciones_ponderadas.append((i, calificaciones_usuario[i] * peso_total))
    
    # Ordenar las rutas por calificación ponderada y seleccionar las top_n
    rutas_ordenadas = sorted(calificaciones_ponderadas, key=lambda x: x[1], reverse=True)[:top_n]
    mejores_rutas = [(rutas[ruta_index], calificacion, urls_imagenes_rutas[rutas[ruta_index]]) for ruta_index, calificacion in rutas_ordenadas]
    
    return mejores_rutas

# Cargar datos iniciales
ruta_csv = 'datos-rutas-Procesado-6-csv-csv.csv'
calificaciones_csv = 'matriz_calificaciones_1.csv'
rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones, nombres_usuarios = cargar_datos(ruta_csv, calificaciones_csv)

# División de datos en entrenamiento y prueba (20% para prueba)
train_indices, test_indices = train_test_split(range(matriz_calificaciones.shape[0]), test_size=0.2, random_state=42)
matriz_entrenamiento = matriz_calificaciones[train_indices]
matriz_prueba = matriz_calificaciones[test_indices]

# Entrenamiento del modelo con el conjunto de entrenamiento
modelo_nmf, W, H, matriz_predicciones_entrenamiento = actualizar_modelo_nmf(matriz_entrenamiento)

# Calcular Precision@k y Recall@k para el conjunto de entrenamiento
precision_k_entrenamiento = calcular_precision_a_k(matriz_entrenamiento, matriz_predicciones_entrenamiento, k=3)
recall_k_entrenamiento = calcular_recall_a_k(matriz_entrenamiento, matriz_predicciones_entrenamiento, k=3)

print(f"Precision@3 en el conjunto de entrenamiento: {precision_k_entrenamiento:.4f}")
print(f"Recall@3 en el conjunto de entrenamiento: {recall_k_entrenamiento:.4f}")

# Calcular las predicciones en el conjunto de prueba
_, _, _, matriz_predicciones_prueba = actualizar_modelo_nmf(matriz_prueba)

# Calcular Precision@k y Recall@k para el conjunto de prueba
precision_k_prueba = calcular_precision_a_k(matriz_prueba, matriz_predicciones_prueba, k=3)
recall_k_prueba = calcular_recall_a_k(matriz_prueba, matriz_predicciones_prueba, k=3)

print(f"Precision@3 en el conjunto de prueba: {precision_k_prueba:.4f}")
print(f"Recall@3 en el conjunto de prueba: {recall_k_prueba:.4f}")

# Seleccionar un usuario aleatorio del conjunto de entrenamiento
usuario_id_local = random.randint(0, len(train_indices) - 1)  # Selección aleatoria dentro del conjunto de entrenamiento
usuario_id_global = train_indices[usuario_id_local]  # Obtener el índice global (original) del usuario seleccionado
nombre_usuario = nombres_usuarios[usuario_id_global]  # Obtener el nombre del usuario a partir del índice global
print(f"Se ha seleccionado al usuario: {nombre_usuario}")

# Solicitar preferencias del usuario
preferencia_dificultad = int(input("Ingrese su preferencia de dificultad (1=Fácil, 2=Moderado, 3=Difícil, 4=Muy difícil, 5=Solo expertos): "))
preferencia_mes = int(input("Ingrese su preferencia de mes (1=Enero, 2=Febrero, ..., 12=Diciembre): "))

# Generar recomendaciones para el usuario existente (excluyendo rutas ya calificadas)
recomendaciones_usuario_existente = recomendar_rutas_para_usuario_existente_nmf_con_imagenes(
    matriz_predicciones_entrenamiento, 
    usuario_id_local,  # El índice es local, dentro de la matriz de entrenamiento
    rutas, 
    dificultades_rutas, 
    mejor_epoca_rutas, 
    urls_imagenes_rutas, 
    preferencia_dificultad, 
    preferencia_mes, 
    matriz_entrenamiento,  # matriz real de calificaciones
    top_n=3
)

print("\nRecomendaciones para el usuario existente:")
for ruta, calificacion, url_imagen in recomendaciones_usuario_existente:
    print(f"- Ruta {ruta}: Calificación ponderada {calificacion:.2f}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url_imagen, headers=headers)
        img = Image.open(BytesIO(response.content))
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Imagen de la ruta {ruta}")
        plt.show()
    except Exception as e:
        print(f"No se pudo cargar la imagen de la ruta {ruta}: {e}")
