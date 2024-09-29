import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
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

    datos_calificaciones = pd.read_csv(calificaciones_csv)

    matriz_calificaciones = datos_calificaciones.drop(columns=['usuario']).values

    return rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones

# Función para actualizar el modelo NMF
def actualizar_modelo_nmf(matriz_calificaciones, n_components=20, max_iter=10000):
    modelo_nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=max_iter)
    W = modelo_nmf.fit_transform(matriz_calificaciones)
    H = modelo_nmf.components_
    matriz_predicciones = np.dot(W, H)
    return modelo_nmf, W, H, matriz_predicciones

# Función para calcular RMSE
def calcular_rmse(matriz_prueba, matriz_predicciones):
    indices_usuarios, indices_rutas = matriz_prueba.nonzero()
    errores = []
    for usuario, ruta in zip(indices_usuarios, indices_rutas):
        error = matriz_prueba[usuario, ruta] - matriz_predicciones[usuario, ruta]
        errores.append(error)
    rmse = sqrt(mean_squared_error(matriz_prueba[matriz_prueba.nonzero()], matriz_predicciones[matriz_prueba.nonzero()]))
    return rmse

# Función para calcular la precisión a K
def calcular_precision_a_k(matriz_prueba, matriz_predicciones, k=3):
    """
    Calcula la precisión a K para las predicciones del modelo.
    :param matriz_prueba: Matriz real de calificaciones de prueba
    :param matriz_predicciones: Matriz de predicciones del modelo
    :param k: Número de elementos en el top K
    :return: Precisión a K
    """
    precisiones = []
    
    # Para cada usuario en la matriz de prueba
    for usuario in range(matriz_prueba.shape[0]):
        # Obtener índices de las rutas con calificaciones reales
        rutas_reales = np.argsort(matriz_prueba[usuario])[::-1][:k]
        
        # Obtener índices de las rutas con las predicciones más altas
        rutas_predichas = np.argsort(matriz_predicciones[usuario])[::-1][:k]
        
        # Calcular intersección entre rutas reales y predichas
        rutas_relevantes = len(set(rutas_reales) & set(rutas_predichas))
        
        # Precisión a K para este usuario
        precisiones.append(rutas_relevantes / k)
    
    # Promedio de precisión a K entre todos los usuarios
    return np.mean(precisiones)

# Función para recomendar rutas para un usuario nuevo con imágenes
def recomendar_rutas_para_usuario_nuevo_nmf_con_imagenes(matriz_predicciones, rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, top_n=3):
    calificaciones_promedio = np.mean(matriz_predicciones, axis=0)
    
    calificaciones_ponderadas = []
    for i, ruta in enumerate(rutas):
        dificultad = dificultades_rutas[ruta]
        epoca = mejor_epoca_rutas[ruta]
        
        peso_dificultad = 1.0 if dificultad == preferencia_dificultad else 0.5 if abs(dificultad - preferencia_dificultad) == 1 else 0.1
        peso_epoca = 1.0 if epoca == preferencia_mes else 0.5 if epoca and abs(epoca - preferencia_mes) <= 1 else 0.1
        
        peso_total = (peso_dificultad + peso_epoca) / 2
        calificaciones_ponderadas.append(calificaciones_promedio[i] * peso_total)
    
    rutas_ordenadas = sorted(enumerate(calificaciones_ponderadas), key=lambda x: x[1], reverse=True)
    mejores_rutas = [(rutas[ruta_index], calificacion, urls_imagenes_rutas[rutas[ruta_index]]) for ruta_index, calificacion in rutas_ordenadas[:top_n]]
    
    return mejores_rutas

# Función para reentrenar el modelo con un nuevo archivo .csv y brindar nuevas recomendaciones con imágenes
def reentrenar_con_nuevo_csv_y_recomendar_con_imagenes(nuevo_calificaciones_csv, rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, top_n=3):
    rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, nuevas_calificaciones = cargar_datos(ruta_csv, nuevo_calificaciones_csv)
    nueva_matriz_calificaciones = np.vstack([matriz_calificaciones, nuevas_calificaciones])
    
    # Reentrenar el modelo con los nuevos datos
    nuevo_modelo_nmf, nuevo_W, nuevo_H, nueva_matriz_predicciones = actualizar_modelo_nmf(nueva_matriz_calificaciones)
    
    # Recomendar nuevas rutas para el usuario nuevo con imágenes
    mejores_rutas_nuevo_usuario = recomendar_rutas_para_usuario_nuevo_nmf_con_imagenes(nueva_matriz_predicciones[-1:], rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, top_n)
    
    # Calcular RMSE después del reentrenamiento
    matriz_prueba = nueva_matriz_calificaciones[-1:]
    rmse_nuevo = calcular_rmse(matriz_prueba, nueva_matriz_predicciones[-1:])
    
    return nuevo_modelo_nmf, nuevo_W, nuevo_H, nueva_matriz_predicciones, mejores_rutas_nuevo_usuario, rmse_nuevo

# Cargar datos iniciales
ruta_csv = 'datos-rutas-Procesado-6-csv-csv.csv'
calificaciones_csv = 'calificaciones_usuarios_v3.csv'
rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, matriz_calificaciones = cargar_datos(ruta_csv, calificaciones_csv)

# División de datos en entrenamiento y prueba
train_indices, test_indices = train_test_split(range(matriz_calificaciones.shape[0]), test_size=0.2, random_state=42)
matriz_entrenamiento = matriz_calificaciones[train_indices]
matriz_prueba = matriz_calificaciones[test_indices]

# Entrenamiento inicial del modelo
modelo_nmf, W, H, matriz_predicciones = actualizar_modelo_nmf(matriz_entrenamiento)

# Evaluar el modelo usando RMSE
rmse_nmf = calcular_rmse(matriz_prueba, matriz_predicciones)
print(f"RMSE del modelo de recomendación con NMF: {rmse_nmf:.4f}")

# Evaluar el modelo usando Precisión a K antes del reentrenamiento
precision_k_inicial = calcular_precision_a_k(matriz_prueba, matriz_predicciones, k=3)
print(f"Precisión a K del modelo de recomendación antes del reentrenamiento: {precision_k_inicial:.4f}")

# Solicitar preferencias del usuario
preferencia_dificultad = int(input("Ingrese su preferencia de dificultad (1=Fácil, 2=Moderado, 3=Difícil, 4=Muy difícil, 5=Solo expertos): "))
preferencia_mes = int(input("Ingrese su preferencia de mes (1=Enero, 2=Febrero, ..., 12=Diciembre): "))

# Generar recomendaciones iniciales antes del reentrenamiento
recomendaciones_iniciales = recomendar_rutas_para_usuario_nuevo_nmf_con_imagenes(matriz_predicciones, rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, top_n=3)

print("\nRecomendaciones iniciales antes del reentrenamiento:")
for ruta, calificacion, url_imagen in recomendaciones_iniciales:
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

# Reentrenar el modelo y generar nuevas recomendaciones
nuevo_calificaciones_csv = 'nuevo_calificaciones_csv_v2.csv'
modelo_nmf_actualizado, W_actualizado, H_actualizado, matriz_predicciones_actualizada, nuevas_recomendaciones, rmse_actualizado = reentrenar_con_nuevo_csv_y_recomendar_con_imagenes(nuevo_calificaciones_csv, rutas, dificultades_rutas, mejor_epoca_rutas, urls_imagenes_rutas, preferencia_dificultad, preferencia_mes, top_n=3)

print(f"\nNuevas recomendaciones después del reentrenamiento:")
if not nuevas_recomendaciones:
    print("No se encontraron rutas recomendadas.")
else:
    for ruta, calificacion, url_imagen in nuevas_recomendaciones:
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

print(f"RMSE del modelo de recomendación con NMF después del reentrenamiento: {rmse_actualizado:.4f}")
# Evaluar el modelo usando Precisión a K después del reentrenamiento
precision_k_actualizada = calcular_precision_a_k(matriz_prueba, matriz_predicciones_actualizada, k=3)
print(f"Precisión a K del modelo de recomendación después del reentrenamiento: {precision_k_actualizada:.4f}")
