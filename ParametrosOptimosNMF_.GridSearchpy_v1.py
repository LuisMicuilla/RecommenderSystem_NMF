import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
import pandas as pd

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
# Función de evaluación para Precision@k
def precision_at_k_scorer(matriz_real, matriz_pred, k=3):
    return calcular_precision_a_k(matriz_real, matriz_pred, k=k)

# Búsqueda personalizada de hiperparámetros
def custom_grid_search_nmf(X, param_grid, cv=5, k=3):
    best_score = -np.inf
    best_params = None
    
    # Dividir los datos en k particiones para validación cruzada
    fold_size = len(X) // cv
    for params in ParameterGrid(param_grid):
        scores = []
        
        # Validación cruzada manual
        for fold in range(cv):
            # Crear conjunto de entrenamiento y prueba para este pliegue
            start, end = fold * fold_size, (fold + 1) * fold_size
            X_train = np.vstack([X[:start], X[end:]])
            X_test = X[start:end]

            # Ajustar el modelo con los parámetros actuales
            nmf = NMF(n_components=params['n_components'], max_iter=params['max_iter'], random_state=42, init='random')
            W_train = nmf.fit_transform(X_train)
            H = nmf.components_

            # Realizar predicciones en el conjunto de prueba
            W_test = nmf.transform(X_test)
            X_pred = np.dot(W_test, H)

            # Calcular precisión@k en el conjunto de prueba
            score = precision_at_k_scorer(X_test, X_pred, k=k)
            scores.append(score)

        # Promediar el puntaje para este conjunto de parámetros
        mean_score = np.mean(scores)
        print(f"Parámetros: {params}, Precision@{k}: {mean_score}")

        # Actualizar el mejor puntaje y parámetros
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score

# Cargar los datos de calificaciones
ruta_csv = 'datos-rutas-Procesado-6-csv-csv.csv'
calificaciones_csv = 'matriz_calificaciones_1.csv'
_, _, _, _, matriz_calificaciones, _ = cargar_datos(ruta_csv, calificaciones_csv)

# Definir el grid de parámetros
param_grid = {
    'n_components': [10, 15, 20, 25, 30],
    'max_iter': [200, 500, 1000]
}

# Ejecutar la búsqueda personalizada de hiperparámetros
mejores_parametros, mejor_score = custom_grid_search_nmf(matriz_calificaciones, param_grid, cv=5, k=3)
print("Mejores parámetros:", mejores_parametros)
print("Mejor puntaje de precisión@k:", mejor_score)