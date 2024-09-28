import pandas as pd

# Leer los datos desde el archivo CSV original
df = pd.read_csv('datos_rutas_desde_csv_v2.csv')

# Obtener los términos únicos para las nuevas columnas
terminos_unicos = df['Término'].unique()

# Crear un nuevo DataFrame con las columnas requeridas
columnas = ['Título', 'URL'] + list(terminos_unicos)
df_transformado = pd.DataFrame(columns=columnas)

# Agrupar los datos por 'Título' y 'URL'
agrupado = df.groupby(['Título', 'URL'])

# Iterar sobre cada grupo para construir el nuevo DataFrame
nuevas_filas = []

for (titulo, url), grupo in agrupado:
    # Crear una nueva fila con el título y URL
    nueva_fila = {'Título': titulo, 'URL': url}
    
    # Añadir las definiciones correspondientes a los términos
    for _, fila in grupo.iterrows():
        termino = fila['Término']
        definicion = fila['Definición']
        nueva_fila[termino] = definicion
    
    # Añadir la fila a la lista de nuevas filas
    nuevas_filas.append(nueva_fila)

# Crear el DataFrame final
df_transformado = pd.DataFrame(nuevas_filas, columns=columnas)

# Guardar el DataFrame transformado en un nuevo archivo CSV
df_transformado.to_csv('datos_rutas_transformado_5.csv', index=False)

print("Los datos transformados se han guardado en datos_rutas_transformado.csv")
