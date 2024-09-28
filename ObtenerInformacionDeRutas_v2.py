import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


# Configuración de Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Ejecutar en modo sin cabeza (sin ventana del navegador)
driver = webdriver.Chrome(options=options)

# Función para leer las URLs desde un archivo CSV
def leer_urls_desde_csv(nombre_archivo):
    urls = []
    with open(nombre_archivo, 'r', newline='', encoding='utf-8') as archivo_csv:
        reader = csv.DictReader(archivo_csv)
        for fila in reader:
            urls.append(fila['URL'])
    return urls

# Función para obtener la URL de la imagen de la ruta
def obtener_url_imagen(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        imagenes = soup.find_all('img')
        for img in imagenes:
            src = img.get('src')
            if src and src.startswith("https://s0.wklcdn.com"):
                return src
    return None

# Nombre del archivo CSV que contiene las URLs
archivo_csv = 'wikiloc_urls_v2.csv'

# Obtener la lista de URLs desde el archivo CSV
urls = leer_urls_desde_csv(archivo_csv)

# Crear un archivo CSV para almacenar los datos
with open('datos_rutas_desde_csv_v2.csv', 'w', newline='', encoding='utf-8') as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(['Título', 'URL', 'Término', 'Definición', 'URL_Imagen'])  # Agregar 'URL_Imagen' como nueva columna

    # Procesar todas las URLs
    for url in urls:
        # Abrir la página web con Selenium
        driver.get(url)

        # Obtener el título de la página
        titulo = driver.title

        # Encontrar todos los elementos <div class="d-item">
        elementos_div = driver.find_elements(By.CLASS_NAME, 'd-item')

        # Lista para almacenar términos y definiciones
        terminos_definiciones = []

        # Iterar sobre los elementos <div> para encontrar <dt> y <dd>
        for elemento_div in elementos_div:
            terminos = elemento_div.find_elements(By.TAG_NAME, 'dt')
            definiciones = elemento_div.find_elements(By.TAG_NAME, 'dd')

            # Agregar cada término y definición a la lista
            for termino, definicion in zip(terminos, definiciones):
                terminos_definiciones.append([titulo, url, termino.text, definicion.text])

        # Obtener la URL de la imagen una sola vez por título
        url_imagen = obtener_url_imagen(url)

        # Escribir todos los términos y definiciones en el archivo CSV
        for td in terminos_definiciones:
            writer.writerow(td)

        # Escribir una fila adicional con el título y la URL de la imagen al final
        if url_imagen:
            writer.writerow([titulo, url, "URL_Imagen", url_imagen])

# Cerrar el navegador
driver.quit()

print("Los datos se han guardado en datos_rutas_desde_csv_v2.csv")
