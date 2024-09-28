import csv
from selenium import webdriver
from selenium.webdriver.common.by import By

def obtener_urls(urls):
    # Inicializa el controlador de Chrome (asegúrate de tener Chromedriver instalado)
    driver = webdriver.Chrome()

    try:
        unique_urls = set()  # Conjunto para almacenar URLs únicas

        for url in urls:
            driver.get(url)

            # Encuentra todos los elementos <a> con la clase "trail-card-with-description__header"
            link_elements = driver.find_elements(By.CSS_SELECTOR, '.trail-card-with-description__header a')

            # Recorre todos los elementos y obtiene sus valores de href
            href_values = [link_element.get_attribute('href') for link_element in link_elements]

            # Agrega las URLs únicas al conjunto
            unique_urls.update(href_values)

            print(f"Datos de {url} obtenidos.")

        # Exporta las URLs únicas a un archivo CSV
        with open('wikiloc_urls_v2.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['URL'])  # Escribe el encabezado
            for href in unique_urls:
                writer.writerow([href])  # Escribe cada URL en una nueva fila

        print("Datos exportados exitosamente a 'wikiloc_urls_v2.csv'.")
    except Exception as e:
        print("Error al obtener las páginas:", str(e))
    finally:
        driver.quit()  # Cierra el navegador al finalizar

# Lista de URLs a procesar
urls_a_procesar = [
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=1',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=2',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=3',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=4',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=5',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=6',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=7',
    'https://es.wikiloc.com/rutas/senderismo/peru/cusco?page=8',
    # Agrega más URLs si es necesario
]

obtener_urls(urls_a_procesar)
