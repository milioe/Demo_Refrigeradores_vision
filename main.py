import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tempfile import NamedTemporaryFile
from Clasificador import ImageClassificator  # Importar la clase ImageClassificator

# Configurar Streamlit para que se muestre en modo "wide"
st.set_page_config(page_title="Clasificación de ONTs", layout="wide")

# Instanciar el clasificador de imágenes
clasificador = ImageClassificator()

# Interfaz de Streamlit
st.title("Detección de ONTs")

# Sidebar para drag and drop y selección de imágenes pre-cargadas
with st.sidebar:
    # Subir una imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    st.write("**... o selecciona una imagen pre-cargada:**")
    
    # Listar imágenes pre-cargadas en el directorio
    PRELOADED_IMAGES_DIR = r"ImagenesPreCargadas"
    preloaded_images = os.listdir(PRELOADED_IMAGES_DIR)
    selected_preloaded_image = None
    
    # Enumerar y mostrar botones con nombres "Imagen 1", "Imagen 2", etc.
    for index, image_name in enumerate(preloaded_images):
        display_name = f"Imagen {index + 1}"
        if st.button(display_name):
            selected_preloaded_image = os.path.join(PRELOADED_IMAGES_DIR, image_name)

# Crear columnas para mostrar la imagen y la respuesta lado a lado
col1, col2 = st.columns(2)

if uploaded_file or selected_preloaded_image:
    try:
        # Determinar la imagen que se va a procesar
        image_path = None
        if uploaded_file:
            # Si se subió un archivo, guardarlo temporalmente
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                image_path = temp_file.name
        elif selected_preloaded_image:
            # Si se seleccionó una imagen precargada
            image_path = selected_preloaded_image

        with col1:
            # Mostrar la imagen seleccionada o subida
            image = Image.open(image_path)
            st.image(image, caption="Imagen Analizada", use_column_width=True)

        with col2:
            # Llamar a la función de clasificación de la clase ImageClassificator
            with st.spinner("Generando predicción..."):
                resultado = clasificador.clasificar_pasillo(image_path)
            
            # Mostrar la respuesta
            st.write("**Resultado de la clasificación:**")
            st.write(resultado)

    except (UnidentifiedImageError, ValueError, RuntimeError) as e:
        st.toast(f"🚨 No se ha podido procesar la imagen: {str(e)}")
    finally:
        # Eliminar el archivo temporal si se creó
        if uploaded_file and image_path:
            os.remove(image_path)
