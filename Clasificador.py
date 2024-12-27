import base64
import os
from mimetypes import guess_type
import streamlit as st
from openai import AzureOpenAI

class ImageClassificator:
    def __init__(self):
        # Configuración de la API
        self.api_base = st.secrets["AZURE_OAI_ENDPOINT"]
        self.api_key = st.secrets["AZURE_OAI_KEY"]
        self.deployment_name = st.secrets["AZURE_OAI_DEPLOYMENT"]
        self.api_version = "2024-02-15-preview"

        # Inicializar el cliente de Azure OpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )

        # Rutas predefinidas para las imágenes de ejemplo
        self.bueno = os.path.join('ImagenesEntrenamiento', 'Bueno.jpeg')
        self.malo = os.path.join('ImagenesEntrenamiento', 'Malo.jpeg')

    def local_image_to_data_url(self, image_path):
        """Codifica una imagen local en formato de data URL."""
        # Verificar si el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró el archivo: {image_path}")

        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def clasificar_pasillo(self, imagen_evaluar_path):
        """Clasifica la imagen del pasillo en función de los ejemplos proporcionados."""
        # Codificar las imágenes en formato de data URL
        bien_instalado_url = self.local_image_to_data_url(self.bueno)
        mal_instalado_url = self.local_image_to_data_url(self.malo)
        imagen_evaluar_data_url = self.local_image_to_data_url(imagen_evaluar_path)

        # Crear la lista de mensajes para enviar a la API
        messages = [
            { "role": "system", "content": """
             Tu objetivo es clasificar si un ONT está bien resguardado o no.
             Tipo de clasificación binaria:
             * Mal instalado
             * Correctamente instalado
             """ },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un ONT bien instalado ya que el está dentro de la canaleta" },
                    { "type": "image_url", "image_url": { "url": bien_instalado_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un  ONT mal instalado ya que el cable se sale de la canaleta" },
                    { "type": "image_url", "image_url": { "url": mal_instalado_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": """
                    Basándote en los ejemplos anteriores, clasifica la siguiente imagen. 
                    
                    El formato será el siguiente:

                    Decisión: (Puedes devolver Correctamente instalado / Mal instalado)
                    
                    Descripción: (detalladamente, qué ves en la imagen y por qué tomas la decisión. Incluye si el cable se sale de un lado, izquierdo, derecho, completamente salido, etc)

                    Justificación: (Aquí pones la justificación en un párrafo o en lista, como creas conveniente.)
                    """ },
                    { "type": "image_url", "image_url": { "url": imagen_evaluar_data_url } }
                ]
            }
        ]

        # Enviar la solicitud a la API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=2000
        )

        # Devolver solo el contenido de la respuesta
        return response.choices[0].message.content


# Ejemplo de uso
# clasificador = ImageClassificator()
# resultado = clasificador.clasificar_pasillo(imagen_evaluar_path=r'C:\Users\EmilioSandovalPalomi\OneDrive - Mobiik\Documents\Oxxo\UISupermarket\ImagenesPreCargadas\71913150_ZAsw8uWgddbYuzBYMNkI9xFGaOqy08W6h4J_3uuI3ZA.jpg')
# print(resultado)
