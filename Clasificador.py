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
        self.organizado = os.path.join('ImagenesEntrenamiento', 'Organizado.jpg')
        self.intermedio = os.path.join('ImagenesEntrenamiento', 'Intermedio.jpg')
        self.desorganizado = os.path.join('ImagenesEntrenamiento', 'Desorganizado.jpg')

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
        organizado_url = self.local_image_to_data_url(self.organizado)
        intermedio_url = self.local_image_to_data_url(self.intermedio)
        desorganizado_url = self.local_image_to_data_url(self.desorganizado)
        imagen_evaluar_data_url = self.local_image_to_data_url(imagen_evaluar_path)

        # Crear la lista de mensajes para enviar a la API
        messages = [
            { "role": "system", "content": """
             Tu objetivo es clasificar si un refrigerador está organizado, medianamente orgnizado o desorganizado.
             Para ellos se te proveerán de imagenes de base para que puedas saber a qué se refiere cada clase.
             Tipo de clasificación multiclase:
             * Organizado
             * Medianamente organizado
             * Desorganizado
             """ },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un refrigerador ordenado ya que NO tiene huecos sin llenar de productos, y aun más importante los productos similares se encuentran cercanos, la leche con la leche, las verduras juntas, y las frutas en otro lado juntas. Es atractivo para los clientes." },
                    { "type": "image_url", "image_url": { "url": organizado_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un refrigerador medianamente ya que hay huecos entre productos sin llenar, aunque sí cumple en cuanto a la cercanía entre productos similares sin combinar de diferentes tipos."},
                    { "type": "image_url", "image_url": { "url": intermedio_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Este es un refrigerador desorganizado ya que los productos están por todos lados, algunos acostados, otros parados y no es atractivo para los clientes. Además, algunas marcas se combinan entre ellas, agrupándose unas entre otras." },
                    { "type": "image_url", "image_url": { "url": desorganizado_url } }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": """
                    Basándote en los ejemplos anteriores, clasifica la siguiente imagen según su nivel de organización. Asegúrate de que las recomendaciones estén orientadas específicamente para un **supermercado**.

                    El formato será el siguiente:

                    **Decisión**: (Selecciona entre 'Organizado', 'Medianamente organizado' o 'Desorganizado')

                    **Descripción**: (Describe brevemente qué ves en la imagen y qué elementos destacan. Sé claro y directo.)

                    **Recomendación**: (Proporciona una recomendación específica y detallada para mejorar la organización, enfocándote en un entorno de supermercado. Usa nombres, colores, marcas, estilos o cualquier detalle relevante. Sé breve pero útil, y evita información innecesaria. Por ejemplo:  
                    - "Noto que las botellas de Coca-Cola están mezcladas con otras marcas. Sugiero agrupar todas las botellas de Coca-Cola juntas, alineadas en la parte superior del estante, mientras que las botellas de Pepsi podrían estar en la fila inferior para diferenciarlas claramente por marca."  
                    - "Veo bolsas de frutos rojos mezcladas con bolsas de verduras verdes. Recomiendo colocar las bolsas de frutos rojos —como las que tienen etiquetas rojas— en el lado izquierdo del estante, y las bolsas de verduras en el lado derecho para separar las categorías."  

                    Si todo está bien organizado, en **Recomendación** escribe: 'No hay recomendación, ¡todo está perfecto!'

                    **Nota**: Recuerda que esta evaluación está diseñada para un supermercado, por lo que las recomendaciones deben ser prácticas para este contexto. Por ejemplo, prioriza la agrupación por marca o categoría de producto en lugar de características como tamaño o color, ya que esto es más relevante para un cliente en este entorno.

                    """ },
                    { "type": "image_url", "image_url": { "url": imagen_evaluar_data_url } }
                ]
            }
        ]

        # Enviar la solicitud a la API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=1_500,
            temperature=0.1
        )

        # Devolver solo el contenido de la respuesta
        return response.choices[0].message.content


# # Ejemplo de uso
# clasificador = ImageClassificator()
# resultado = clasificador.clasificar_pasillo(imagen_evaluar_path=r'C:\Users\EmilioSandovalPalomi\OneDrive - Mobiik\Documents\Sigma\Demo_Refrigeradores_vision\ImagenesPreCargadas\DSCN7183.jpg')
# print(resultado)
