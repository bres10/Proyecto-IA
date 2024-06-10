import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Cargar el modelo guardado
model = tf.keras.models.load_model('emotion_classifier_model.keras')

# Definir las etiquetas de clase manualmente
class_labels = ['anger', 'contempt', 'disgust', 'happy', 'sad', 'surprise']

# Ruta a la imagen que deseas analizar
img_path = 'pruebas caras/12 (2).jpg'

# Preprocesar la imagen
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Realizar la predicción
preds = model.predict(img_array)

# Decodificar la predicción
predicted_class = class_labels[np.argmax(preds)]
print(f"Predicción: {predicted_class}")
