
import numpy as np
import json
import os

def model_fn(model_dir):
    # Carga el modelo desde el directorio de artefactos
    print("Cargando modelo desde: " + model_dir)
    weights = np.load(os.path.join(model_dir, "model_weights.npy"))
    return weights

def input_fn(request_body, request_content_type):
    # Procesa la entrada. Espera JSON.
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("Este modelo solo soporta application/json")

def predict_fn(input_data, model):
    # Realiza la predicción
    # model es el vector de pesos w_final
    data = np.array(input_data)

    # Manejo de dimensiones
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Agregar Sesgo (Bias) - Columna de 1s al inicio
    # Asumimos que el cliente envía clean features sin bias
    m = data.shape[0]
    data_bias = np.hstack([np.ones((m, 1)), data])

    # Función Sigmoide
    z = np.dot(data_bias, model)
    # Clip para estabilidad
    z = np.clip(z, -500, 500)
    prob = 1.0 / (1.0 + np.exp(-z))

    return prob

def output_fn(prediction, response_content_type):
    # Formatea la salida
    return json.dumps(prediction.tolist())
