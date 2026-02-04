import json
import numpy as np
import os
from flask import Flask, request, jsonify
from flasgger import Swagger

# Configuración
MODEL_DIR = 'local_deployment'
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'model_weights.npy')

template = {
  "swagger": "2.0",
  "info": {
    "title": "Heart Disease Prediction API",
    "description": "API para predicción de riesgo de enfermedad cardíaca usando Regresión Logística.\\n\\n**Cómo usar:**\\nEnvía una petición POST a `/predict` con los datos clínicos del paciente.\\n\\n**Formato de entrada:**\\nLista de valores normalizados: `[Age, Sex, BP, Cholesterol, Max HR, ST depression]`",
    "version": "1.0.0"
  }
}

app = Flask(__name__)
swagger = Swagger(app, template=template)

# Cargar Modelo al iniciar
print("Cargando modelo...")
if not os.path.exists(WEIGHTS_FILE):
    print(f"Error: No se encontró {WEIGHTS_FILE}. Asegúrate de ejecutar la Sección 11 del notebook primero.")
    w_final = None
else:
    w_final = np.load(WEIGHTS_FILE)
    print("Modelo cargado exitosamente.")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicción de Enfermedad Cardíaca
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: array
          items:
            type: number
          example: [0.5, 1.0, 0.2, -0.5, -1.0, 2.0]
        description: Lista de features normalizados [Age, Sex, BP, Cholesterol, Max HR, ST depression]
    responses:
      200:
        description: Probabilidad predicha de enfermedad cardíaca
        schema:
          type: array
          items:
            type: number
            example: 0.85
      400:
        description: Error en los datos de entrada
      500:
        description: Modelo no cargado
    """
    if w_final is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        data = request.get_json(force=True)
        # Esperamos una lista de features: [Age, Sex, BP, Cholesterol, Max HR, ST depression]
        # O una lista de listas para batch
        
        input_data = np.array(data)
        
        # Manejo de dimensiones (1D a 2D)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        m = input_data.shape[0]
        # Agregar Bias
        data_bias = np.hstack([np.ones((m, 1)), input_data])
        
        # Predicción
        z = np.dot(data_bias, w_final)
        prob = sigmoid(z)
        
        return jsonify(prob.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return "<h1>Modelo de Enfermedad Cardíaca - API</h1><p>Ir a <a href='/apidocs'>/apidocs</a> para ver Swagger UI.</p>"

if __name__ == '__main__':
    print("\n---------------------------------------------------------")
    print("Swagger UI disponible en: http://localhost:5000/apidocs")
    print("---------------------------------------------------------\n")
    app.run(debug=True, port=5000)
