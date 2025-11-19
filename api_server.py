# api_server.py
# INSTALAR: pip install Flask pillow scikit-learn numpy opencv-python

from flask import Flask, request, jsonify
import pickle
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import base64
import os
from sklearn.preprocessing import StandardScaler  # Necesario para cargar la clase

# Importar funciones clave del proyecto
from landmark_pipeline import landmark_pipeline  # Asumimos que landmark_pipeline.py está disponible
from model import SignMLP, DEVICE, INPUT_SIZE, HIDDEN_SIZE  # Importar estructura MLP y DEVICE

app = Flask(__name__)

# --- CONFIGURACIÓN DE ARCHIVOS ---
MODEL_FILENAME = 'signsense_mlp_model.pth'
ENCODER_FILENAME = 'signsense_label_encoder.pkl'
SCALER_FILENAME = 'signsense_scaler.pkl'
NUM_CLASSES = 26  # Sabemos que el entrenamiento fue solo A-Z

# Variables globales para el modelo y componentes
MODEL = None
SCALER = None
ENCODER = None


def load_components():
    """Carga el modelo, el codificador y el escalador antes de que se inicie el servidor."""
    global MODEL, SCALER, ENCODER
    try:
        # 1. Cargar Escalador y Codificador
        with open(SCALER_FILENAME, 'rb') as f:
            SCALER = pickle.load(f)
        with open(ENCODER_FILENAME, 'rb') as f:
            ENCODER = pickle.load(f)

        # 2. Inicializar y Cargar el Modelo PyTorch
        model = SignMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Modo evaluación
        MODEL = model

        print("✅ Servidor listo: Componentes de ML cargados exitosamente.")
    except Exception as e:
        print(f"❌ ERROR AL CARGAR COMPONENTES: {e}")
        # Terminar la aplicación si falla la carga
        exit(1)


@app.before_request
def check_components():
    """Verifica si los componentes se cargaron antes de cada solicitud."""
    if MODEL is None:
        load_components()


@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    """Endpoint para recibir la imagen codificada en base64 y devolver la predicción."""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No se encontró el campo "image" en la solicitud'}), 400

        # Decodificar Base64 y convertir a formato de imagen
        image_bytes = base64.b64decode(data['image'])

        # Guardar temporalmente como JPG (necesario para la función landmark_pipeline basada en ruta/CV2)
        # Nota: En producción, esto se haría usando cv2.imdecode(np.frombuffer(...))
        temp_image_path = 'temp_frame.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        # 1. Pipeline de Extracción de Características
        features = landmark_pipeline(temp_image_path)
        os.remove(temp_image_path)  # Limpiar el archivo temporal

        if features is None:
            return jsonify({'prediction': 'Mano no detectada', 'code': ''}), 200

        # 2. Preprocesamiento (Escalado)
        X_new = np.array(features).reshape(1, -1)
        X_new_scaled = SCALER.transform(X_new)

        # 3. Inferencia de PyTorch
        with torch.no_grad():
            tensor_input = torch.tensor(X_new_scaled, dtype=torch.float32).to(DEVICE)
            output = MODEL(tensor_input)
            _, predicted_index = torch.max(output.data, 1)

        # 4. Decodificación
        predicted_class = ENCODER.inverse_transform(predicted_index.cpu().numpy())[0]

        return jsonify({'prediction': predicted_class, 'code': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_components()
    # Ejecutar en modo debug para desarrollo. Para demo/prod, usar gunicorn/waitress
    app.run(host='0.0.0.0', port=5000)