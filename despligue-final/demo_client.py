# app_web_demo.py
# INSTALAR: pip install Flask Flask-SocketIO opencv-python pillow

import os
import cv2
import time
import base64
import numpy as np
import threading
import pickle
import torch

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit, join_room, leave_room
from io import BytesIO

# Importar funciones y componentes de ML
from landmark_pipeline import landmark_pipeline
from model import SignMLP, DEVICE, INPUT_SIZE, HIDDEN_SIZE  # Importar estructura MLP y DEVICE

# --- CONFIGURACIÓN DE ML (Misma que api_server.py) ---
MODEL_FILENAME = 'signsense_mlp_model.pth'
ENCODER_FILENAME = 'signsense_label_encoder.pkl'
SCALER_FILENAME = 'signsense_scaler.pkl'
NUM_CLASSES = 26
CAPTURE_INTERVAL = 3.0  # Segundos entre envío de frames
# ----------------------------------------------------

# Variables Globales de Estado
MODEL = None
SCALER = None
ENCODER = None
is_running = False
camera = None
thread = None
last_send_time = 0.0
is_processing = False  # Control síncrono

# Inicialización de Flask y SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackathon_signsense_secret'
socketio = SocketIO(app)


# ----------------------------------------------------
# LÓGICA DE ML (Carga y Predicción)
# ----------------------------------------------------

def load_components():
    """Carga componentes de ML al iniciar la aplicación."""
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
        model.eval()
        MODEL = model

        print("✅ Servidor ML listo: Componentes cargados.")
        return True
    except Exception as e:
        print(f"❌ ERROR AL CARGAR COMPONENTES ML: {e}")
        return False


def get_prediction(frame):
    """Ejecuta el pipeline de ML para un frame dado."""
    global SCALER, MODEL, ENCODER, is_processing

    # Simular guardado de archivo temporal (necesario si landmark_pipeline usa rutas)
    # En producción, se usaría cv2.imdecode(np.frombuffer(...)) para evitar archivos temporales
    temp_image_path = 'temp_frame_web.jpg'
    cv2.imwrite(temp_image_path, frame)

    features = landmark_pipeline(temp_image_path)
    os.remove(temp_image_path)

    if features is None:
        return "Mano no detectada", ""

    # 2. Inferencia
    X_new = np.array(features).reshape(1, -1)
    X_new_scaled = SCALER.transform(X_new)

    with torch.no_grad():
        tensor_input = torch.tensor(X_new_scaled, dtype=torch.float32).to(DEVICE)
        output = MODEL(tensor_input)
        _, predicted_index = torch.max(output.data, 1)

    predicted_class = ENCODER.inverse_transform(predicted_index.cpu().numpy())[0]
    return predicted_class, 'OK'


# ----------------------------------------------------
# LÓGICA DE VIDEO Y WEBSOCKETS
# ----------------------------------------------------

def video_stream_thread():
    """Captura de video y lógica de envío de frames."""
    global camera, is_running, last_send_time, is_processing

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        socketio.emit('status', {'message': 'ERROR: No se pudo acceder a la cámara.'})
        return

    print("Hilo de video iniciado. Listo.")

    while is_running:
        ret, frame = camera.read()
        if not ret:
            break

        # 1. Mostrar frame en el navegador (para UX)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_feed', {'image': frame_base64})

        # 2. Lógica de Predicción (Control estricto)
        current_time = time.time()

        if current_time - last_send_time >= CAPTURE_INTERVAL and not is_processing:
            is_processing = True  # Bloquea el envío
            last_send_time = current_time

            # Ejecutar la predicción en un hilo separado para no bloquear la UI/Video
            def process_and_emit(current_frame):
                global is_processing

                prediction, status_code = get_prediction(current_frame)

                # Emitir el resultado
                socketio.emit('prediction_result', {'sign': prediction, 'status': status_code})

                is_processing = False  # Desbloquea el envío

            # Iniciar el proceso de ML en hilo separado
            threading.Thread(target=process_and_emit, args=(frame.copy(),)).start()

        # Pausa mínima para no saturar
        socketio.sleep(0.01)

    # Limpieza final
    camera.release()
    print("Hilo de video detenido.")


@app.route('/')
def index():
    """Sirve la página principal de la demo (Frontend)."""
    return render_template_string(HTML_PAGE)


@socketio.on('start_camera')
def handle_start_camera():
    """Inicia la captura de video y la lógica de predicción."""
    global is_running, thread
    if not is_running:
        is_running = True
        thread = threading.Thread(target=video_stream_thread, daemon=True)
        thread.start()
        emit('status', {'message': 'Cámara y Predicción iniciada.'})


@socketio.on('stop_camera')
def handle_stop_camera():
    """Detiene la captura de video."""
    global is_running, thread, camera
    if is_running:
        is_running = False
        if camera:
            camera.release()
            camera = None
        if thread and thread.is_alive():
            thread.join(timeout=1)
        emit('status', {'message': 'Cámara y Predicción detenida.'})


# ----------------------------------------------------
# PÁGINA HTML (Frontend)
# ----------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>SignSense Demo - Hackathon</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #f4f4f9; }
        .container { max-width: 900px; margin: 30px auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        #video-feed { width: 100%; max-width: 640px; border: 3px solid #ccc; border-radius: 8px; margin-bottom: 15px; }
        #prediction-display { font-size: 4em; font-weight: bold; color: #007bff; min-height: 100px; line-height: 100px; }
        #history { text-align: left; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }
        .control-btn { padding: 10px 20px; font-size: 1.1em; margin: 5px; cursor: pointer; border: none; border-radius: 5px; transition: background-color 0.3s; }
        .start { background-color: #28a745; color: white; }
        .stop { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SignSense - Traductor de Señas (MLP Demo)</h1>

        <div id="status">Esperando conexión...</div>

        <img id="video-feed" src="placeholder.jpg" alt="Video Feed">

        <div id="control-panel">
            <button class="control-btn start" onclick="startCamera()">Iniciar Cámara ('S')</button>
            <button class="control-btn stop" onclick="stopCamera()">Finalizar ('Q')</button>
        </div>

        <h2>Predicción Actual</h2>
        <div id="prediction-display">---</div>

        <div id="history-container">
            <h2>Historial de Señas</h2>
            <div id="history"></div>
        </div>
    </div>

    <script>
        // Conexión SocketIO
        var socket = io.connect('http://' + document.domain + ':' + location.port);
        var historyDiv = document.getElementById('history');
        var predictionDisplay = document.getElementById('prediction-display');
        var lastValidSign = '';

        // Manejadores de eventos de teclado
        document.addEventListener('keydown', function(event) {
            if (event.key === 's' || event.key === 'S') {
                startCamera();
            } else if (event.key === 'q' || event.key === 'Q') {
                stopCamera();
            }
        });

        function startCamera() {
            socket.emit('start_camera');
        }

        function stopCamera() {
            socket.emit('stop_camera');
        }

        // 1. Mostrar estado del servidor
        socket.on('status', function(data) {
            document.getElementById('status').innerText = 'Estado: ' + data.message;
        });

        // 2. Mostrar video en tiempo real
        socket.on('video_feed', function(data) {
            document.getElementById('video-feed').src = 'data:image/jpeg;base64,' + data.image;
        });

        // 3. Recibir predicción síncrona
        socket.on('prediction_result', function(data) {

            // Lógica de visualización: Solo si se detectó una seña válida
            if (data.sign !== 'Mano no detectada') {

                // Si la nueva predicción es diferente a la última válida, la añadimos al historial
                if (data.sign !== lastValidSign) {
                    historyDiv.innerHTML += '<span>' + data.sign + ' </span>';
                    lastValidSign = data.sign;
                }

                predictionDisplay.innerText = data.sign;
                predictionDisplay.style.color = '#28a745'; // Color verde al acertar
            } else {
                predictionDisplay.innerText = '¿?';
                predictionDisplay.style.color = '#dc3545'; // Color rojo si no hay detección
            }

            // Desplazar el historial al final
            historyDiv.scrollTop = historyDiv.scrollHeight; 
        });

    </script>
</body>
</html>
"""

if __name__ == '__main__':
    if load_components():
        print("Servidor web escuchando en http://127.0.0.1:5000")
        socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)