# landmark_pipeline.py

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean

# Inicializar MediaPipe Hands UNA SOLA VEZ
mp_hands = mp.solutions.hands
HANDS_DETECTOR = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.35 
)

def apply_clahe(image):
    """Aplica la Ecualización de Histograma CLAHE para mejorar el contraste."""
    if image is None:
        return None
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels[0] = clahe.apply(channels[0]) 
    cv2.merge(channels, ycrcb) 
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def calculate_angle(p1, p2, p3):
    """Calcula el ángulo (en grados) en el punto p2 entre los vectores p2->p1 y p2->p3."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2 # Vector p2 -> p1
    v2 = p3 - p2 # Vector p2 -> p3

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    dot_product = np.dot(v1, v2)

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    
    # Clip para manejar posibles errores de precisión flotante
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_radians = np.arccos(cosine_angle)
    return np.degrees(angle_radians)


def calculate_geometric_features(landmarks):
    """
    Calcula un vector de características (distancias y ÁNGULOS) 
    invariantes a la traslación y escala.
    """
    if landmarks is None or len(landmarks) == 0:
        return None

    # Normalización: Traslación (Muñeca L0 a origen)
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    
    # Normalización: Escala (Factor L0 a L9)
    scale_factor = euclidean(normalized_landmarks[0], normalized_landmarks[9])
    if scale_factor < 1e-6: 
        return None
        
    scaled_landmarks = normalized_landmarks / scale_factor

    # --- Inicialización del Vector de Características ---
    features = []
    finger_tips = [4, 8, 12, 16, 20]
    
    # 1. Coordenadas de las puntas (10 características)
    for tip_index in finger_tips:
        features.extend(scaled_landmarks[tip_index].tolist()) 
        
    # 2. Distancias entre Puntas de Dedos (10 características)
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            dist = euclidean(scaled_landmarks[finger_tips[i]], scaled_landmarks[finger_tips[j]])
            features.append(dist)
            
    # 3. Distancia de Puntas de Dedos a la Muñeca L0 (5 características)
    for tip_index in finger_tips:
        dist = euclidean(scaled_landmarks[tip_index], [0, 0]) 
        features.append(dist)
        
    # 4. ÁNGULOS DE FLEXIÓN (16 características)
    # Tripletas de landmarks (p1, p2, p3) donde p2 es el vértice del ángulo
    angle_triplets = [
        # Pulgar (2 ángulos)
        (4, 3, 2), # Articulación IP
        (3, 2, 1), # Articulación MCP

        # Índice, Medio, Anular, Meñique (3 ángulos cada uno = 12 ángulos)
        # La seña de la mano es más robusta si se mide respecto a la muñeca (L0)
        # Índice
        (8, 7, 6), # DIP
        (7, 6, 5), # PIP
        (6, 5, 0), # MCP relativo a la muñeca

        # Medio
        (12, 11, 10), # DIP
        (11, 10, 9),  # PIP
        (10, 9, 0),   # MCP relativo a la muñeca

        # Anular
        (16, 15, 14), # DIP
        (15, 14, 13), # PIP
        (14, 13, 0),  # MCP relativo a la muñeca

        # Meñique
        (20, 19, 18), # DIP
        (19, 18, 17), # PIP
        (18, 17, 0)   # MCP relativo a la muñeca
    ]
    
    for i1, i2, i3 in angle_triplets:
        angle = calculate_angle(scaled_landmarks[i1], scaled_landmarks[i2], scaled_landmarks[i3])
        features.append(angle)
        
    # Total: 10 + 10 + 5 + 16 = 41 características
        
    return features


def landmark_pipeline(image_path):
    """
    Implementa el pipeline completo: Carga -> Ecualización -> Detección -> Características.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_eq = apply_clahe(image)
    results = HANDS_DETECTOR.process(cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB))
    
    if not results.multi_hand_landmarks:
        return None

    # Convertir a array de 21x2
    landmarks_raw = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks_raw.extend([lm.x, lm.y])
    landmarks = np.array(landmarks_raw).reshape(21, 2)
    
    return calculate_geometric_features(landmarks)