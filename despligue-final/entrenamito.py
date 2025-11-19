# train_and_save_model.py - CORREGIDO

import os
import numpy as np
import pickle
import pandas as pd
import torch
from model import train_mlp_model, DEVICE
from sklearn.model_selection import train_test_split

# No necesitamos importar landmark_pipeline si cargamos el CSV

# --- Configuración de Archivos ---
TEST_DIR = 'asl_alphabet_test'
CSV_FILENAME = 'asl_features_39d.csv'
MODEL_FILENAME = 'signsense_mlp_model.pth'
ENCODER_FILENAME = 'signsense_label_encoder.pkl'
SCALER_FILENAME = 'signsense_scaler.pkl'
NOISE_CLASSES = ['del', 'nothing', 'space']  # Clases que el modelo no conoce


def load_final_test_set(test_dir):
    """
    Carga el set de prueba final. Después de extraer las características,
    filtra las clases de ruido que el modelo no conoce.
    """
    try:
        from landmark_pipeline import landmark_pipeline
    except ImportError:
        print("❌ ERROR: No se puede importar 'landmark_pipeline'.")
        return None, None, 0, None

    X_data = []
    y_data = []
    filenames = []
    skipped_count = 0

    sorted_items = sorted(os.listdir(test_dir))

    for item in sorted_items:
        path = os.path.join(test_dir, item)

        if item.endswith(('.jpg', '.jpeg', '.png')):
            class_name = item.split('_')[0]

            # 💥 CORRECCIÓN: Filtrar el ruido AHORA, antes de codificar.
            if class_name in NOISE_CLASSES:
                # Omitir la muestra del set de prueba, ya que el modelo no conoce esta clase.
                continue

            features = landmark_pipeline(path)

            if features is not None:
                X_data.append(features)
                y_data.append(class_name)
                filenames.append(item)
            else:
                skipped_count += 1

    return np.array(X_data), np.array(y_data), skipped_count, filenames


def train_and_save_model():
    """Flujo principal: Cargar CSV, Entrenar MLP, Guardar componentes."""

    # 1. Cargar Datos de Entrenamiento DESDE CSV
    if not os.path.exists(CSV_FILENAME):
        print(f"❌ ERROR: El archivo de características '{CSV_FILENAME}' no existe.")
        print("Ejecuta 'python prepare_data.py' primero para crearlo. Terminando.")
        return

    print(f"✅ Cargando datos de entrenamiento desde {CSV_FILENAME}...")
    df = pd.read_csv(CSV_FILENAME)

    # El CSV ya debería estar filtrado, pero aseguramos la carga de X y Y
    X_train = df.drop(columns=['target']).values
    y_train = df['target'].values

    # 2. División de Datos
    X_train_use, X_val_split, y_train_use, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # 3. Cargar datos de prueba FINAL (ya filtrados en la función)
    print(f"\n📢 Cargando y filtrando conjunto de prueba final: {TEST_DIR}/")
    X_test_final, y_test_final, skipped_test, test_filenames = load_final_test_set(TEST_DIR)

    print("-" * 40)
    print(f"Entrenamiento para MLP: {len(X_train_use)} muestras (A-Z).")
    print(f"Prueba Final analizada: {len(X_test_final)} muestras (solo A-Z). Omitidas por MediaPipe: {skipped_test}")

    if len(X_train_use) < 100 or len(X_test_final) < 5:
        print("Error: Datos insuficientes para entrenamiento. Terminando.")
        return

    # 4. Entrenamiento del Modelo MLP (Ahora las etiquetas están garantizadas en A-Z)
    trained_model, label_encoder, scaler, final_accuracy = train_mlp_model(
        X_train_use, y_train_use, X_test_final, y_test_final)

    # 5. Guardar Modelo y Componentes
    # ... (Guardado de archivos, igual)
    torch.save(trained_model.state_dict(), MODEL_FILENAME)
    with open(ENCODER_FILENAME, 'wb') as file:
        pickle.dump(label_encoder, file)
    with open(SCALER_FILENAME, 'wb') as file:
        pickle.dump(scaler, file)

    print(f"\n✅ Componentes guardados: {MODEL_FILENAME}, {ENCODER_FILENAME}, {SCALER_FILENAME}.")

    # 6. Verificación Final Detallada
    X_test_scaled = scaler.transform(X_test_final)

    y_test_pred_encoded = trained_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).argmax(
        dim=1).cpu().numpy()
    y_pred_test = label_encoder.inverse_transform(y_test_pred_encoded)

    print("-" * 40)
    print(f"Precisión Final (MLP): {final_accuracy * 100:.2f}%")

    results_df = pd.DataFrame({
        'Archivo': test_filenames,
        'Clase Real': y_test_final,
        'Predicción': y_pred_test
    })
    results_df['Resultado'] = np.where(results_df['Clase Real'] == results_df['Predicción'], '✅ Correcto',
                                       '❌ Incorrecto')

    print("\n--- Resultados Detallados (MLP) ---")
    pd.set_option('display.max_rows', None)
    print(results_df)
    print("-" * 40)


if __name__ == '__main__':
    train_and_save_model()