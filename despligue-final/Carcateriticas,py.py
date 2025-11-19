# prepare_data.py

import os
import numpy as np
import pandas as pd
from landmark_pipeline import landmark_pipeline  # Importa tu pipeline
from sklearn.model_selection import train_test_split

# --- Configuración ---
TRAIN_DIR = 'asl_alphabet_train'
CSV_FILENAME = 'asl_features_39d.csv'
NOISE_CLASSES = ['del', 'nothing', 'space']


def create_feature_csv():
    """Ejecuta el pipeline en el dataset de entrenamiento y guarda el resultado en CSV."""
    print("--- 1. Extracción de Características y Filtrado de Ruido ---")

    X_data = []
    y_data = []
    skipped_count = 0

    sorted_items = sorted(os.listdir(TRAIN_DIR))

    for item in sorted_items:
        path = os.path.join(TRAIN_DIR, item)

        if os.path.isdir(path):
            class_name = item

            if class_name in NOISE_CLASSES:
                print(f"  Saltando clase de ruido: {class_name}")
                continue

            print(f"  Procesando clase: {class_name}...")

            for filename in os.listdir(path):
                image_path = os.path.join(path, filename)
                features = landmark_pipeline(image_path)

                if features is not None:
                    # **VERIFICACIÓN DE TAMAÑO**
                    if len(features) != 39:
                        print(
                            f"⚠️ Advertencia: Clase {class_name} devolvió {len(features)} características, se espera 39. Saltando.")
                        continue

                    X_data.append(features)
                    y_data.append(class_name)
                else:
                    skipped_count += 1

    print("-" * 40)
    print(f"Total de muestras procesadas (39D): {len(X_data)}")
    print(f"Total de muestras omitidas (MediaPipe): {skipped_count}")

    # 2. Creación del DataFrame
    X_df = pd.DataFrame(X_data)
    X_df['target'] = y_data

    # 3. Guardar en CSV
    X_df.to_csv(CSV_FILENAME, index=False)
    print(f"✅ Datos de entrenamiento guardados en: {CSV_FILENAME}")
    print("-" * 40)


if __name__ == '__main__':
    # Ejecuta este script SÓLO una vez para generar el CSV.
    create_feature_csv()