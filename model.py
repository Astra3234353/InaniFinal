# model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler

# 🚨 Soporte CUDA: Detecta automáticamente la GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# --- Configuración del Modelo y Entrenamiento ---
INPUT_SIZE = 39
HIDDEN_SIZE = 128
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class SignMLP(nn.Module):
    """Red Neuronal Multicapa (MLP) simple para clasificación de señas."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(SignMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def train_mlp_model(X_train, y_train, X_test, y_test):
    """Entrena el modelo MLP de PyTorch, usando CUDA si está disponible."""
    print("\n--- Preparando el Entrenamiento de MLP (PyTorch) ---")

    # 1. Codificación de Etiquetas
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    print(f"Número de clases detectadas: {num_classes}")

    # 2. Escalado de Características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Conversión a Tensores y DataLoader
    train_data = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.long)
    )
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Inicialización y Movimiento a DEVICE
    model = SignMLP(INPUT_SIZE, HIDDEN_SIZE, num_classes).to(DEVICE)  # 🚨 Mover el MODELO a la GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Bucle de Entrenamiento
    print(f"Iniciando entrenamiento por {NUM_EPOCHS} épocas...")
    for epoch in range(NUM_EPOCHS):
        for i, (features, labels) in enumerate(train_loader):
            # 🚨 Mover DATOS a la GPU en cada iteración
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # LOGGING: Evaluación del progreso
        if (epoch + 1) % 10 == 0:
            train_acc = evaluate_model(model, X_train_scaled, y_train_encoded, scaler, batch_size=512)
            print(
                f'Época [{epoch + 1}/{NUM_EPOCHS}], Pérdida: {loss.item():.4f}, Precisión de Entrenamiento: {train_acc:.4f}')

    # 6. Evaluación Final
    test_accuracy = evaluate_model(model, X_test_scaled, y_test_encoded, scaler, batch_size=BATCH_SIZE)
    print(f"\nPrecisión Final del Modelo en la Prueba: {test_accuracy:.4f}")

    return model, label_encoder, scaler, test_accuracy


def evaluate_model(model, X_data_scaled, y_data_encoded, scaler, batch_size):
    """Calcula la precisión del modelo en un conjunto de datos dado, usando DEVICE."""
    model.eval()  # Modo evaluación
    with torch.no_grad():
        data = TensorDataset(
            torch.tensor(X_data_scaled, dtype=torch.float32),
            torch.tensor(y_data_encoded, dtype=torch.long)
        )
        data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        for features, labels in data_loader:
            # 🚨 Mover datos a la GPU
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()  # Volver al modo entrenamiento
    return correct / total