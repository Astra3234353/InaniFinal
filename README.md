# 🚀 SignSense — Traductor de Lenguaje de Señas Americano (ASL)

**SignSense** es un proyecto de hackathon que implementa una solución de Visión por Computadora de baja latencia para la traducción de las **26 letras** del alfabeto manual ASL. La arquitectura usa comunicación **Cliente–Servidor** para simular un despliegue real.

---

## ⚙️ Estructura del Proyecto

```
signsense/
├── api_server.py           # 🧠 Backend de Inferencia (Terminal 1)
├── demo_client.py          # 🎥 Frontend de Demo (Terminal 2: Cámara y Control)
├── model.py                # Definición de la arquitectura MLP (PyTorch/CUDA)
├── landmark_pipeline.py    # Pipeline de preprocesamiento (CLAHE, Ángulos)
└── signsense_*.pkl/.pth    # Archivos de Modelo y Componentes (pesos, escalador, encoder)
```

> Los archivos entre corchetes (`[]`) indican módulos clave y los archivos `signsense_*.pkl/.pth` corresponden a los modelos/artefactos entrenados necesarios para la inferencia.

---

## 📦 Requisitos e Instalación

Instala las dependencias necesarias (se asume Python 3.8+):

```bash
pip install torch torchvision torchaudio Flask requests opencv-python pillow
```

> Si usas GPU, asegúrate de instalar la versión de `torch` compatible con tu CUDA. Consulta la documentación oficial de PyTorch si necesitas una versión específica.

---

## 🚀 Despliegue y Ejecución (Dos Terminales)

La demo requiere que el servidor y el cliente se inicien en un orden específico.

### 1. Iniciar el Servidor (Terminal 1 — Backend) 🧠

El servidor de inferencia cargará los archivos del modelo (pesos, escalador y codificador) y escuchará en el puerto **5000**.

Abre la **Terminal 1** y ejecuta:

```bash
python api_server.py
```

Deberías ver:

```
✅ Servidor listo: Componentes de ML cargados exitosamente.
```

Mantén esta terminal abierta mientras el servidor esté en ejecución.

---

### 2. Iniciar la Demo (Terminal 2 — Frontend) 🎥

El cliente iniciará la cámara web y esperará la señal de inicio (`s`) para comenzar a enviar frames al servidor.

Abre la **Terminal 2** y ejecuta:

```bash
python demo_client.py
```

---

## 3. Control y Funcionamiento (Lógica Síncrona)

Una vez que el cliente esté en ejecución, se abrirá una ventana de la cámara web.

| Acción          | Tecla / Instrucción | Resultado / Comportamiento                                                                                                         |
| --------------- | ------------------: | ---------------------------------------------------------------------------------------------------------------------------------- |
| Iniciar Captura |        Presiona `s` | Comienza la captura: se envía un frame cada **3 segundos**.                                                                        |
| Lógica Síncrona |                   — | No se enviará un nuevo frame hasta que el servidor haya respondido al frame anterior.                                              |
| Visualización   |                   — | La ventana de la cámara mostrará la última letra válida reconocida. El historial de estado y conexión se imprime en la Terminal 2. |
| Finalizar       |        Presiona `q` | Detiene la cámara y cierra el script cliente.                                                                                      |

---

## Notas Técnicas

* **Preprocesamiento**: `landmark_pipeline.py` aplica mejoras de contraste (CLAHE), normalización de puntos clave y cálculo de ángulos relevantes para la MLP.
* **Modelo**: `model.py` contiene la definición del modelo MLP implementado en PyTorch; los pesos se cargan desde los archivos `signsense_*.pth` o `*.pkl`.
* **Comunicación**: El cliente captura frames, extrae landmarks (si aplica) y los envía al servidor vía HTTP (o el método especificado en `api_server.py` / `demo_client.py`) para inferencia síncrona.
* **Latencia**: La lógica está diseñada para baja latencia y evita el envío concurrente de múltiples frames sin respuesta del servidor.

---

## Sugerencias / Mejoras Futuras

* Soporte para palabras completas (secuencias de letras) y detección de palabras comunes.
* Pipeline de landmarks optimizado con modelos de detección de manos más robustos.
* Interfaz web para despliegue más amigable y visualización del historial.
* Optimización y quantización del modelo para despliegue en dispositivos edge.

---

## Licencia

Incluye aquí la licencia que prefieras (por ejemplo, MIT).
Ejemplo:

```
MIT License
```

---

## Contacto

Si quieres colaborar, reportar issues o mejorar el proyecto: abre un *issue* en el repositorio o envía un PR con tus cambios.