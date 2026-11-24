"""Demo: clasificación de tejido histopatológico con MobileNetV2 + Grad-CAM.

Uso:  streamlit run app.py
Requiere el archivo modelo_pathmnist.keras (generado por el notebook) en esta carpeta.
"""
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image

MODEL_SIZE = 96
CLASES = [
    "Tejido adiposo", "Fondo", "Detritos", "Linfocitos", "Mucus", "Músculo liso",
    "Mucosa colónica normal", "Estroma asociado a cáncer", "Epitelio de adenocarcinoma",
]

st.set_page_config(page_title="Clasificador de tejido H&E", page_icon="🔬", layout="centered")


@st.cache_resource
def cargar():
    modelo = keras.models.load_model("modelo_pathmnist.keras")
    gradcam_model = keras.Model(modelo.inputs, [modelo.get_layer("out_relu").output, modelo.output])
    return gradcam_model


def predecir(gradcam_model, img):
    x = tf.image.resize(tf.cast(img[None], tf.float32), (MODEL_SIZE, MODEL_SIZE))
    with tf.GradientTape() as tape:
        mapas, prob = gradcam_model(x, training=False)
        clase = int(tf.argmax(prob[0]))
        puntaje = prob[:, clase]
    grads = tape.gradient(puntaje, mapas)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(mapas[0] * pesos, axis=-1)).numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-8), (img.shape[1], img.shape[0]))
    return prob[0].numpy(), clase, cam


st.title("🔬 Clasificador de tejido histopatológico")
st.caption("MobileNetV2 entrenada sobre PathMNIST (tejido de colon, tinción H&E) · Interpretación con Grad-CAM")
st.warning("Proyecto de portfolio con fines educativos. No es una herramienta de diagnóstico.")

archivo = st.file_uploader("Subí un parche de tejido H&E (PNG/JPG)", type=["png", "jpg", "jpeg"])
if archivo:
    img = np.array(Image.open(archivo).convert("RGB"))
    prob, clase, cam = predecir(cargar(), img)
    calor = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
    superpuesta = np.uint8(0.6 * img + 0.4 * calor)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Imagen original", use_container_width=True)
    c2.image(superpuesta, caption="Grad-CAM: zonas que usó el modelo", use_container_width=True)

    st.subheader(f"Predicción: {CLASES[clase]} ({prob[clase]:.0%})")
    orden = np.argsort(prob)[::-1][:3]
    for i in orden:
        st.progress(float(prob[i]), text=f"{CLASES[i]}: {prob[i]:.1%}")
