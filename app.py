import streamlit as st
from PIL import Image
import os
import sys

# Adicionar src ao path
sys.path.append("src")

from predict import load_trained_model, predict_image

MODEL_PATH = "outputs/model_tcc.pth"

st.set_page_config(page_title="Detector de Deepfake", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🔎 Detector de Imagens Geradas por IA</h1>
    <p style='text-align: center;'>Classificação binária baseada em ResNet18</p>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return load_trained_model(MODEL_PATH)

model = load_model()

st.divider()

uploaded_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Imagem enviada", use_container_width=True)

    if st.button("🚀 Analisar Imagem"):

        result = predict_image(model, image)

        classe = result["class"]
        confidence = result["confidence"] * 100

        st.divider()

        if classe == "REAL":
            st.success("Resultado: IMAGEM REAL ✅")
        else:
            st.error("Resultado: IMAGEM GERADA POR IA ❌")

        st.write(f"Confiança do modelo: {confidence:.2f}%")
        st.progress(int(confidence))

        st.caption("Protótipo acadêmico - TCC")