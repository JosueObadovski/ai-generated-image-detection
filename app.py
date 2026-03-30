import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

# Adicionar src ao path
sys.path.append("src")

from model import get_model

MODEL_PATH = "outputs/model_tcc.pth"

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Detector de Imagens Geradas por IA",
    layout="centered"
)

st.title("🔍 Detector de Imagens Geradas por IA")
st.caption("Protótipo acadêmico - TCC")

# =========================
# MÉTRICAS DO MODELO
# =========================
st.subheader("📊 Desempenho no Conjunto de Teste (CIFAKE)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "0.97")
col2.metric("F1-score", "0.97")
col3.metric("Precision", "0.97")
col4.metric("Recall", "0.97")

st.markdown("---")

# =========================
# CARREGAR MODELO
# =========================
@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# UPLOAD DE IMAGEM
# =========================
st.subheader("🖼️ Envie uma imagem para análise")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=400)

    if st.button("🔎 Analisar Imagem"):

        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        prob_fake = probabilities[0].item()
        prob_real = probabilities[1].item()

        if prob_real > 0.5:
            st.success("Resultado: IMAGEM REAL")
        else:
            st.error("Resultado: IMAGEM GERADA POR IA")

        st.write(f"Probabilidade REAL: {prob_real*100:.2f}%")
        st.write(f"Probabilidade FAKE: {prob_fake*100:.2f}%")

        # =========================
        # GRÁFICO DE PROBABILIDADE
        # =========================
        fig, ax = plt.subplots()
        ax.bar(["FAKE", "REAL"], [prob_fake, prob_real])
        ax.set_ylim([0,1])
        ax.set_ylabel("Probabilidade")
        st.pyplot(fig)

st.markdown("---")

# =========================
# MATRIZ DE CONFUSÃO
# =========================
if os.path.exists("outputs/matriz_confusao.png"):
    st.subheader("📌 Matriz de Confusão")
    st.image("outputs/matriz_confusao.png", width=400)