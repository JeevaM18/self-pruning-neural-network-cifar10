import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.network import PrunableNN

st.set_page_config(
    page_title="Self-Pruning Neural Network",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
h1 {
    color: #38bdf8;
    text-align: center;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e293b;
    text-align: center;
    font-size: 20px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

model = PrunableNN()
model.load_state_dict(
    torch.load("outputs/model_lambda_0.01.pth", map_location="cpu")
)
model.eval()

st.title("Self-Pruning Neural Network")
st.markdown("### CIFAR-10 Image Classification with Dynamic Pruning")

st.write("Upload an image to see model prediction")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    conf = confidence.item() * 100

    with col2:
        st.markdown(f"""
        <div class="result-box">
        Prediction: <b>{label}</b><br><br>
        Confidence: <b>{conf:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)