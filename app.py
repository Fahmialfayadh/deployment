import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# 1. Konfigurasi Halaman & CSS Monokrom Hitam-Putih Penuh (Override Dark Mode)
st.set_page_config(page_title="Fashion-MNIST Classifier", layout="wide")

st.markdown("""
    <style>
    /* Paksa Light Mode & Warna Dasar */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    header, footer, #MainMenu { visibility: hidden; }
    * {
        border-radius: 0px !important;
        box-shadow: none !important;
    }
    p, span, label, h1, h2, h3, h4, small {
        color: #000000 !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FAFAFA !important;
        border: 1px solid #000000 !important;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] p {
        color: #555555 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Override File Uploader (Fix Kotak Hitam) */
    [data-testid="stFileUploader"] section {
        background-color: #FAFAFA !important;
        border: 1px dashed #000000 !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    [data-testid="stFileUploader"] section button span {
        color: #FFFFFF !important;
    }
    [data-testid="stFileUploader"] small {
        color: #555555 !important;
    }

    /* Override Expander (Fix Header Hitam) */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    [data-testid="stExpander"] details summary {
        background-color: #FAFAFA !important;
        color: #000000 !important;
    }
    [data-testid="stExpander"] details summary:hover {
        background-color: #F0F0F0 !important;
    }

    /* Override Alert / Info Box (Fix Kotak Biru Gelap) */
    [data-testid="stAlert"] {
        background-color: #FAFAFA !important;
        border: 1px solid #000000 !important;
        color: #000000 !important;
    }
    [data-testid="stAlert"] svg {
        fill: #000000 !important;
    }

    /* Progress Bar Hitam-Putih */
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #000000 !important;
    }
    [data-testid="stProgress"] > div > div > div {
        background-color: #E5E5E5 !important;
    }

    /* Divider & Kartu Hasil */
    hr { border-top: 1px solid #000000 !important; margin: 1.5rem 0; }
    .result-box {
        border: 2px solid #000000;
        padding: 20px;
        background-color: #FAFAFA;
        margin-bottom: 16px;
    }
    .pred-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #555555;
        margin-bottom: 4px;
    }
    .pred-value {
        font-size: 2rem;
        font-weight: 800;
        color: #000000;
        line-height: 1.2;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Definisi Arsitektur Model (CNN_BN_Dropout)
class FashionModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(FashionModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 3. Load Model ke CPU (Cache)
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = FashionModel().to(device)
    state_dict = torch.load("model_CNN_BN_Dropout.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# --- HEADER & STATISTIK MODEL ---
st.markdown("## FASHION-MNIST CLASSIFIER")
st.caption("Arsitektur: **CNN + Batch Normalization + Dropout** | Total Parameter: **421.834** | Input: **28×28 Grayscale**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy", "92.82%")
m2.metric("Precision (Macro)", "92.86%")
m3.metric("Recall (Macro)", "92.82%")
m4.metric("F1-Score (Macro)", "92.83%")

st.divider()

# --- AREA UTAMA ---
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown("#### 1. Input Citra")
    uploaded_file = st.file_uploader(
        "Unggah gambar pakaian (PNG/JPG)", 
        type=["png", "jpg", "jpeg"]
    )
    
    with st.expander("Opsi Preprocessing", expanded=True):
        invert = st.checkbox("Invert warna (centang jika latar belakang putih)", value=True)
        use_std_norm = st.checkbox("Terapkan Z-Score Standardization (StandardScaler)", value=False)

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("L")
        processed_image = ImageOps.invert(raw_image) if invert else raw_image
        
        st.markdown("**Pratinjau Input Model (Grayscale):**")
        st.image(processed_image, width=220)

with col_right:
    st.markdown("#### 2. Hasil Klasifikasi")
    
    if uploaded_file is not None:
        transform_steps = [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
        if use_std_norm:
            transform_steps.append(transforms.Normalize((0.2860,), (0.3530,)))
            
        transform = transforms.Compose(transform_steps)
        tensor_img = transform(processed_image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor_img)
            probs = torch.softmax(output, dim=1).squeeze(0)
            conf, pred = torch.max(probs, 0)

        st.markdown(f"""
            <div class="result-box">
                <div class="pred-label">Prediksi Kelas</div>
                <div class="pred-value">{CLASSES[pred.item()]}</div>
                <div style="margin-top: 8px; font-size: 0.95rem;">
                    Tingkat Keyakinan (Confidence): <b>{conf.item() * 100:.2f}%</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("**Distribusi Probabilitas (Top 3):**")
        top3_prob, top3_idx = torch.topk(probs, 3)
        
        for i in range(3):
            cls_name = CLASSES[top3_idx[i].item()]
            prob_val = top3_prob[i].item()
            st.write(f"{cls_name} — **{prob_val * 100:.2f}%**")
            st.progress(prob_val)
            
    else:
        st.info("Silakan unggah gambar pada panel di sebelah kiri untuk memulai klasifikasi.")