import os
import torch
import streamlit as st
from PIL import Image
from model import build_model
from data_loader import get_transforms
import config

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── CSS: dark medical theme ───────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #080d1a; }
    [data-testid="stHeader"]           { background-color: #080d1a; }
    [data-testid="stToolbar"]          { display: none; }

    html, body, [class*="css"] {
        color: #e8eaf6;
        font-family: 'Segoe UI', sans-serif;
    }

    .header-banner {
        background: linear-gradient(135deg, #0d1b2e 0%, #1a2f4e 100%);
        border: 1px solid #1e4080;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 28px;
        text-align: center;
    }
    .header-banner h1 {
        font-size: 2rem; font-weight: 700;
        color: #ffffff; margin: 8px 0 4px 0;
    }
    .header-banner p { color: #8da9c4; font-size: 0.95rem; margin: 0; }

    [data-testid="stFileUploader"] {
        background-color: #0d1b2e;
        border: 2px dashed #1e4080;
        border-radius: 10px;
        padding: 12px;
    }

    .result-tumor {
        background: linear-gradient(135deg, #3b0a0a, #6b1414);
        border: 1px solid #c0392b; border-radius: 10px;
        padding: 20px 28px; text-align: center;
        font-size: 1.6rem; font-weight: 700;
        color: #ff6b6b; letter-spacing: 1px; margin: 12px 0;
    }
    .result-no-tumor {
        background: linear-gradient(135deg, #0a2e0a, #0f4f0f);
        border: 1px solid #27ae60; border-radius: 10px;
        padding: 20px 28px; text-align: center;
        font-size: 1.6rem; font-weight: 700;
        color: #55efc4; letter-spacing: 1px; margin: 12px 0;
    }

    .conf-row   { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
    .conf-label { width: 100px; font-size: 0.85rem; color: #8da9c4; }
    .conf-bar-bg {
        flex: 1; background-color: #1a2a40;
        border-radius: 6px; height: 14px; overflow: hidden;
    }
    .conf-bar-tumor    { height:100%; background: linear-gradient(90deg,#c0392b,#e74c3c); border-radius:6px; }
    .conf-bar-no-tumor { height:100%; background: linear-gradient(90deg,#1e8449,#27ae60); border-radius:6px; }
    .conf-pct { width:46px; text-align:right; font-size:0.85rem; color:#fff; font-weight:600; }

    .section-divider { border:none; border-top:1px solid #1a2f4e; margin:24px 0; }
    .img-caption { text-align:center; font-size:0.8rem; color:#8da9c4; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached — only loads once per session) ─────────────────────────
@st.cache_resource
def load_model():
    model = build_model(num_classes=config.NUM_CLASSES)
    model_path = os.path.join(config.OUTPUTS_DIR, "brain_tumor_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ── Preprocessing transforms ──────────────────────────────────────────────────
# Import from data_loader so the transforms here are always identical to those
# used during training — no risk of accidental mismatch if values ever change.
# get_transforms() returns (train_transform, val_test_transform); we want the second.
_, transform = get_transforms()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(image_pil):
    tensor = transform(image_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    return float(probs[0]), float(probs[1])   # (no_tumor_prob, tumor_prob)


# ── UI Layout ─────────────────────────────────────────────────────────────────
model = load_model()

st.markdown("""
<div class="header-banner">
    <div style="font-size:2.8rem;">🧠</div>
    <h1>Brain Tumor Classifier</h1>
    <p>Upload a 2D brain MRI image (JPG/PNG) for AI-assisted tumor detection</p>
</div>
""", unsafe_allow_html=True)

# File uploader — Streamlit only shows drag/drop by default (no webcam, no clipboard)
uploaded_file = st.file_uploader(
    "Drop an MRI image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    prob_no_tumor, prob_tumor = predict(image)
    label      = "Tumor" if prob_tumor >= 0.5 else "No Tumor"
    confidence = prob_tumor if label == "Tumor" else prob_no_tumor

    # Color-coded result
    if label == "Tumor":
        st.markdown(
            f'<div class="result-tumor">TUMOR DETECTED &nbsp;·&nbsp; {confidence*100:.1f}% confidence</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-no-tumor">NO TUMOR DETECTED &nbsp;·&nbsp; {confidence*100:.1f}% confidence</div>',
            unsafe_allow_html=True
        )

    # Confidence bars
    st.markdown(f"""
    <div class="conf-row">
        <span class="conf-label">Tumor</span>
        <div class="conf-bar-bg">
            <div class="conf-bar-tumor" style="width:{prob_tumor*100:.1f}%"></div>
        </div>
        <span class="conf-pct">{prob_tumor*100:.1f}%</span>
    </div>
    <div class="conf-row">
        <span class="conf-label">No Tumor</span>
        <div class="conf-bar-bg">
            <div class="conf-bar-no-tumor" style="width:{prob_no_tumor*100:.1f}%"></div>
        </div>
        <span class="conf-pct">{prob_no_tumor*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Uploaded image centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)
        st.markdown('<p class="img-caption">Uploaded MRI</p>', unsafe_allow_html=True)
