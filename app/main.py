"""
main.py — Entry point Coffee Bean Classifier Streamlit App.

Tanggung jawab:
- Setup page config global
- Inisialisasi CoffeeBeanPredictor via st.cache_resource
  (model hanya di-load sekali, tidak reload saat navigasi antar halaman)
- Render sidebar brand + navigasi
- Render halaman utama (landing/home)
"""

import sys
from pathlib import Path

import streamlit as st

# Pastikan root project ada di sys.path agar src/ bisa di-import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui import inject_global_css, render_sidebar_brand

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bean Classifier",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
inject_global_css()

# ── Predictor — Load sekali, share ke semua halaman ───────────────────────────
@st.cache_resource
def load_predictor():
    """
    Load CoffeeBeanPredictor sekali dan cache hasilnya.

    st.cache_resource memastikan model ONNX hanya di-load satu kali
    per session server — tidak reload saat user navigasi antar halaman.
    Jika model belum ada, return None dan tampilkan warning.
    """
    from src.inference.predictor import CoffeeBeanPredictor
    registry_path = ROOT / "models" / "registry.json"
    try:
        predictor = CoffeeBeanPredictor(str(registry_path))
        return predictor
    except FileNotFoundError as e:
        return None

# Simpan predictor di session state agar bisa diakses semua halaman
if "predictor" not in st.session_state:
    st.session_state.predictor = load_predictor()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()

# ── Home Page ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    max-width: 640px;
    margin: 60px auto 0 auto;
    text-align: center;
    padding: 0 20px;
">
    <div style="font-size: 4rem; margin-bottom: 16px;">☕</div>
    <h1 style="
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--cream);
        line-height: 1.2;
        margin-bottom: 16px;
    ">Coffee Bean<br>Quality Classifier</h1>
    <p style="
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        color: var(--cream-mute);
        line-height: 1.7;
        margin-bottom: 36px;
    ">
        Klasifikasi otomatis kualitas biji kopi menggunakan
        deep learning. Upload foto biji kopi dan dapatkan
        prediksi kelas beserta tingkat kepercayaan model.
    </p>
    <div style="
        display: flex;
        gap: 12px;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 48px;
    ">
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            color: var(--cream-mute);
        ">⚠️ Defect</div>
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            color: var(--cream-mute);
        ">🫘 Longberry</div>
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            color: var(--cream-mute);
        ">🔵 Peaberry</div>
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            color: var(--cream-mute);
        ">⭐ Premium</div>
    </div>
    <p style="
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--cream-mute);
        opacity: 0.6;
    ">
        Gunakan menu di sidebar untuk mulai →
    </p>
</div>
""", unsafe_allow_html=True)