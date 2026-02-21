"""
1_Prediction.py — Halaman prediksi Coffee Bean Classifier.

Flow:
1. User upload gambar
2. Preview gambar ditampilkan
3. User klik tombol Classify
4. Hasil ditampilkan: badge kelas, confidence, probability bars, deskripsi kelas
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui import (
    CLASS_META,
    inject_global_css,
    render_class_description,
    render_info_box,
    render_page_header,
    render_prediction_badge,
    render_probability_bars,
    render_sidebar_brand,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction — Bean Classifier",
    page_icon="🔍",
    layout="wide",
)
inject_global_css()

with st.sidebar:
    render_sidebar_brand()

# ── Header ─────────────────────────────────────────────────────────────────────
render_page_header(
    title="Bean Classification",
    subtitle="Upload foto biji kopi untuk mendapatkan prediksi kualitas.",
)

# ── Cek predictor tersedia ─────────────────────────────────────────────────────
predictor = st.session_state.get("predictor")
if predictor is None:
    render_info_box(
        "Model belum tersedia. Pastikan file "
        "<code>models/baseline/EfficientNetB0.onnx</code> sudah ada "
        "dan <code>models/registry.json</code> sudah dikonfigurasi.",
        type="warning"
    )
    st.stop()

# ── Layout: dua kolom ──────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("""
    <div style="
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--cream-mute);
        margin-bottom: 12px;
        font-family: 'DM Sans', sans-serif;
    ">Upload Image</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(
            image,
            use_container_width=True,
            caption=f"{uploaded.name}  •  {image.size[0]}×{image.size[1]}px",
        )

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        classify_btn = st.button("✦ Classify", use_container_width=True)
    else:
        # Placeholder saat belum upload
        st.markdown("""
        <div style="
            background: var(--roast-card);
            border: 1px dashed var(--border);
            border-radius: 10px;
            height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
        ">
            <div style="font-size: 2.4rem; opacity: 0.4;">🫘</div>
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                color: var(--cream-mute);
                opacity: 0.6;
            ">Drag & drop atau klik untuk upload</div>
        </div>
        """, unsafe_allow_html=True)
        classify_btn = False

# ── Kolom kanan: hasil prediksi ────────────────────────────────────────────────
with col_result:
    st.markdown("""
    <div style="
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--cream-mute);
        margin-bottom: 12px;
        font-family: 'DM Sans', sans-serif;
    ">Classification Result</div>
    """, unsafe_allow_html=True)

    if not uploaded:
        # State awal — belum ada gambar
        st.markdown("""
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                color: var(--cream-mute);
                opacity: 0.5;
                text-align: center;
            ">Hasil prediksi akan muncul di sini<br>setelah gambar di-upload.</div>
        </div>
        """, unsafe_allow_html=True)

    elif classify_btn:
        # Jalankan prediksi
        with st.spinner("Menganalisis biji kopi..."):
            try:
                result = predictor.predict(image)

                # Badge utama
                render_prediction_badge(result["class"], result["confidence"])

                # Probability bars
                render_probability_bars(result["probabilities"])

                # Deskripsi kelas
                render_class_description(result["class"])

                # Simpan hasil ke session state untuk referensi
                st.session_state["last_result"] = result
                st.session_state["last_image"]  = image

            except Exception as e:
                render_info_box(
                    f"Prediksi gagal: {str(e)}",
                    type="error"
                )

    elif "last_result" in st.session_state:
        # Tampilkan hasil terakhir jika ada (user ganti gambar belum classify lagi)
        result = st.session_state["last_result"]
        render_prediction_badge(result["class"], result["confidence"])
        render_probability_bars(result["probabilities"])
        render_class_description(result["class"])
        render_info_box(
            "Ini adalah hasil dari gambar sebelumnya. Klik Classify untuk menganalisis gambar baru.",
            type="info"
        )

    else:
        # Gambar sudah upload tapi belum di-classify
        st.markdown("""
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.88rem;
                color: var(--cream-mute);
                opacity: 0.5;
                text-align: center;
            ">Klik tombol <strong style='color:var(--gold);'>Classify</strong><br>untuk memulai analisis.</div>
        </div>
        """, unsafe_allow_html=True)