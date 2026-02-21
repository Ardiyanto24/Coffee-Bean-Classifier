"""
2_Explainability.py — Halaman XAI placeholder (Phase 2).

Menampilkan pesan "Coming Soon" dengan penjelasan singkat
tentang apa yang akan ada di Phase 2.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui import (
    inject_global_css,
    render_page_header,
    render_sidebar_brand,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainability — Bean Classifier",
    page_icon="🔬",
    layout="wide",
)
inject_global_css()

with st.sidebar:
    render_sidebar_brand()

# ── Header ─────────────────────────────────────────────────────────────────────
render_page_header(
    title="Explainability",
    subtitle="Memahami alasan di balik setiap prediksi model.",
)

# ── Coming Soon ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    max-width: 560px;
    margin: 40px auto;
    text-align: center;
">
    <div style="
        font-size: 3.5rem;
        margin-bottom: 20px;
        opacity: 0.6;
    ">🔬</div>
    <h2 style="
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: var(--cream);
        margin-bottom: 12px;
    ">Coming Soon</h2>
    <p style="
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: var(--cream-mute);
        line-height: 1.75;
        margin-bottom: 36px;
    ">
        Halaman ini akan hadir di <strong style='color:var(--gold);'>Phase 2</strong>
        — menampilkan visualisasi XAI (Explainable AI) yang membantu
        memahami bagian mana dari gambar yang paling berpengaruh
        terhadap keputusan model.
    </p>

    <div style="
        background: var(--roast-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        text-align: left;
    ">
        <div style="
            font-size: 0.72rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cream-mute);
            margin-bottom: 16px;
            font-family: 'DM Sans', sans-serif;
        ">Planned for Phase 2</div>

        <div style="display:flex; flex-direction:column; gap:12px;">
            <div style="display:flex; align-items:flex-start; gap:12px;">
                <div style="
                    background: rgba(200,146,42,0.15);
                    color: var(--gold);
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-size: 0.78rem;
                    font-family: 'DM Sans', sans-serif;
                    white-space: nowrap;
                ">Grad-CAM</div>
                <div style="
                    font-size: 0.85rem;
                    color: var(--cream-mute);
                    font-family: 'DM Sans', sans-serif;
                    line-height: 1.5;
                ">Heatmap aktivasi — wilayah gambar yang paling diperhatikan model saat membuat prediksi.</div>
            </div>
            <div style="display:flex; align-items:flex-start; gap:12px;">
                <div style="
                    background: rgba(200,146,42,0.15);
                    color: var(--gold);
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-size: 0.78rem;
                    font-family: 'DM Sans', sans-serif;
                    white-space: nowrap;
                ">LIME</div>
                <div style="
                    font-size: 0.85rem;
                    color: var(--cream-mute);
                    font-family: 'DM Sans', sans-serif;
                    line-height: 1.5;
                ">Segmentasi superpixel — area mana yang mendorong atau menurunkan confidence prediksi.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)