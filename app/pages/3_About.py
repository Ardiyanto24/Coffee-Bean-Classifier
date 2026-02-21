"""
3_About.py — Halaman informasi model dan project.

Menampilkan:
- Info model aktif (backbone, versi, training date)
- Metrics test set
- Penjelasan 4 kelas kopi
- Link GitHub
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui import (
    inject_global_css,
    render_class_catalog,
    render_info_box,
    render_metrics_card,
    render_model_info_card,
    render_page_header,
    render_sidebar_brand,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="About — Bean Classifier",
    page_icon="📋",
    layout="wide",
)
inject_global_css()

with st.sidebar:
    render_sidebar_brand()

# ── Header ─────────────────────────────────────────────────────────────────────
render_page_header(
    title="About",
    subtitle="Informasi model, metrics, dan penjelasan kelas biji kopi.",
)

# ── Load metadata ──────────────────────────────────────────────────────────────
def load_metadata() -> dict:
    """
    Load metadata.json dari model aktif berdasarkan registry.json.
    Return dict kosong jika file tidak ditemukan.
    """
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        active_key = registry.get("active_model", "")
        phase      = active_key.split("/")[0] if "/" in active_key else "baseline"
        meta_path  = ROOT / "models" / phase / "metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}

metadata = load_metadata()

# ── Layout: dua kolom ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    if not metadata:
        render_info_box(
            "metadata.json belum tersedia atau belum diisi. "
            "Lengkapi file setelah training selesai.",
            type="warning"
        )
    else:
        # Info model aktif
        predictor = st.session_state.get("predictor")
        if predictor:
            model_info = predictor.get_model_info()
        else:
            # Fallback: baca langsung dari metadata
            model_info = {
                "active_model": metadata.get("model_name", "—"),
                "backbone":     metadata.get("architecture", {}).get("backbone", "—"),
                "version":      metadata.get("version", "—"),
                "phase":        metadata.get("phase", "—"),
                "training":     metadata.get("training", {}),
            }
        render_model_info_card(model_info)

        # Architecture details
        arch = metadata.get("architecture", {})
        st.markdown(f"""
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <div style="
                font-size: 0.75rem;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                color: var(--cream-mute);
                margin-bottom: 14px;
                font-family: 'DM Sans', sans-serif;
            ">Architecture</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.85rem; display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--cream-mute);">Input Size</span>
                    <span style="color:var(--cream);">{arch.get('input_size', '—')}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--cream-mute);">Head</span>
                    <span style="color:var(--cream); text-align:right; max-width:60%;">{arch.get('head', '—')}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--cream-mute);">Backbone Frozen</span>
                    <span style="color:var(--cream);">{'Yes' if not arch.get('backbone_trainable', False) else 'No'}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--cream-mute);">Pretrained On</span>
                    <span style="color:var(--cream);">{arch.get('backbone_weights', 'imagenet').title()}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        render_metrics_card(metadata.get("metrics", {}))

with col_right:
    # Kelas kopi
    render_class_catalog()

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # Link GitHub
    st.markdown("""
    <div style="
        background: var(--roast-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 24px;
    ">
        <div style="
            font-size: 0.75rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cream-mute);
            margin-bottom: 14px;
            font-family: 'DM Sans', sans-serif;
        ">Project</div>
        <a href="https://github.com/Ardiyanto24/Coffee-Bean-Classifier"
           target="_blank"
           style="
               display: inline-flex;
               align-items: center;
               gap: 8px;
               background: rgba(200,146,42,0.12);
               border: 1px solid rgba(200,146,42,0.35);
               border-radius: 7px;
               padding: 10px 18px;
               font-family: 'DM Sans', sans-serif;
               font-size: 0.88rem;
               color: var(--gold);
               text-decoration: none;
               transition: background 0.2s;
           ">
            <span>⟨/⟩</span>
            <span>github.com/Ardiyanto24/Coffee-Bean-Classifier</span>
        </a>
        <div style="
            margin-top: 14px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.82rem;
            color: var(--cream-mute);
            line-height: 1.6;
        ">
            Pipeline: Preprocessing → Baseline Modeling → XAI (Phase 2) → Hyperparameter Tuning (Phase 3)
        </div>
    </div>
    """, unsafe_allow_html=True)