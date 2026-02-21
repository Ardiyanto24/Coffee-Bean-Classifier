"""
ui.py — Reusable UI components untuk Coffee Bean Classifier Streamlit app.

Semua fungsi di sini menghasilkan HTML/CSS yang dirender via st.markdown().
Dipakai oleh semua halaman di app/pages/.
"""

import streamlit as st

# ── Konstanta ──────────────────────────────────────────────────────────────────

CLASS_META = {
    "defect": {
        "color": "#e05c5c",
        "emoji": "⚠️",
        "label": "Defect",
        "description": (
            "Biji kopi mengalami kerusakan fisik atau cacat pada proses pengolahan. "
            "Cacat dapat berupa biji hitam, biji asam, biji pecah, atau kontaminasi "
            "serangga. Kualitas cup score rendah dan umumnya ditolak untuk specialty coffee."
        ),
    },
    "longberry": {
        "color": "#c8922a",
        "emoji": "🫘",
        "label": "Longberry",
        "description": (
            "Varietas Arabika dengan biji memanjang — kira-kira dua kali ukuran biji standar. "
            "Berasal dari Ethiopia, dikenal dengan profil rasa yang kompleks: "
            "floral, bright acidity, dan body yang medium. Termasuk specialty grade."
        ),
    },
    "peaberry": {
        "color": "#4a9edd",
        "emoji": "🔵",
        "label": "Peaberry",
        "description": (
            "Anomali alami di mana hanya satu biji berkembang dalam cherry, "
            "menghasilkan bentuk bulat seperti kelereng. Peaberry menerima seluruh "
            "nutrisi cherry sendirian, menghasilkan rasa lebih concentrated dan "
            "bright. Dipercaya memiliki kualitas lebih tinggi dari biji flat biasa."
        ),
    },
    "premium": {
        "color": "#4caf82",
        "emoji": "⭐",
        "label": "Premium",
        "description": (
            "Biji kopi kualitas terbaik — ukuran seragam, tidak ada cacat fisik, "
            "moisture content optimal, dan density tinggi. Memenuhi standar specialty "
            "coffee (SCA score ≥ 80). Cocok untuk single origin dan espresso premium."
        ),
    },
}

# ── Global CSS ─────────────────────────────────────────────────────────────────

def inject_global_css():
    """
    Inject CSS global — tema Artisan Roastery.
    Dipanggil sekali di main.py via setiap halaman.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --espresso:   #1a1209;
        --roast-dark: #241a0d;
        --roast-mid:  #2e2010;
        --roast-card: #332415;
        --gold:       #c8922a;
        --gold-light: #e0aa45;
        --cream:      #f5ead0;
        --cream-mute: #b8a88a;
        --red:        #e05c5c;
        --blue:       #4a9edd;
        --green:      #4caf82;
        --border:     rgba(200, 146, 42, 0.2);
    }

    /* ── Base ── */
    .stApp {
        background-color: var(--espresso) !important;
        color: var(--cream) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hide default streamlit header & footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--roast-dark) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--cream) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 6px !important;
        padding: 6px 12px !important;
        transition: background 0.2s !important;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: rgba(200, 146, 42, 0.15) !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: var(--cream) !important;
    }
    p, li, span, div {
        color: var(--cream) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--gold) !important;
        color: var(--espresso) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 28px !important;
        letter-spacing: 0.5px !important;
        transition: background 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover {
        background: var(--gold-light) !important;
        transform: translateY(-1px) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--roast-card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--border) !important;
        margin: 24px 0 !important;
    }

    /* ── Metric ── */
    [data-testid="stMetric"] {
        background: var(--roast-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--gold) !important;
        font-family: 'Playfair Display', serif !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Header Components ──────────────────────────────────────────────────────────

def render_page_header(title: str, subtitle: str = ""):
    """
    Render header halaman dengan gaya Artisan Roastery.

    Parameters
    ----------
    title : str
        Judul halaman — dirender dengan Playfair Display.
    subtitle : str
        Subjudul opsional — dirender lebih kecil dan muted.
    """
    subtitle_html = f'<p style="font-family:\'DM Sans\',sans-serif; color:var(--cream-mute); font-size:1rem; margin:0 0 24px 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <h1 style="
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--cream);
            margin: 0 0 4px 0;
            line-height: 1.2;
        ">{title}</h1>
        {subtitle_html}
    </div>
    <hr style="border:none; border-top:1px solid var(--border); margin: 0 0 28px 0;">
    """, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render brand identity di sidebar."""
    st.markdown("""
    <div style="padding: 20px 8px 12px 8px; text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 6px;">☕</div>
        <div style="
            font-family: 'Playfair Display', serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--gold);
            letter-spacing: 0.5px;
        ">Bean Classifier</div>
        <div style="
            font-size: 0.72rem;
            color: var(--cream-mute);
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-top: 2px;
        ">Quality Inspection</div>
    </div>
    <hr style="border:none; border-top:1px solid var(--border); margin: 0 0 12px 0;">
    """, unsafe_allow_html=True)


# ── Prediction Result Components ──────────────────────────────────────────────

def render_prediction_badge(class_name: str, confidence: float):
    """
    Render badge hasil prediksi utama — nama kelas + confidence besar.

    Parameters
    ----------
    class_name : str
        Nama kelas hasil prediksi (defect/longberry/peaberry/premium).
    confidence : float
        Nilai confidence 0.0–1.0.
    """
    meta  = CLASS_META.get(class_name, {})
    color = meta.get("color", "#c8922a")
    emoji = meta.get("emoji", "🫘")
    label = meta.get("label", class_name.title())
    pct   = f"{confidence * 100:.1f}%"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}18, {color}08);
        border: 1px solid {color}55;
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 20px;
    ">
        <div style="font-size: 2.8rem; line-height:1;">{emoji}</div>
        <div>
            <div style="
                font-family: 'Playfair Display', serif;
                font-size: 1.8rem;
                font-weight: 700;
                color: {color};
                line-height: 1.1;
            ">{label}</div>
            <div style="
                font-size: 0.85rem;
                color: var(--cream-mute);
                margin-top: 4px;
                font-family: 'DM Sans', sans-serif;
                letter-spacing: 0.3px;
            ">Confidence: <span style="color:{color}; font-weight:600;">{pct}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(probabilities: dict):
    """
    Render probability bar untuk semua kelas.

    Parameters
    ----------
    probabilities : dict
        Dict {class_name: probability} dari predictor.predict().
    """
    st.markdown("""
    <div style="
        background: var(--roast-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 24px;
        margin: 8px 0;
    ">
        <div style="
            font-size: 0.75rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cream-mute);
            margin-bottom: 16px;
            font-family: 'DM Sans', sans-serif;
        ">Probability Distribution</div>
    """, unsafe_allow_html=True)

    # Sort by probability descending
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_probs:
        meta  = CLASS_META.get(class_name, {})
        color = meta.get("color", "#c8922a")
        label = meta.get("label", class_name.title())
        pct   = prob * 100

        st.markdown(f"""
        <div style="margin-bottom: 14px;">
            <div style="
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            ">
                <span style="
                    font-size: 0.88rem;
                    color: var(--cream);
                    font-family: 'DM Sans', sans-serif;
                ">{label}</span>
                <span style="
                    font-size: 0.88rem;
                    font-weight: 600;
                    color: {color};
                    font-family: 'DM Sans', sans-serif;
                ">{pct:.1f}%</span>
            </div>
            <div style="
                background: rgba(255,255,255,0.06);
                border-radius: 99px;
                height: 7px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}aa, {color});
                    border-radius: 99px;
                    transition: width 0.6s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_class_description(class_name: str):
    """
    Render kartu deskripsi kelas hasil prediksi.

    Parameters
    ----------
    class_name : str
        Nama kelas (defect/longberry/peaberry/premium).
    """
    meta  = CLASS_META.get(class_name, {})
    color = meta.get("color", "#c8922a")
    label = meta.get("label", class_name.title())
    desc  = meta.get("description", "")
    emoji = meta.get("emoji", "🫘")

    st.markdown(f"""
    <div style="
        background: var(--roast-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 24px;
        margin: 8px 0;
    ">
        <div style="
            font-size: 0.75rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cream-mute);
            margin-bottom: 12px;
            font-family: 'DM Sans', sans-serif;
        ">About this class</div>
        <div style="
            display: flex;
            align-items: flex-start;
            gap: 14px;
        ">
            <div style="font-size:1.6rem; margin-top:2px;">{emoji}</div>
            <div>
                <div style="
                    font-family: 'Playfair Display', serif;
                    font-size: 1.05rem;
                    font-weight: 600;
                    color: {color};
                    margin-bottom: 6px;
                ">{label}</div>
                <div style="
                    font-size: 0.88rem;
                    line-height: 1.65;
                    color: var(--cream-mute);
                    font-family: 'DM Sans', sans-serif;
                ">{desc}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── About Page Components ──────────────────────────────────────────────────────

def render_model_info_card(model_info: dict):
    """
    Render kartu info model aktif.

    Parameters
    ----------
    model_info : dict
        Output dari predictor.get_model_info().
    """
    backbone      = model_info.get("backbone", "—")
    version       = model_info.get("version", "—")
    phase         = model_info.get("phase", "—").title()
    training_date = model_info.get("training", {}).get("training_date") or "—"
    active        = model_info.get("active_model", "—")

    st.markdown(f"""
    <div style="
        background: var(--roast-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--gold);
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
        ">Active Model</div>
        <div style="
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--gold);
            margin-bottom: 14px;
        ">{backbone}</div>
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
        ">
            <div>
                <span style="color:var(--cream-mute);">Phase</span><br>
                <span style="color:var(--cream); font-weight:500;">{phase}</span>
            </div>
            <div>
                <span style="color:var(--cream-mute);">Version</span><br>
                <span style="color:var(--cream); font-weight:500;">{version}</span>
            </div>
            <div>
                <span style="color:var(--cream-mute);">Training Date</span><br>
                <span style="color:var(--cream); font-weight:500;">{training_date}</span>
            </div>
            <div>
                <span style="color:var(--cream-mute);">Registry Key</span><br>
                <span style="color:var(--cream); font-weight:500;">{active}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_card(metrics: dict):
    """
    Render kartu metrics model.

    Parameters
    ----------
    metrics : dict
        Dict metrics dari metadata.json["metrics"].
    """
    def fmt(v):
        if v is None:
            return "<span style='color:var(--cream-mute);'>—</span>"
        return f"<span style='color:var(--gold);font-weight:600;'>{v:.4f}</span>"

    overall = [
        ("Accuracy",        metrics.get("test_accuracy")),
        ("F1 Macro",        metrics.get("test_f1_macro")),
        ("Precision Macro", metrics.get("test_precision_macro")),
        ("Recall Macro",    metrics.get("test_recall_macro")),
        ("ROC-AUC Macro",   metrics.get("test_roc_auc_macro")),
    ]

    rows_html = "".join(f"""
        <div style="
            display:flex;
            justify-content:space-between;
            padding: 9px 0;
            border-bottom: 1px solid var(--border);
            font-family:'DM Sans',sans-serif;
            font-size:0.87rem;
        ">
            <span style="color:var(--cream-mute);">{name}</span>
            <span>{fmt(val)}</span>
        </div>
    """ for name, val in overall)

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
        ">Test Set Metrics</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


def render_class_catalog():
    """Render katalog 4 kelas kopi di halaman About."""
    st.markdown("""
    <div style="
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--cream-mute);
        margin: 24px 0 14px 0;
        font-family: 'DM Sans', sans-serif;
    ">Coffee Bean Classes</div>
    """, unsafe_allow_html=True)

    for class_name, meta in CLASS_META.items():
        color = meta["color"]
        emoji = meta["emoji"]
        label = meta["label"]
        desc  = meta["description"]
        st.markdown(f"""
        <div style="
            background: var(--roast-card);
            border: 1px solid var(--border);
            border-left: 3px solid {color};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 10px;
            display: flex;
            gap: 14px;
            align-items: flex-start;
        ">
            <div style="font-size:1.5rem;margin-top:2px;">{emoji}</div>
            <div>
                <div style="
                    font-family:'Playfair Display',serif;
                    font-size:1rem;
                    font-weight:600;
                    color:{color};
                    margin-bottom:5px;
                ">{label}</div>
                <div style="
                    font-size:0.85rem;
                    line-height:1.6;
                    color:var(--cream-mute);
                    font-family:'DM Sans',sans-serif;
                ">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Utility Components ─────────────────────────────────────────────────────────

def render_info_box(message: str, type: str = "info"):
    """
    Render kotak info/warning/error dengan styling konsisten.

    Parameters
    ----------
    message : str
        Pesan yang ditampilkan.
    type : str
        "info" | "warning" | "error"
    """
    colors = {
        "info":    ("#4a9edd", "ℹ️"),
        "warning": ("#c8922a", "⚠️"),
        "error":   ("#e05c5c", "❌"),
    }
    color, emoji = colors.get(type, colors["info"])
    st.markdown(f"""
    <div style="
        background: {color}12;
        border: 1px solid {color}44;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        color: var(--cream-mute);
        margin: 8px 0;
    ">{emoji} {message}</div>
    """, unsafe_allow_html=True)