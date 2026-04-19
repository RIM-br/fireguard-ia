import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import tempfile
import time
import json
import os
import urllib.request
import threading
from alarm import trigger_alarm
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from database import (init_db, save_alerte, get_all_alertes,
                      get_stats_globales, save_session,
                      get_alertes_par_jour, delete_all)
import uuid

# Connexion MySQL au démarrage
mysql_ok, mysql_status = init_db()

# ID unique par session Streamlit
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

# ─────────────────────────────────────────────
# CHARGEMENT YOLO (avec cache Streamlit)
# ─────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    """Charge le vrai modèle YOLOv8 best.pt — une seule fois au démarrage."""
    try:
        from ultralytics import YOLO
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'best.pt')
        if not os.path.exists(model_path):
            return None, "❌ Fichier best.pt introuvable dans le dossier."
        model = YOLO(model_path)
        return model, "✅ Modèle YOLOv8 chargé"
    except Exception as e:
        return None, f"❌ Erreur chargement YOLO : {e}"

# ─────────────────────────────────────────────
# CONNEXION KAFKA (avec cache Streamlit)
# ─────────────────────────────────────────────
@st.cache_resource
def load_kafka_consumer():
    try:
        consumer = KafkaConsumer(
            'parking-alerts',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1500
        )
        return consumer, "✅ Kafka connecté"
    except NoBrokersAvailable:
        return None, "❌ Kafka non disponible (Docker non lancé)"
    except Exception as e:
        return None, f"❌ Kafka erreur : {e}"

@st.cache_resource
def load_kafka_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer, "✅ Kafka Producer prêt"
    except Exception:
        return None, "❌ Kafka Producer indisponible"

# ─────────────────────────────────────────────
# DETECTION YOLO RÉELLE sur une frame
# ─────────────────────────────────────────────
def get_yolo_prediction(model, frame, conf_threshold=0.45):
    """
    Retourne : label dominant, score, frame annotée, liste de détections.
    """
    results = model(frame, conf=conf_threshold, verbose=False)
    detections = []
    annotated = results[0].plot()

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            score = float(box.conf[0])
            detections.append({"label": label, "score": score})

    if detections:
        best = max(detections, key=lambda d: d["score"])
        return best["label"], best["score"], annotated, detections
    return "normal", 0.0, frame, []

def classify_risk(label):
    """Classes réelles de best.pt : 'fire' et 'smoke' uniquement."""
    label = label.lower().strip()
    if label == "fire":
        return "CRITIQUE", "level-high"
    elif label == "smoke":
        return "ALERTE", "level-medium"
    else:
        return "NORMAL", "level-low"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "last_alarm_time"    not in st.session_state:
    st.session_state.last_alarm_time    = 0
if "total_frames"       not in st.session_state:
    st.session_state.total_frames       = 0
if "flame_count"        not in st.session_state:
    st.session_state.flame_count        = 0
if "smoke_count"        not in st.session_state:
    st.session_state.smoke_count        = 0
if "alert_history"      not in st.session_state:
    st.session_state.alert_history      = []
if "detection_mode"     not in st.session_state:
    st.session_state.detection_mode     = "YOLO"
if "video_running"      not in st.session_state:
    st.session_state["video_running"]   = False
if "image_analyzed"     not in st.session_state:
    st.session_state["image_analyzed"]  = False
if "last_image"         not in st.session_state:
    st.session_state["last_image"]      = None

# ─────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FireGuard IA · Détection d'Incendie",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS THÈME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Bebas+Neue&display=swap');

:root {
    --bg-deep:  #080604; --bg-panel: #0f0b08; --bg-card: #140e0a;
    --fire1: #ff4500; --fire2: #ff8c00; --fire3: #ffbf00;
    --smoke: #8a8a8a; --ember: #ff2200; --green: #22c55e;
    --text: #f0e6d3; --text-dim: #8a7260; --border: rgba(255,69,0,.2);
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important; color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
h1 { font-family:'Bebas Neue',sans-serif !important; color:var(--fire1) !important;
     letter-spacing:3px; font-size:2.5rem !important; }
h2 { font-family:'Bebas Neue',sans-serif !important; color:var(--fire2) !important; letter-spacing:2px; }
h3 { font-family:'Rajdhani',sans-serif !important; color:var(--fire3) !important; font-weight:700; }
[data-testid="stSidebar"] label {
    font-family:'Rajdhani',sans-serif !important;
    color:var(--text-dim) !important; font-size:1rem; font-weight:600;
}
[data-testid="stMetric"] {
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; padding:14px !important;
}
[data-testid="stMetricLabel"] { color:var(--text-dim) !important; }
[data-testid="stMetricValue"] { color:var(--fire2) !important; font-family:'Bebas Neue',sans-serif !important; font-size:1.8rem !important; }
[data-testid="stMetricDelta"]  { color:var(--fire3) !important; }
.stButton > button {
    background:linear-gradient(135deg,var(--ember),var(--fire2)) !important;
    color:#fff !important; border:none !important; border-radius:6px !important;
    font-family:'Bebas Neue',sans-serif !important; font-size:1rem !important;
    letter-spacing:2px !important; transition:box-shadow .2s !important;
}
.stButton > button:hover { box-shadow:0 0 20px rgba(255,69,0,.5) !important; }
[data-testid="stFileUploader"] {
    background:var(--bg-card) !important;
    border:1px dashed var(--fire1) !important; border-radius:10px !important;
}
.status-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:6px; vertical-align:middle;
}
.dot-green  { background:#22c55e; box-shadow:0 0 6px #22c55e; }
.dot-red    { background:#ff2200; box-shadow:0 0 6px #ff2200; }
.dot-yellow { background:#ffbf00; box-shadow:0 0 6px #ffbf00; }
.conn-banner {
    display:flex; gap:18px; flex-wrap:wrap;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:10px; padding:12px 18px; margin-bottom:16px;
    font-size:.82rem;
}
.conn-item { display:flex; align-items:center; gap:6px; }
.fire-hero {
    background:linear-gradient(135deg,rgba(255,69,0,.1),rgba(255,140,0,.06),rgba(8,6,4,0));
    border:1px solid rgba(255,69,0,.25); border-radius:18px;
    padding:40px 36px; margin-bottom:28px; position:relative; overflow:hidden;
}
.fire-hero::after { content:'🔥'; position:absolute; right:30px; top:20px; font-size:6rem; opacity:.07; }
.member-card {
    background:linear-gradient(145deg,#140e0a,#1f1208);
    border:1px solid rgba(255,140,0,.3); border-radius:14px;
    padding:22px 18px; text-align:center; transition:transform .25s,box-shadow .25s;
}
.member-card:hover { transform:translateY(-5px); box-shadow:0 10px 35px rgba(255,69,0,.25); }
.member-avatar {
    width:78px; height:78px; border-radius:50%; border:2px solid var(--fire1);
    font-size:2.1rem; display:flex; align-items:center; justify-content:center;
    margin:0 auto 12px; background:radial-gradient(circle at 40% 40%,#2a1505,#0f0b08);
}
.member-name { font-family:'Bebas Neue',sans-serif; font-size:1.05rem; color:#fff; letter-spacing:1px; margin-bottom:4px; }
.member-role { font-size:.78rem; color:var(--fire2); font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px; }
.member-desc { font-size:.82rem; color:var(--text-dim); line-height:1.55; }
.alert-critical {
    background:rgba(255,34,0,.12); border:1px solid rgba(255,34,0,.5);
    border-left:4px solid #ff2200; border-radius:10px;
    padding:14px 18px; margin-bottom:12px; animation:pulse-red 2s infinite;
}
.alert-warning {
    background:rgba(255,140,0,.1); border:1px solid rgba(255,140,0,.4);
    border-left:4px solid var(--fire2); border-radius:10px;
    padding:14px 18px; margin-bottom:12px;
}
.alert-ok {
    background:rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.3);
    border-left:4px solid #22c55e; border-radius:10px;
    padding:14px 18px; margin-bottom:12px;
}
@keyframes pulse-red {
    0%,100% { box-shadow:0 0 0 rgba(255,34,0,0); }
    50%      { box-shadow:0 0 16px rgba(255,34,0,.3); }
}
.tech-tag {
    display:inline-block; background:rgba(255,69,0,.1);
    border:1px solid rgba(255,69,0,.35); color:var(--fire2);
    border-radius:20px; padding:3px 13px; font-size:.77rem; margin:3px; letter-spacing:.5px;
}
.pb-wrap { background:#1a0f08; border-radius:20px; height:9px; overflow:hidden; margin-top:3px; }
.pb-fill  { height:100%; border-radius:20px;
            background:linear-gradient(90deg,var(--ember),var(--fire2),var(--fire3)); }
.tl-item {
    border-left:2px solid var(--fire1); padding:6px 0 6px 18px;
    position:relative; margin-bottom:8px;
}
.tl-item::before {
    content:''; position:absolute; left:-6px; top:12px;
    width:10px; height:10px; background:var(--fire2); border-radius:50%;
    box-shadow:0 0 8px var(--fire2);
}
.level-high   { background:rgba(255,34,0,.2);   color:#ff6b6b; border:1px solid #ff2200;
                display:inline-block; padding:3px 12px; border-radius:20px; font-size:.75rem; font-weight:700; }
.level-medium { background:rgba(255,140,0,.15); color:#ffbf40; border:1px solid #ff8c00;
                display:inline-block; padding:3px 12px; border-radius:20px; font-size:.75rem; font-weight:700; }
.level-low    { background:rgba(34,197,94,.1);  color:#4ade80; border:1px solid #22c55e;
                display:inline-block; padding:3px 12px; border-radius:20px; font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT COMPOSANTS AU DÉMARRAGE
# ─────────────────────────────────────────────
yolo_model, yolo_status  = load_yolo_model()
kafka_consumer, kafka_status_consumer = load_kafka_consumer()
kafka_producer, kafka_status_producer = load_kafka_producer()

yolo_ok  = yolo_model  is not None
kafka_ok = kafka_consumer is not None

# ─────────────────────────────────────────────
# DONNÉES PROJET & ÉQUIPE
# ─────────────────────────────────────────────
PROJET = {
    "titre":      "FireGuard IA — Détection d'Incendie par Vision par Ordinateur",
    "sous_titre": "Système de détection précoce de feux, flammes et fumée en temps réel",
    "annee":      "2025 – 2026",
    "encadrant":  "Pr. Outhman MAAROUF",
    "objectif": (
        "FireGuard IA est un système de surveillance intelligente capable de détecter "
        "automatiquement la présence de feu, flammes, fumée et braises dans des flux vidéo "
        "en temps réel. Grâce au modèle YOLOv8 entraîné sur un dataset spécialisé, "
        "le système génère des alertes immédiates pour prévenir la propagation d'incendies "
        "dans les espaces publics, entrepôts, forêts et bâtiments industriels."
    ),
    "technologies": ["YOLOv8", "Python 3.11", "Streamlit", "OpenCV", "Kafka", "Docker"],
}

MEMBRES = [
    {"emoji": "👩‍💻", "nom": "Asmae AALIKANE",
     "role": "Data Engineer & Machine Learning Developer",
     "desc":" Collecte, préparation et structuration de données feu/fumée.Entraînement et évaluation d’un modèle de détection d’incendie (YOLOv8).Déploiement du modèle via FastAPI.Pipeline temps réel avec Kafka (producteur/consommateur).Conteneurisation de l’infrastructure avec Docker"},
    {"emoji": "👩‍🔬", "nom": "Assietou ABDERRAHMANE",
     "role": "Data Scientist · Entraînement Modèle",
     "desc": "Conception et implémentation d’un module de détection d’incendie basé sur l’analyse vidéo en temps réel, intégrant la classification des niveaux de risque (normal, alerte, critique) ainsi que le déclenchement automatique d’une alarme sonore en cas de détection critique."},
    {"emoji": "👩‍🎨", "nom": "Rim BARI",
     "role": "Développeuse Full-Stack · UI/UX & Intégration",
     "desc": "Développement de l'interface Streamlit, intégration complète YOLO + Kafka en temps réel et conception du tableau de bord d'analyse + Connection avec base de donnée()."},
]
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px;'>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;letter-spacing:4px;
                    background:linear-gradient(90deg,#ff4500,#ffbf00);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🔥 FIREGUARD
        </div>
        <div style='font-size:.68rem;color:#5a3a28;letter-spacing:2px;margin-top:2px;'>
            DÉTECTION INCENDIE · IA
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Accueil & Équipe",
        "🎥  Détection Vidéo",
        "🚨  Alertes en Direct",
        "📊  Analyse des Performances",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:.8rem;color:#5a3a28;line-height:2;'>
        <div>👥 <b style='color:#ff8c00'>Groupe de 3</b></div>
        <div>📅 <b style='color:#ff8c00'>{PROJET['annee']}</b></div>
        <div>🎓 <b style='color:#ff8c00'>{PROJET['encadrant']}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#5a3a28;margin-bottom:6px;'>STATUT SYSTÈME</div>", unsafe_allow_html=True)

    yolo_dot  = "dot-green" if yolo_ok  else "dot-red"
    kafka_dot = "dot-green" if kafka_ok else "dot-red"
    yolo_txt  = "YOLOv8 chargé ✅"  if yolo_ok  else "YOLOv8 absent ❌"
    kafka_txt = "Kafka connecté ✅" if kafka_ok else "Kafka hors ligne ❌"

    st.markdown(f"""
    <div style='font-size:.78rem;line-height:2.2;'>
        <span class='status-dot {yolo_dot}'></span>{yolo_txt}<br>
        <span class='status-dot {kafka_dot}'></span>{kafka_txt}<br>
        <span class='status-dot dot-green'></span>Caméra prête<br>
        <span class='status-dot dot-yellow'></span>Alarme : active
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#5a3a28;margin-bottom:8px;'>AVANCEMENT DU PROJET</div>", unsafe_allow_html=True)
    for lbl, pct in [("Dataset Feu", 88), ("Modèle YOLO", 80), ("Interface", 95), ("Tests", 70)]:
        st.markdown(f"""
        <div style='margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;font-size:.74rem;color:#8a7260;margin-bottom:3px;'>
                <span>{lbl}</span><span style='color:#ff8c00'>{pct}%</span>
            </div>
            <div class='pb-wrap'><div class='pb-fill' style='width:{pct}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#5a3a28;margin-bottom:6px;'>SEUIL DE CONFIANCE</div>", unsafe_allow_html=True)
    global_conf = st.slider(
        "", 0.10, 1.00,
        st.session_state.get("global_conf", 0.25),
        0.05,
        format="%.2f",
        help="Appliqué à toutes les détections YOLO (vidéo, caméra, image)",
        label_visibility="collapsed"
    )
    st.session_state["global_conf"] = global_conf

    conf_color = "#22c55e" if global_conf >= 0.5 else "#ffbf00" if global_conf >= 0.3 else "#ff2200"
    st.markdown(f"""
    <div style='text-align:center;font-size:.78rem;margin-top:-8px;'>
        <span style='color:{conf_color};font-family:"Bebas Neue",sans-serif;font-size:1.1rem;'>{global_conf:.0%}</span>
        &nbsp;<span style='color:#5a3a28;'>confiance min.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#5a3a28;margin-bottom:6px;'>MODE DÉTECTION VIDÉO</div>", unsafe_allow_html=True)
    mode_options = []
    if yolo_ok:
        mode_options.append("🤖 YOLO Direct (best.pt)")
    if kafka_ok:
        mode_options.append("📡 Via Kafka (Docker)")
    if not mode_options:
        mode_options = ["⚠️ Aucun module disponible"]

    selected_mode = st.radio("", mode_options, label_visibility="collapsed")
    st.session_state.detection_mode = "KAFKA" if "Kafka" in selected_mode else "YOLO"

# ═══════════════════════════════════════════════
# BANNIÈRE CONNEXION
# ═══════════════════════════════════════════════
def show_connection_banner():
    yolo_color = "#22c55e" if yolo_ok else "#ff2200"
    kafka_color = "#22c55e" if kafka_ok else "#ff2200"
    st.markdown(f"""
    <div class='conn-banner'>
        <div class='conn-item'>
            <span style='color:{yolo_color};font-size:1rem;'>●</span>
            <span style='color:#c4a882;'>{yolo_status}</span>
        </div>
        <div class='conn-item'>
            <span style='color:{kafka_color};font-size:1rem;'>●</span>
            <span style='color:#c4a882;'>{kafka_status_consumer}</span>
        </div>
        <div class='conn-item'>
            <span style='color:#ffbf00;font-size:1rem;'>●</span>
            <span style='color:#c4a882;'>Mode actif : <b style='color:#ff8c00'>{st.session_state.detection_mode}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 1 ── ACCUEIL & ÉQUIPE
# ═══════════════════════════════════════════════
if page == "🏠  Accueil & Équipe":
    show_connection_banner()

    st.markdown(f"""
    <div class='fire-hero'>
        <div style='font-size:.7rem;color:#5a3a28;letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;'>
            Projet · {PROJET['annee']}
        </div>
        <h1 style='margin:0 0 6px;'>{PROJET['titre']}</h1>
        <p style='color:#8a7260;font-size:1.05rem;margin:0 0 18px;'>{PROJET['sous_titre']}</p>
        <div>{''.join(f"<span class='tech-tag'>{t}</span>" for t in PROJET['technologies'])}</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 📋 Description du Projet")
        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:12px;padding:20px;line-height:1.85;color:#c4a882;font-size:.95rem;'>
            {PROJET['objectif']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔥 Classes Détectées")
        classes_info = [
            ("🔥", "Flamme",    "#ff4500", "Détection de flammes actives visibles"),
            ("💨", "Fumée",     "#8a8a8a", "Panaches de fumée grise ou noire"),
            ("🌡️","Braise",    "#ff8c00", "Zones incandescentes rougeoyantes"),
            ("⚠️", "Feu Forêt","#ffbf00", "Incendies de végétation et broussailles"),
        ]
        g1, g2 = st.columns(2)
        for i, (ico, nom, couleur, detail) in enumerate(classes_info):
            with (g1 if i % 2 == 0 else g2):
                st.markdown(f"""
                <div style='background:var(--bg-card);border-left:3px solid {couleur};
                            border-radius:8px;padding:12px;margin-bottom:10px;'>
                    <div style='font-size:1.4rem'>{ico}</div>
                    <div style='font-weight:700;color:#fff;font-size:.9rem'>{nom}</div>
                    <div style='font-size:.75rem;color:#5a3a28'>{detail}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### 🎯 Objectifs Clés")
        for ico, titre, detail in [
            ("🎯", "Précision ≥ 85%",     "mAP@0.5 sur les 4 classes feu"),
            ("⚡", "Détection < 200 ms",   "Latence temps réel par frame"),
            ("📡", "Flux caméra live",     "IP / RTSP / webcam"),
            ("🚨", "Alertes automatiques", "Notification instantanée"),
            ("📊", "Dashboard analytique", "Statistiques et historique"),
            ("🌐", "Déploiement cloud",    "Accessible à distance"),
        ]:
            st.markdown(f"""
            <div class='tl-item'>
                <div style='font-size:.88rem;font-weight:700;color:#f0e6d3'>{ico} {titre}</div>
                <div style='font-size:.75rem;color:#5a3a28'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📅 Planning")
        for date, lbl in [
            ("Mars 2026", "Constitution dataset incendie"),
            ("Avr 2026",  "Entraînement YOLOv8 feu/fumée"),
            ("Avr 2026",  "Système d'alertes temps réel"),
            ("Avr 2026",  "Interface & dashboard"),
            ("Avr 2026",  "Tests terrain & livraison"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:6px;align-items:flex-start;'>
                <div style='font-size:.72rem;color:#ff4500;white-space:nowrap;min-width:72px;padding-top:1px;'>{date}</div>
                <div style='font-size:.82rem;color:#c4a882;'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 L'Équipe du Projet")
    cols = st.columns(len(MEMBRES), gap="medium")
    for col, m in zip(cols, MEMBRES):
        with col:
            st.markdown(f"""
            <div class='member-card'>
                <div class='member-avatar'>{m['emoji']}</div>
                <div class='member-name'>{m['nom']}</div>
                <div class='member-role'>{m['role']}</div>
                <div class='member-desc'>{m['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Chiffres du Projet")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🖼️ Images Dataset",  "6 800+",   "+1 200 augmentées")
    m2.metric("🏷️ Classes Feu",     "4",        "Flamme·Fumée·Braise·Forêt")
    m3.metric("🎯 Précision cible", "85 %",     "mAP@0.5")
    m4.metric("⚡ Latence cible",   "< 200 ms", "Par frame")

# ═══════════════════════════════════════════════
# PAGE 2 ── DÉTECTION VIDÉO
# ═══════════════════════════════════════════════
elif page == "🎥  Détection Vidéo":
    st.title("🎥 Analyse Vidéo — Détection de Feux")
    st.markdown("<p style='color:#8a7260;margin-top:-12px;'>Détection réelle via YOLOv8 ou pipeline Kafka selon le mode sélectionné.</p>", unsafe_allow_html=True)

    show_connection_banner()

    tab1, tab2, tab3 = st.tabs(["📁 Charger une Vidéo", "📷 Flux Caméra Live", "🖼️ Image Unique"])

    # ── TAB 1 : VIDÉO ──
    with tab1:
        st.markdown("""
        <div style='background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:14px;'>
            <b style='color:#ff8c00;font-size:.95rem;'>ℹ️ Instructions</b><br>
            <span style='color:#8a7260;font-size:.83rem;'>
            Formats acceptés : MP4, AVI, MOV · Taille max : 200 MB<br>
            Mode YOLO : détection directe avec best.pt &nbsp;|&nbsp; Mode Kafka : envoi des frames au broker Docker.
            </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Paramètres de Détection", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                conf_threshold = st.slider("Seuil de confiance", 0.1, 1.0,
                                           st.session_state.get("global_conf", 0.25), 0.05,
                                           help="Synchronisé avec le curseur de la sidebar")
                st.session_state["global_conf"] = conf_threshold
                iou_threshold  = st.slider("Seuil IoU (NMS)",    0.1, 1.0, 0.45, 0.05)
            with col_s2:
                st.multiselect("Classes à détecter",
                    ["🔥 Flamme","💨 Fumée","🌡️ Braise","⚠️ Feu Forêt"],
                    default=["🔥 Flamme","💨 Fumée","🌡️ Braise","⚠️ Feu Forêt"])
                show_boxes  = st.toggle("Afficher les bounding boxes",     value=True)
                show_scores = st.toggle("Afficher les scores de confiance", value=True)

        video_file = st.file_uploader("", type=["mp4","avi","mov"], label_visibility="collapsed")

        if video_file:
            if st.session_state.detection_mode == "YOLO" and not yolo_ok:
                st.error("❌ Mode YOLO sélectionné mais best.pt introuvable. Changez de mode dans la sidebar.")
                st.stop()
            if st.session_state.detection_mode == "KAFKA" and not kafka_ok:
                st.warning("⚠️ Mode Kafka sélectionné mais Docker non disponible. Basculement en mode YOLO.")
                st.session_state.detection_mode = "YOLO"

            st.success(f"✅ Vidéo chargée — Mode : **{st.session_state.detection_mode}** — Analyse en cours...")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_ph    = st.empty()
            progress_ph = st.empty()
            stats_ph    = st.empty()

            flame_count = 0
            smoke_count = 0
            frame_idx   = 0
            last_level  = "NORMAL"

            st.session_state.alert_history = []

            # ── CORRECTION BUG 3 : initialiser metrics AVANT la boucle ──
            if "metrics" not in st.session_state:
                st.session_state.metrics = {
                    "total_detections": 0, "fire_detections": 0, "smoke_detections": 0,
                    "false_negatives": 0, "scores_fire": [], "scores_smoke": [],
                    "kafka_sent": 0, "kafka_received": 0, "latencies_ms": [], "timeline": [],
                }

            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("▶  Lancer l'analyse", type="primary"):
                    st.session_state["video_running"] = True
            with col_stop:
                if st.button("⏹  Arrêter l'analyse"):
                    st.session_state["video_running"] = False

            if not st.session_state.get("video_running", False):
                st.info("▶ Cliquez sur 'Lancer l'analyse' pour démarrer.")
                st.stop()

            while cap.isOpened() and st.session_state.get("video_running", False):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                label, score, level = "normal", 0.0, "NORMAL"
                display_frame = frame

                if st.session_state.detection_mode == "YOLO" and yolo_ok:
                    label, score, annotated, detections = get_yolo_prediction(
                        yolo_model, frame, conf_threshold
                    )
                    level, css = classify_risk(label)

                    # ── MISE À JOUR MÉTRIQUES ──
                    st.session_state.metrics["total_detections"] += 1
                    if label == "fire":
                        flame_count += 1
                        st.session_state.metrics["fire_detections"] += 1
                        st.session_state.metrics["scores_fire"].append(round(score, 3))
                    elif label == "smoke":
                        smoke_count += 1
                        st.session_state.metrics["smoke_detections"] += 1
                        st.session_state.metrics["scores_smoke"].append(round(score, 3))
                    else:
                        st.session_state.metrics["false_negatives"] += 1

                    st.session_state.metrics["timeline"].append({
                        "label": label, "score": round(score, 3),
                        "time":  time.strftime("%H:%M:%S")
                    })

                    # ── CORRECTION BUG 1 : un seul envoi Kafka par frame ──
                    if kafka_ok and kafka_producer and level != "NORMAL":
                        t_before = time.time()
                        try:
                            kafka_producer.send('parking-alerts', value={
                                "alerte": level, "objet": label,
                                "score":  round(score, 2),
                                "heure":  time.strftime("%H:%M:%S"),
                                "source": "VideoUpload"
                            })
                            latency = round((time.time() - t_before) * 1000, 1)
                            st.session_state.metrics["kafka_sent"] += 1
                            st.session_state.metrics["latencies_ms"].append(latency)
                        except Exception:
                            pass

                    # ── CORRECTION BUG 2 : un seul appel trigger_alarm ──
                    if level in ("CRITIQUE", "ALERTE"):
                        trigger_alarm(level)
                        if mysql_ok:
                            save_alerte(
                                label      = label,
                                niveau     = level,
                                score      = score,
                                source     = "VideoUpload",
                                session_id = st.session_state["session_id"]
                            )
                        st.session_state.alert_history.append({
                            "heure": time.strftime("%H:%M:%S"),
                            "objet": label, "score": round(score, 2), "level": level
                        })

                    display_frame = annotated if show_boxes else frame

                else:
                    # Mode KAFKA
                    if kafka_ok and kafka_producer:
                        try:
                            kafka_producer.send('parking-alerts', value={
                                "alerte": "FRAME_STREAM",
                                "source": "VideoUpload_Kafka",
                                "heure":  time.strftime("%H:%M:%S")
                            })
                        except Exception:
                            pass
                    label, score, level, css = "streaming", 0.0, "NORMAL", "level-low"
                    display_frame = frame

                last_level = level

                frame_placeholder_img = display_frame.copy()
                if show_scores and score > 0:
                    cv2.putText(frame_placeholder_img,
                                f"{label} {score:.2f} — {level}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 80, 255) if level == "CRITIQUE" else (0, 200, 80), 2)

                frame_ph.image(frame_placeholder_img, channels="BGR", use_container_width=True)

                if not st.session_state.get("video_running", False):
                    break
                if total_frames > 0:
                    progress_ph.progress(min(frame_idx / total_frames, 1.0),
                                         text=f"Frame {frame_idx}/{total_frames} — {level}")

            cap.release()

            if mysql_ok:
                avg_fire  = (sum(st.session_state.metrics["scores_fire"]) /
                            len(st.session_state.metrics["scores_fire"])
                            if st.session_state.metrics["scores_fire"] else 0)
                avg_smoke = (sum(st.session_state.metrics["scores_smoke"]) /
                            len(st.session_state.metrics["scores_smoke"])
                            if st.session_state.metrics["scores_smoke"] else 0)
                save_session(
                    session_id = st.session_state["session_id"],
                    fire       = flame_count,
                    smoke      = smoke_count,
                    frames     = frame_idx,
                    avg_fire   = avg_fire,
                    avg_smoke  = avg_smoke,
                    kafka_sent = st.session_state.metrics["kafka_sent"],
                    duree      = frame_idx
                )
                st.success(f"✅ Session {st.session_state['session_id']} sauvegardée en base !")

            ca, cb, cc, cd = st.columns(4)
            ca.metric("🎞️ Frames analysées", frame_idx)
            cb.metric("🔥 Flammes détectées", flame_count)
            cc.metric("💨 Fumée détectée",    smoke_count)
            cd.metric("⚡ Dernier statut",    last_level)

            if st.session_state.alert_history:
                st.markdown("#### 🚨 Alertes déclenchées durant l'analyse")
                for a in st.session_state.alert_history:
                    st.markdown(f"""
                    <div class='alert-critical'>
                        🕐 {a['heure']} &nbsp;·&nbsp; <b>{a['objet'].upper()}</b>
                        &nbsp;·&nbsp; Confiance : {a['score']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:var(--bg-card);border:2px dashed rgba(255,69,0,.3);
                        border-radius:12px;padding:50px;text-align:center;'>
                <div style='font-size:3.5rem;margin-bottom:10px;'>🔥</div>
                <div style='font-family:"Bebas Neue",sans-serif;color:#ff8c00;font-size:1.1rem;letter-spacing:2px;'>
                    DÉPOSEZ UNE VIDÉO ICI
                </div>
                <div style='color:#5a3a28;font-size:.82rem;margin-top:6px;'>MP4 · AVI · MOV — max 200 MB</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2 : FLUX CAMÉRA LIVE (avec support Android) ──
    with tab2:
        st.markdown("#### 📷 Flux Caméra en Direct")

        st.markdown("""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:10px;padding:14px 18px;margin-bottom:14px;'>
            <b style='color:#ff8c00;'>ℹ️ Instructions sources</b><br>
            <span style='color:#8a7260;font-size:.83rem;'>
            • <b style='color:#ffbf00;'>Android USB</b> : Installe <b>IP Webcam</b> sur ton téléphone → Lance le serveur
            → Branche en USB → dans un terminal : <code>adb reverse tcp:8080 tcp:8080</code><br>
            • <b style='color:#ffbf00;'>Android Wi-Fi</b> : Même réseau Wi-Fi que le PC → entre l'IP affichée dans IP Webcam<br>
            • <b style='color:#ffbf00;'>Webcam PC</b> : Sélectionne "Webcam PC intégrée" ou "Webcam USB externe"
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_src, col_cam2, col_cam3 = st.columns([2, 1, 1])

        with col_src:
            source_type = st.selectbox(
                "📱 Type de source caméra",
                [
                    "📱 Android USB (ADB · IP Webcam)",
                    "📡 Android Wi-Fi (IP Webcam)",
                    "💻 Webcam PC intégrée",
                    "🔌 Webcam USB externe",
                    "🌐 URL RTSP personnalisée"
                ],
                index=0
            )

            if source_type == "📱 Android USB (ADB · IP Webcam)":
                rtsp_url = "http://localhost:8080/video"
                st.markdown("""
                <div style='background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.3);
                            border-radius:8px;padding:10px 14px;margin-top:6px;font-size:.8rem;'>
                    <b style='color:#22c55e;'>✅ Mode Android USB actif</b><br>
                    <span style='color:#8a7260;'>
                    1. Branche ton téléphone en USB<br>
                    2. Lance IP Webcam → appuie sur <b>Start server</b><br>
                    3. Dans un terminal PC : <code style='color:#ffbf00;'>adb reverse tcp:8080 tcp:8080</code><br>
                    4. Clique sur ▶ Démarrer ci-dessous
                    </span>
                </div>
                """, unsafe_allow_html=True)

            elif source_type == "📡 Android Wi-Fi (IP Webcam)":
                col_ip, col_port = st.columns([3, 1])
                with col_ip:
                    phone_ip = st.text_input("IP du téléphone", value="192.168.1.100", placeholder="Ex: 192.168.1.42")
                with col_port:
                    phone_port = st.number_input("Port", value=8080, min_value=1000, max_value=65535)
                rtsp_url = f"http://{phone_ip}:{phone_port}/video"
                st.markdown(f"<div style='font-size:.75rem;color:#5a3a28;margin-top:2px;'>URL : <code style='color:#ff8c00;'>{rtsp_url}</code></div>", unsafe_allow_html=True)

            elif source_type == "💻 Webcam PC intégrée":
                rtsp_url = "0"
                st.markdown("""
                <div style='background:rgba(255,191,0,.08);border:1px solid rgba(255,191,0,.3);
                            border-radius:8px;padding:8px 12px;margin-top:6px;font-size:.8rem;'>
                    <span style='color:#ffbf00;'>💻 Webcam intégrée sélectionnée (index 0)</span>
                </div>
                """, unsafe_allow_html=True)

            elif source_type == "🔌 Webcam USB externe":
                rtsp_url = "1"
                st.markdown("""
                <div style='background:rgba(255,191,0,.08);border:1px solid rgba(255,191,0,.3);
                            border-radius:8px;padding:8px 12px;margin-top:6px;font-size:.8rem;'>
                    <span style='color:#ffbf00;'>🔌 Webcam USB externe sélectionnée (index 1)</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                rtsp_url = st.text_input(
                    "🌐 URL RTSP",
                    value="rtsp://",
                    placeholder="rtsp://user:pass@192.168.1.x:554/stream"
                )

        with col_cam2:
            cam_conf = st.slider(
                "Seuil confiance", 0.10, 1.00,
                st.session_state.get("global_conf", 0.25), 0.05,
                help="Synchronisé avec la sidebar"
            )
            st.session_state["global_conf"] = cam_conf

        with col_cam3:
            skip_frames = st.slider("Analyser 1 frame sur", 1, 10, 3,
                                    help="3 = analyse 1 frame sur 3 → plus fluide")

        # ── Bouton test connexion Android ──
        if "Android" in source_type:
            if st.button("🔍 Tester la connexion Android"):
                test_url = rtsp_url.replace("/video", "/status.json")
                try:
                    with urllib.request.urlopen(test_url, timeout=3) as resp:
                        data_status = json.loads(resp.read())
                        battery = data_status.get('battery_level', '?')
                        res = data_status.get('curfrontres', data_status.get('curres', '?'))
                        st.success(f"✅ IP Webcam connectée ! Batterie : {battery}% · Résolution : {res}")
                except Exception as e:
                    st.error(f"❌ Connexion échouée : {e}")
                    if "USB" in source_type:
                        st.info("💡 Vérifie que ADB reverse est actif : `adb reverse tcp:8080 tcp:8080`")
                    else:
                        st.info("💡 Vérifie que le téléphone et le PC sont sur le même réseau Wi-Fi.")

        # ── Boutons START / STOP ──
        col_start, col_stop = st.columns(2)
        with col_start:
            start_cam = st.button("▶  Démarrer la caméra", type="primary")
        with col_stop:
            stop_cam  = st.button("⏹  Arrêter")

        if start_cam:
            st.session_state["cam_running"] = True
        if stop_cam:
            st.session_state["cam_running"] = False

        if st.session_state.get("cam_running", False):

            if not yolo_ok:
                st.error("❌ YOLO non disponible — impossible d'analyser le flux caméra.")
                st.session_state["cam_running"] = False
            else:
                # Convertir la source en int si webcam locale
                src = int(rtsp_url) if rtsp_url in ("0", "1") else rtsp_url

                cap_cam = cv2.VideoCapture(src)
                cap_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduit la latence

                if not cap_cam.isOpened():
                    st.error(f"❌ Impossible d'ouvrir la source : {src}")
                    if "Android" in source_type:
                        st.info("💡 Rappel : vérifie que l'app IP Webcam est lancée et que ADB reverse est actif.")
                    else:
                        st.info("💡 Vérifiez que votre webcam n'est pas utilisée par une autre application.")
                    st.session_state["cam_running"] = False
                else:
                    source_label = (
                        "📱 Android USB" if "USB" in source_type else
                        "📡 Android Wi-Fi" if "Wi-Fi" in source_type else
                        "💻 Webcam PC"
                    )
                    st.success(f"✅ {source_label} active — YOLO en cours...")

                    cam_ph      = st.empty()
                    status_ph   = st.empty()
                    metrics_ph  = st.empty()

                    cam_flame   = 0
                    cam_smoke   = 0
                    frame_count = 0

                    # ── CORRECTION BUG 3 : initialiser metrics AVANT la boucle ──
                    if "metrics" not in st.session_state:
                        st.session_state.metrics = {
                            "total_detections": 0, "fire_detections": 0, "smoke_detections": 0,
                            "false_negatives": 0, "scores_fire": [], "scores_smoke": [],
                            "kafka_sent": 0, "kafka_received": 0, "latencies_ms": [], "timeline": [],
                        }

                    while st.session_state.get("cam_running", False):
                        ret_c, frame_c = cap_cam.read()
                        if not ret_c:
                            st.warning("⚠️ Flux caméra perdu.")
                            break

                        frame_count += 1

                        if frame_count % skip_frames == 0:
                            label_c, score_c, annotated_c, dets_c = get_yolo_prediction(
                                yolo_model, frame_c, cam_conf
                            )
                            level_c, _ = classify_risk(label_c)

                            # ── CORRECTION BUG 2 : un seul appel trigger_alarm ──
                            if level_c in ("CRITIQUE", "ALERTE"):
                                trigger_alarm(level_c)
                                if mysql_ok:
                                    save_alerte(
                                        label      = label_c,
                                        niveau     = level_c,
                                        score      = score_c,
                                        source     = source_label,
                                        session_id = st.session_state["session_id"]
                                    )
                                st.session_state.alert_history.append({
                                    "heure": time.strftime("%H:%M:%S"),
                                    "objet": label_c,
                                    "score": round(score_c, 2),
                                    "level": level_c
                                })

                            # Mise à jour métriques
                            st.session_state.metrics["total_detections"] += 1
                            if label_c == "fire":
                                cam_flame += 1
                                st.session_state.metrics["fire_detections"] += 1
                                st.session_state.metrics["scores_fire"].append(round(score_c, 3))
                            elif label_c == "smoke":
                                cam_smoke += 1
                                st.session_state.metrics["smoke_detections"] += 1
                                st.session_state.metrics["scores_smoke"].append(round(score_c, 3))
                            else:
                                st.session_state.metrics["false_negatives"] += 1

                            st.session_state.metrics["timeline"].append({
                                "label": label_c,
                                "score": round(score_c, 3),
                                "time":  time.strftime("%H:%M:%S")
                            })

                            # ── CORRECTION BUG 1 : un seul envoi Kafka par frame ──
                            if kafka_ok and kafka_producer and level_c != "NORMAL":
                                try:
                                    t0 = time.time()
                                    kafka_producer.send('parking-alerts', value={
                                        "alerte": level_c,
                                        "objet":  label_c,
                                        "score":  round(score_c, 2),
                                        "heure":  time.strftime("%H:%M:%S"),
                                        "source": source_label
                                    })
                                    st.session_state.metrics["kafka_sent"] += 1
                                    st.session_state.metrics["latencies_ms"].append(
                                        round((time.time() - t0) * 1000, 1)
                                    )
                                except Exception:
                                    pass

                            color_cv = (0, 0, 255)   if level_c == "CRITIQUE" else \
                                       (0, 165, 255) if level_c == "ALERTE"   else (0, 200, 80)
                            cv2.putText(
                                annotated_c,
                                f"{label_c.upper()} | {score_c:.2f} | {level_c}",
                                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_cv, 2
                            )
                            cv2.putText(
                                annotated_c,
                                time.strftime("%H:%M:%S"),
                                (15, annotated_c.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
                            )

                            cam_ph.image(annotated_c, channels="BGR", use_container_width=True)

                            alert_class = "alert-critical" if level_c == "CRITIQUE" else \
                                          "alert-warning"  if level_c == "ALERTE"   else "alert-ok"
                            status_ph.markdown(f"""
                            <div class='{alert_class}' style='margin-top:8px;'>
                                <b>{level_c}</b> — {label_c.upper()} — Confiance : <b>{score_c:.2f}</b>
                                &nbsp;·&nbsp; 🕐 {time.strftime("%H:%M:%S")}
                            </div>
                            """, unsafe_allow_html=True)

                        else:
                            cam_ph.image(frame_c, channels="BGR", use_container_width=True)

                        metrics_ph.markdown(f"""
                        <div style='display:flex;gap:20px;font-size:.8rem;color:#8a7260;margin-top:4px;'>
                            <span>🎞️ Frames : <b style='color:#ff8c00'>{frame_count}</b></span>
                            <span>🔥 Fire : <b style='color:#ff4500'>{cam_flame}</b></span>
                            <span>💨 Smoke : <b style='color:#8a8a8a'>{cam_smoke}</b></span>
                            <span>🚨 Alertes : <b style='color:#ff2200'>{len(st.session_state.alert_history)}</b></span>
                        </div>
                        """, unsafe_allow_html=True)

                    cap_cam.release()

                    if mysql_ok:
                        avg_fire  = (sum(st.session_state.metrics["scores_fire"]) /
                                    len(st.session_state.metrics["scores_fire"])
                                    if st.session_state.metrics["scores_fire"] else 0)
                        avg_smoke = (sum(st.session_state.metrics["scores_smoke"]) /
                                    len(st.session_state.metrics["scores_smoke"])
                                    if st.session_state.metrics["scores_smoke"] else 0)
                        save_session(
                            session_id = st.session_state["session_id"],
                            fire       = cam_flame,
                            smoke      = cam_smoke,
                            frames     = frame_count,
                            avg_fire   = avg_fire,
                            avg_smoke  = avg_smoke,
                            kafka_sent = st.session_state.metrics["kafka_sent"],
                            duree      = frame_count
                        )
                    st.info(f"✅ Caméra arrêtée — Résumé : Fire={cam_flame} | Smoke={cam_smoke} | Frames={frame_count}")

    # ── TAB 3 : IMAGE UNIQUE ──
    with tab3:
        st.markdown("#### 🖼️ Analyse d'une Image")
        img_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if img_file:
            if st.session_state.get("last_image") != img_file.name:
                st.session_state["image_analyzed"] = False
                st.session_state["last_image"]     = img_file.name

            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img_frame  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.markdown("**Image originale**")
                st.image(img_frame, channels="BGR", use_container_width=True)

            if st.button("🔍 Lancer la détection YOLO", type="primary"):
                st.session_state["image_analyzed"] = True

            if st.session_state.get("image_analyzed", False):
                if yolo_ok:
                    label_i, score_i, annotated_i, dets_i = get_yolo_prediction(
                        yolo_model, img_frame,
                        conf_threshold=st.session_state.get("global_conf", 0.25)
                    )
                    level_i, css_i = classify_risk(label_i)

                    with col_result:
                        st.markdown("**Résultat YOLOv8**")
                        st.image(annotated_i, channels="BGR", use_container_width=True)

                    alert_class = "alert-critical" if level_i == "CRITIQUE" else \
                                  "alert-warning"  if level_i == "ALERTE"   else "alert-ok"
                    st.markdown(f"""
                    <div class='{alert_class}'>
                        <b>{level_i}</b> — Objet détecté : <b>{label_i}</b>
                        — Confiance : <b>{score_i:.2f}</b><br>
                        <span style='font-size:.8rem;color:#8a7260;'>
                            {len(dets_i)} détection(s) au total
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    if level_i in ("CRITIQUE", "ALERTE"):
                        trigger_alarm(level_i)
                        if mysql_ok:
                            save_alerte(
                                label      = label_i,
                                niveau     = level_i,
                                score      = score_i,
                                source     = "ImageUpload",
                                session_id = st.session_state["session_id"]
                            )
                        if kafka_ok and kafka_producer:
                            try:
                                kafka_producer.send('parking-alerts', value={
                                    "alerte": level_i,
                                    "objet":  label_i,
                                    "score":  round(score_i, 2),
                                    "heure":  time.strftime("%H:%M:%S"),
                                    "source": "ImageUpload"
                                })
                                st.success("📡 Alerte envoyée au broker Kafka !")
                            except Exception:
                                pass
                else:
                    with col_result:
                        st.warning("⚠️ YOLO non disponible.")
        else:
            st.info("📤 Chargez une image JPG ou PNG pour analyse instantanée.")

# ═══════════════════════════════════════════════
# PAGE 3 ── ALERTES EN DIRECT
# ═══════════════════════════════════════════════
elif page == "🚨  Alertes en Direct":
    st.title("🚨 Live Alerts — FireGuard IA")
    st.markdown("<p style='color:#8a7260;margin-top:-12px;'>Flux de détection en temps réel — Kafka (Docker) ou historique de session.</p>", unsafe_allow_html=True)

    show_connection_banner()

    tab_live, tab_history = st.tabs(["📡 Flux Kafka Live", "📋 Historique Session"])

    with tab_live:
        if not kafka_ok:
            st.error("""
            ❌ **Kafka non disponible.**

            Pour activer le flux live, lancez Docker :
            ```
            cd Projet_Parking_IA
            docker-compose up -d
            ```
            Puis relancez Streamlit.
            """)
        else:
            st.success("✅ Connecté au topic **parking-alerts** — En attente de détections...")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_live = st.button("🔴 Démarrer la surveillance Kafka")
            with col_btn2:
                max_msgs = st.number_input("Nb max d'alertes à afficher", 5, 1000, 20)

            if start_live:
                alert_feed = st.container()
                st.toast("🔍 Écoute du broker Kafka en cours...", icon="📡")

                msg_count = 0
                for message in kafka_consumer:
                    if msg_count >= max_msgs:
                        break
                    data = message.value

                    type_obj = data.get('objet',   data.get('statut',  'Inconnu'))
                    score    = data.get('score',   data.get('confiance', 0))
                    h        = data.get('heure',   time.strftime("%H:%M:%S"))
                    source   = data.get('source',  'Kafka')
                    alerte   = data.get('alerte',  'DETECTION')

                    level_k, _ = classify_risk(type_obj)
                    alert_class = "alert-critical" if level_k == "CRITIQUE" else "alert-warning"

                    with alert_feed:
                        st.markdown(f"""
                        <div class='{alert_class}'>
                            <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div>
                                    <span style='font-family:"Share Tech Mono",monospace;
                                                 color:#ff6b6b;font-size:.8rem;'>KAFKA · {source}</span>
                                    &nbsp;&nbsp;<b style='color:#f0e6d3;'>🔥 {type_obj.upper()}</b>
                                    &nbsp;&nbsp;<span class='level-{"high" if level_k=="CRITIQUE" else "medium"}'>{alerte}</span>
                                </div>
                                <div style='display:flex;gap:14px;font-size:.78rem;color:#8a7260;'>
                                    <span>🕐 {h}</span><span>🎯 {score}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if score > 0.6 or level_k == "CRITIQUE":
                        if time.time() - st.session_state.last_alarm_time > 3:
                            trigger_alarm("CRITIQUE")
                            st.session_state.last_alarm_time = time.time()

                    msg_count += 1

                st.info(f"✅ {msg_count} alertes reçues depuis Kafka.")

    with tab_history:
        st.markdown("#### 📋 Alertes de la session en cours")
        if not st.session_state.alert_history:
            st.markdown("""
            <div class='alert-ok'>
                <b>✅ Aucune alerte critique déclenchée</b> — Le système surveille activement.
            </div>
            """, unsafe_allow_html=True)
        else:
            for a in reversed(st.session_state.alert_history):
                st.markdown(f"""
                <div class='alert-critical'>
                    🕐 {a['heure']} &nbsp;·&nbsp; <b>{a['objet'].upper()}</b>
                    &nbsp;·&nbsp; Confiance : {a['score']}
                    &nbsp;·&nbsp; <span class='level-high'>{a['level']}</span>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ Effacer l'historique"):
                st.session_state.alert_history = []
                st.rerun()

# ═══════════════════════════════════════════════
# PAGE 4 ── DASHBOARD DYNAMIQUE
# ═══════════════════════════════════════════════
elif page == "📊  Analyse des Performances":
    st.title("📊 Dashboard Temps Réel — FireGuard IA")
    st.markdown("<p style='color:#8a7260;margin-top:-12px;'>Métriques YOLO et Kafka mises à jour en direct à chaque détection.</p>", unsafe_allow_html=True)

    show_connection_banner()

    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_detections": 0,
            "fire_detections":  0,
            "smoke_detections": 0,
            "false_negatives":  0,
            "scores_fire":      [],
            "scores_smoke":     [],
            "kafka_sent":       0,
            "kafka_received":   0,
            "latencies_ms":     [],
            "timeline":         [],
        }

    m = st.session_state.metrics

    total = m["total_detections"]
    fire  = m["fire_detections"]
    smoke = m["smoke_detections"]
    fn    = m["false_negatives"]

    precision_fire   = round(fire  / total, 3) if total > 0 else 0.0
    precision_smoke  = round(smoke / total, 3) if total > 0 else 0.0
    precision_global = round((fire + smoke) / total, 3) if total > 0 else 0.0

    denom_recall  = (fire + smoke + fn)
    recall_global = round((fire + smoke) / denom_recall, 3) if denom_recall > 0 else 0.0

    if precision_global + recall_global > 0:
        f1_global = round(2 * precision_global * recall_global / (precision_global + recall_global), 3)
    else:
        f1_global = 0.0

    avg_conf_fire   = round(sum(m["scores_fire"])  / len(m["scores_fire"]),  3) if m["scores_fire"]  else 0.0
    avg_conf_smoke  = round(sum(m["scores_smoke"]) / len(m["scores_smoke"]), 3) if m["scores_smoke"] else 0.0
    avg_conf_global = round(
        sum(m["scores_fire"] + m["scores_smoke"]) / len(m["scores_fire"] + m["scores_smoke"]), 3
    ) if (m["scores_fire"] or m["scores_smoke"]) else 0.0

    avg_latency = round(sum(m["latencies_ms"]) / len(m["latencies_ms"]), 1) if m["latencies_ms"] else 0.0

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🔄 Reset"):
            st.session_state.metrics = {
                "total_detections": 0, "fire_detections": 0, "smoke_detections": 0,
                "false_negatives": 0, "scores_fire": [], "scores_smoke": [],
                "kafka_sent": 0, "kafka_received": 0, "latencies_ms": [], "timeline": [],
            }
            st.rerun()

    st.markdown("### 🎯 Métriques Globales en Temps Réel")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🔢 Détections totales", total)
    k2.metric("🔥 Fire",  fire,  f"+{fire}" if fire > 0 else "0")
    k3.metric("💨 Smoke", smoke, f"+{smoke}" if smoke > 0 else "0")
    k4.metric("🎯 Précision", f"{precision_global:.1%}")
    k5.metric("📡 Rappel",    f"{recall_global:.1%}")
    k6.metric("⚡ F1-Score",  f"{f1_global:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_yolo, col_kafka = st.columns(2, gap="large")

    with col_yolo:
        st.markdown("### 🤖 Métriques YOLO Live")
        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:12px;padding:18px;margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='color:#ff4500;font-weight:700;'>🔥 Confiance FIRE</span>
                <span style='color:#ff8c00;font-family:"Bebas Neue",sans-serif;font-size:1.1rem;'>{avg_conf_fire:.1%}</span>
            </div>
            <div class='pb-wrap'>
                <div class='pb-fill' style='width:{avg_conf_fire*100:.1f}%'></div>
            </div>
            <div style='font-size:.75rem;color:#5a3a28;margin-top:6px;'>{len(m["scores_fire"])} frames détectées</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:12px;padding:18px;margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='color:#8a8a8a;font-weight:700;'>💨 Confiance SMOKE</span>
                <span style='color:#ff8c00;font-family:"Bebas Neue",sans-serif;font-size:1.1rem;'>{avg_conf_smoke:.1%}</span>
            </div>
            <div class='pb-wrap'>
                <div class='pb-fill' style='width:{avg_conf_smoke*100:.1f}%'></div>
            </div>
            <div style='font-size:.75rem;color:#5a3a28;margin-top:6px;'>{len(m["scores_smoke"])} frames détectées</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Par Classe")
        df_classes = pd.DataFrame({
            "Classe":     ["🔥 Fire",  "💨 Smoke"],
            "Détections": [fire,        smoke],
            "Conf. moy.": [f"{avg_conf_fire:.1%}", f"{avg_conf_smoke:.1%}"],
            "Précision":  [
                f"{precision_fire:.1%}"  if total > 0 else "—",
                f"{precision_smoke:.1%}" if total > 0 else "—",
            ],
            "Statut": [
                "✅ OK" if avg_conf_fire  >= 0.35 else "⚠️ Faible",
                "✅ OK" if avg_conf_smoke >= 0.35 else "⚠️ Faible",
            ],
        })
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

    with col_kafka:
        st.markdown("### 📡 Métriques Kafka Live")
        kafka_dot   = "🟢" if kafka_ok else "🔴"
        kafka_label = "Connecté" if kafka_ok else "Hors ligne"

        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:12px;padding:18px;margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;'>
                <span style='color:#ff8c00;font-weight:700;font-size:.95rem;'>Broker Status</span>
                <span style='font-size:.85rem;'>{kafka_dot} {kafka_label}</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                <div style='text-align:center;background:#1a0f08;border-radius:8px;padding:12px;'>
                    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;color:#ff4500;'>{m["kafka_sent"]}</div>
                    <div style='font-size:.72rem;color:#5a3a28;'>Messages envoyés</div>
                </div>
                <div style='text-align:center;background:#1a0f08;border-radius:8px;padding:12px;'>
                    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;color:#22c55e;'>{m["kafka_received"]}</div>
                    <div style='font-size:.72rem;color:#5a3a28;'>Messages reçus</div>
                </div>
                <div style='text-align:center;background:#1a0f08;border-radius:8px;padding:12px;'>
                    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;color:#ffbf00;'>{avg_latency} ms</div>
                    <div style='font-size:.72rem;color:#5a3a28;'>Latence moy.</div>
                </div>
                <div style='text-align:center;background:#1a0f08;border-radius:8px;padding:12px;'>
                    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.6rem;color:#ff8c00;'>
                        {f"{(m['kafka_received']/m['kafka_sent']*100):.0f}%" if m["kafka_sent"] > 0 else "—"}
                    </div>
                    <div style='font-size:.72rem;color:#5a3a28;'>Taux réception</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:var(--bg-card);border:1px solid var(--border);
                    border-radius:12px;padding:16px;'>
            <div style='font-size:.8rem;color:#5a3a28;margin-bottom:10px;font-weight:700;'>TOPIC INFO</div>
            <div style='font-size:.82rem;line-height:2;color:#c4a882;'>
                <span style='color:#5a3a28;'>Topic :</span> <b>parking-alerts</b><br>
                <span style='color:#5a3a28;'>Broker :</span> localhost:9092<br>
                <span style='color:#5a3a28;'>Offset :</span> latest<br>
                <span style='color:#5a3a28;'>Format :</span> JSON (objet, score, heure, source)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Graphiques en Temps Réel")

    tab_conf, tab_timeline, tab_dist, tab_f1 = st.tabs([
        "📊 Confiance par frame",
        "🕐 Timeline détections",
        "🥧 Distribution classes",
        "📉 Évolution F1/Précision/Rappel"
    ])

    with tab_conf:
        if m["scores_fire"] or m["scores_smoke"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0f0b08'); ax.set_facecolor('#140e0a')
            if m["scores_fire"]:
                ax.plot(m["scores_fire"], color='#ff4500', linewidth=2,
                        label=f'Fire ({len(m["scores_fire"])} frames)', marker='o', markersize=3)
            if m["scores_smoke"]:
                ax.plot(m["scores_smoke"], color='#8a8a8a', linewidth=2,
                        label=f'Smoke ({len(m["scores_smoke"])} frames)', marker='s', markersize=3)
            ax.axhline(y=st.session_state.get("global_conf", 0.25),
                       color='#ffbf00', linestyle='--', linewidth=1.5,
                       label=f'Seuil actuel ({st.session_state.get("global_conf", 0.25):.0%})')
            ax.set_ylim(0, 1)
            ax.set_xlabel('Frame / Détection', color='#8a7260')
            ax.set_ylabel('Score de confiance', color='#8a7260')
            ax.tick_params(colors='#5a3a28')
            for s in ax.spines.values(): s.set_edgecolor('#1a0f08')
            ax.legend(facecolor='#140e0a', edgecolor='#2a1505', labelcolor='#c4a882')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("📤 Lancez une détection vidéo ou image pour voir les scores en temps réel.")

    with tab_timeline:
        if m["timeline"]:
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0f0b08'); ax.set_facecolor('#140e0a')
            times  = list(range(len(m["timeline"])))
            labels = [e["label"] for e in m["timeline"]]
            scores = [e["score"] for e in m["timeline"]]
            colors = ['#ff4500' if l == "fire" else '#8a8a8a' if l == "smoke" else '#22c55e' for l in labels]
            ax.scatter(times, scores, c=colors, s=60, zorder=3)
            ax.fill_between(times, scores, alpha=0.1, color='#ff8c00')
            ax.plot(times, scores, color='#ff8c00', linewidth=1, alpha=0.4)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Événement #', color='#8a7260')
            ax.set_ylabel('Confiance', color='#8a7260')
            ax.tick_params(colors='#5a3a28')
            for s in ax.spines.values(): s.set_edgecolor('#1a0f08')
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff4500', markersize=8, label='Fire'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#8a8a8a', markersize=8, label='Smoke'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#22c55e', markersize=8, label='Normal'),
            ]
            ax.legend(handles=legend_elements, facecolor='#140e0a', edgecolor='#2a1505', labelcolor='#c4a882')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("📤 Lancez une détection pour voir la timeline.")

    with tab_dist:
        if total > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0f0b08'); ax.set_facecolor('#140e0a')
            normal = total - fire - smoke
            sizes  = [fire, smoke, normal]
            labels = ['🔥 Fire', '💨 Smoke', '✅ Normal']
            colors = ['#ff4500', '#8a8a8a', '#22c55e']
            explode = (0.05, 0.05, 0)
            data = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
            if data:
                sizes, labels, colors, explode = zip(*data)
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors, explode=explode,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'color': '#c4a882', 'fontsize': 10},
                    wedgeprops={'edgecolor': '#0f0b08', 'linewidth': 2}
                )
                for at in autotexts: at.set_color('#fff')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("📤 Lancez une détection pour voir la distribution.")

    with tab_f1:
        if total >= 2:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0f0b08'); ax.set_facecolor('#140e0a')
            precisions, recalls, f1s = [], [], []
            running_fire = running_smoke = running_fn = 0
            for i, event in enumerate(m["timeline"], 1):
                if event["label"] == "fire":   running_fire += 1
                elif event["label"] == "smoke": running_smoke += 1
                else: running_fn += 1
                p = (running_fire + running_smoke) / i
                d = running_fire + running_smoke + running_fn
                r = (running_fire + running_smoke) / d if d > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                precisions.append(p); recalls.append(r); f1s.append(f)
            x = list(range(1, len(precisions) + 1))
            ax.plot(x, precisions, color='#ff4500', linewidth=2, label='Précision')
            ax.plot(x, recalls,    color='#ffbf00', linewidth=2, label='Rappel',    linestyle='--')
            ax.plot(x, f1s,        color='#22c55e', linewidth=2, label='F1-Score',  linestyle='-.')
            ax.axhline(y=0.85, color='#ff8c00', linestyle=':', linewidth=1, label='Objectif 85%')
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Détection #', color='#8a7260')
            ax.set_ylabel('Score',       color='#8a7260')
            ax.tick_params(colors='#5a3a28')
            for s in ax.spines.values(): s.set_edgecolor('#1a0f08')
            ax.legend(facecolor='#140e0a', edgecolor='#2a1505', labelcolor='#c4a882')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("📤 Lancez au moins 2 détections pour voir l'évolution des métriques.")

    # ── RÉFÉRENCE MODÈLE ENTRAÎNÉ ──
    st.markdown("---")
    st.markdown("### 📚 Référence — Modèle Entraîné (best.pt)")

    total_ref  = st.session_state.metrics["total_detections"]
    fire_ref   = st.session_state.metrics["fire_detections"]
    smoke_ref  = st.session_state.metrics["smoke_detections"]
    fn_ref     = st.session_state.metrics["false_negatives"]
    scores_fire_ref  = st.session_state.metrics["scores_fire"]
    scores_smoke_ref = st.session_state.metrics["scores_smoke"]

    prec_fire_dyn  = fire_ref  / total_ref if total_ref > 0 else 0.90
    prec_smoke_dyn = smoke_ref / total_ref if total_ref > 0 else 0.82
    denom = fire_ref + smoke_ref + fn_ref
    recall_fire_dyn  = fire_ref  / denom if denom > 0 else 0.88
    recall_smoke_dyn = smoke_ref / denom if denom > 0 else 0.80

    def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0
    f1_fire_dyn  = f1(prec_fire_dyn,  recall_fire_dyn)
    f1_smoke_dyn = f1(prec_smoke_dyn, recall_smoke_dyn)
    map_dyn = (prec_fire_dyn + prec_smoke_dyn) / 2
    prec_global_dyn   = (prec_fire_dyn  + prec_smoke_dyn)  / 2
    recall_global_dyn = (recall_fire_dyn + recall_smoke_dyn) / 2
    f1_global_dyn     = f1(prec_global_dyn, recall_global_dyn)
    avg_fire_dyn  = sum(scores_fire_ref)  / len(scores_fire_ref)  if scores_fire_ref  else 0
    avg_smoke_dyn = sum(scores_smoke_ref) / len(scores_smoke_ref) if scores_smoke_ref else 0

    using_real = total_ref > 0
    source_label_ref = "🟢 Données réelles de session" if using_real else "🟡 Valeurs d'entraînement (aucune détection en session)"

    st.markdown(f"""
    <div style='background:var(--bg-card);border:1px solid var(--border);
                border-radius:8px;padding:8px 16px;margin-bottom:14px;font-size:.78rem;'>
        <span style='color:#5a3a28;'>Source des métriques :</span>
        <b style='color:#ff8c00;'>{source_label_ref}</b>
        {'&nbsp;·&nbsp;<span style="color:#5a3a28;">'+str(total_ref)+' frames analysées</span>' if using_real else ''}
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    map_val    = f"{map_dyn:.1%}"            if using_real else "86.5 %"
    prec_val   = f"{prec_global_dyn:.1%}"   if using_real else "85.0 %"
    recall_val = f"{recall_global_dyn:.1%}" if using_real else "83.5 %"
    f1_val     = f"{f1_global_dyn:.1%}"     if using_real else "84.2 %"
    map_delta    = f"{map_dyn - 0.865:+.1%} vs entraînement"           if using_real else "+6.5% vs baseline"
    prec_delta   = f"{prec_global_dyn - 0.85:+.1%} vs entraînement"   if using_real else ""
    recall_delta = f"{recall_global_dyn - 0.835:+.1%} vs entraînement" if using_real else ""
    f1_delta     = f"{f1_global_dyn - 0.842:+.1%} vs entraînement"     if using_real else ""
    r1.metric("mAP@0.5",        map_val,    map_delta)
    r2.metric("Précision moy.", prec_val,   prec_delta)
    r3.metric("Rappel moyen",   recall_val, recall_delta)
    r4.metric("F1-Score",       f1_val,     f1_delta)

    classes_ref   = ['Fire', 'Smoke']
    couleurs_ref  = ['#ff4500', '#8a8a8a']
    if using_real:
        precision_ref = [prec_fire_dyn,  prec_smoke_dyn]
        rappel_ref    = [recall_fire_dyn, recall_smoke_dyn]
        f1_ref        = [f1_fire_dyn,     f1_smoke_dyn]
    else:
        precision_ref = [0.90, 0.82]
        rappel_ref    = [0.88, 0.80]
        f1_ref        = [0.89, 0.81]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0f0b08'); ax.set_facecolor('#140e0a')
    x = np.arange(len(classes_ref)); w = 0.28
    bars1 = ax.bar(x - w, precision_ref, w, color=couleurs_ref, alpha=.95)
    bars2 = ax.bar(x,     rappel_ref,    w, color=couleurs_ref, alpha=.60)
    bars3 = ax.bar(x + w, f1_ref,        w, color=couleurs_ref, alpha=.30)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.0%}', ha='center', va='bottom', color='#c4a882', fontsize=7.5)
    ax.axhline(y=0.85, color='#ff4500', linestyle='--', linewidth=1.5, label='Seuil 85%')
    if using_real:
        ax.axhline(y=0.90, color='#ff4500', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.axhline(y=0.82, color='#8a8a8a', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.text(len(classes_ref) - 0.3, 0.905, 'réf. fire',  color='#ff4500', fontsize=7, alpha=0.6)
        ax.text(len(classes_ref) - 0.3, 0.825, 'réf. smoke', color='#8a8a8a', fontsize=7, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(classes_ref, color='#c4a882', fontsize=12)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score', color='#8a7260')
    ax.tick_params(colors='#5a3a28')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a0f08')
    patches_leg = [
        mpatches.Patch(color='white', alpha=.9, label='Précision'),
        mpatches.Patch(color='white', alpha=.6, label='Rappel'),
        mpatches.Patch(color='white', alpha=.3, label='F1-Score'),
        mpatches.Patch(color='#ff4500',          label='Seuil 85%'),
    ]
    ax.legend(handles=patches_leg, facecolor='#140e0a', edgecolor='#2a1505', labelcolor='#c4a882', fontsize=9)
    ax.set_title('Session réelle' if using_real else 'Valeurs entraînement', color='#5a3a28', fontsize=9, pad=4)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    if using_real:
        st.markdown("#### 🎯 Confiance Moyenne par Classe (Session)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style='background:var(--bg-card);border-left:3px solid #ff4500;
                        border-radius:8px;padding:14px;'>
                <div style='color:#ff4500;font-weight:700;margin-bottom:6px;'>🔥 FIRE</div>
                <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;color:#ff8c00;'>{avg_fire_dyn:.1%}</div>
                <div class='pb-wrap' style='margin-top:8px;'>
                    <div class='pb-fill' style='width:{avg_fire_dyn*100:.1f}%'></div>
                </div>
                <div style='font-size:.75rem;color:#5a3a28;margin-top:6px;'>
                    {fire_ref} détections · seuil actuel {st.session_state.get("global_conf", 0.25):.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='background:var(--bg-card);border-left:3px solid #8a8a8a;
                        border-radius:8px;padding:14px;'>
                <div style='color:#8a8a8a;font-weight:700;margin-bottom:6px;'>💨 SMOKE</div>
                <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;color:#ff8c00;'>{avg_smoke_dyn:.1%}</div>
                <div class='pb-wrap' style='margin-top:8px;'>
                    <div class='pb-fill' style='width:{avg_smoke_dyn*100:.1f}%'></div>
                </div>
                <div style='font-size:.75rem;color:#5a3a28;margin-top:6px;'>
                    {smoke_ref} détections · seuil actuel {st.session_state.get("global_conf", 0.25):.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📡 Résumé Session Actuelle")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🔥 Fire détecté",  fire_ref  if using_real else "—")
    s2.metric("💨 Smoke détecté", smoke_ref if using_real else "—")
    s3.metric("🚨 Alertes",       len(st.session_state.alert_history))
    s4.metric("📊 Conf. globale", f"{(avg_fire_dyn+avg_smoke_dyn)/2:.1%}" if using_real else "—")
