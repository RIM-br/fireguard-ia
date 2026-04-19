import streamlit as st
import os
import base64

def trigger_alarm(level):
    """
    Déclenche l'alarme sonore À CHAQUE détection fire/smoke.
    Pas de blocage après la première fois.
    """
    if level not in ("CRITIQUE", "ALERTE"):
        return

    # Cherche alarm.mp3
    base_path = os.path.dirname(os.path.abspath(__file__))
    alarm_path = os.path.join(base_path, "alarm.mp3")
    if not os.path.exists(alarm_path):
        alarm_path = os.path.join(base_path, "Projet_Parking_IA", "alarm.mp3")

    # ── Audio HTML base64 — se rejoue à chaque appel ──
    if os.path.exists(alarm_path):
        with open(alarm_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        # Timestamp unique → force le navigateur à rejouer même si même fichier
        import time
        unique_id = int(time.time() * 1000)

        st.markdown(f"""
        <audio id="alarm_{unique_id}" autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

    # ── Bannière visuelle ──
    if level == "CRITIQUE":
        st.markdown("""
        <div style='
            background:rgba(255,34,0,0.2); border:2px solid #ff2200;
            border-radius:10px; padding:16px 20px; text-align:center;
            font-family:"Bebas Neue",sans-serif; font-size:1.4rem;
            letter-spacing:3px; color:#ff6b6b;
            animation: pulse-red 1s infinite;
        '>
            🚨 ALERTE CRITIQUE — FEU DÉTECTÉ — ÉVACUATION IMMÉDIATE 🚨
        </div>
        <style>
        @keyframes pulse-red {
            0%,100% { box-shadow:0 0 0 rgba(255,34,0,0); }
            50%      { box-shadow:0 0 24px rgba(255,34,0,0.6); }
        }
        </style>
        """, unsafe_allow_html=True)

    elif level == "ALERTE":
        st.markdown("""
        <div style='
            background:rgba(255,140,0,0.15); border:2px solid #ff8c00;
            border-radius:10px; padding:14px 20px; text-align:center;
            font-family:"Bebas Neue",sans-serif; font-size:1.2rem;
            letter-spacing:2px; color:#ffbf40;
        '>
            ⚠️ ALERTE — FUMÉE DÉTECTÉE — VÉRIFICATION REQUISE ⚠️
        </div>
        """, unsafe_allow_html=True)
