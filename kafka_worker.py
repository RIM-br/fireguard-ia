"""
kafka_worker.py — Worker YOLO + Kafka
======================================
Ce script tourne EN PARALLÈLE de Streamlit.
Il écoute 'parking-frames', fait la détection YOLO,
et publie les résultats sur 'parking-results'.

Lancement :
    python kafka_worker.py

Prérequis :
    - Docker lancé : docker-compose up -d
    - best.pt dans le même dossier
    - pip install kafka-python ultralytics opencv-python
"""

import json
import time
import os
import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BROKER           = 'localhost:9092'
TOPIC_IN         = 'parking-frames'    # Reçoit les frames JPEG depuis Streamlit
TOPIC_OUT        = 'parking-results'   # Publie les résultats YOLO vers Streamlit
TOPIC_ALERTS     = 'parking-alerts'    # Publie les alertes critiques
CONF_THRESHOLD   = 0.25
MODEL_PATH       = os.path.join(os.path.dirname(__file__), 'best.pt')

# ─────────────────────────────────────────────
# CHARGEMENT YOLO
# ─────────────────────────────────────────────
def load_model():
    try:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            print(f"[ERREUR] best.pt introuvable : {MODEL_PATH}")
            return None
        model = YOLO(MODEL_PATH)
        print(f"[OK] Modèle YOLOv8 chargé depuis {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[ERREUR] Chargement YOLO : {e}")
        return None

# ─────────────────────────────────────────────
# DÉTECTION
# ─────────────────────────────────────────────
def detect(model, frame_bytes):
    """
    Décode les bytes JPEG, lance YOLO, retourne label + score.
    """
    try:
        nparr  = np.frombuffer(frame_bytes, np.uint8)
        frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return "normal", 0.0

        results    = model(frame, conf=CONF_THRESHOLD, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                detections.append({"label": label, "score": score})

        if detections:
            best = max(detections, key=lambda d: d["score"])
            return best["label"], best["score"]

        return "normal", 0.0

    except Exception as e:
        print(f"[ERREUR] Détection : {e}")
        return "normal", 0.0

def classify_risk(label):
    label = label.lower().strip()
    if label == "fire":
        return "CRITIQUE"
    elif label == "smoke":
        return "ALERTE"
    return "NORMAL"

# ─────────────────────────────────────────────
# CONNEXION KAFKA
# ─────────────────────────────────────────────
def connect_kafka():
    print(f"[INFO] Connexion au broker Kafka : {BROKER}")
    retries = 0
    while retries < 10:
        try:
            consumer = KafkaConsumer(
                TOPIC_IN,
                bootstrap_servers=[BROKER],
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='yolo-worker-group',
                # Pas de value_deserializer → on reçoit des bytes bruts (JPEG)
            )
            producer = KafkaProducer(
                bootstrap_servers=[BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            print(f"[OK] Kafka connecté — écoute sur '{TOPIC_IN}'")
            return consumer, producer

        except NoBrokersAvailable:
            retries += 1
            print(f"[ATTENTE] Kafka pas encore prêt, tentative {retries}/10 dans 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"[ERREUR] Kafka : {e}")
            time.sleep(3)
            retries += 1

    print("[ERREUR] Impossible de se connecter à Kafka après 10 tentatives.")
    return None, None

# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  FireGuard IA — Kafka YOLO Worker")
    print("=" * 50)

    model = load_model()
    if model is None:
        print("[ERREUR] Impossible de charger le modèle. Arrêt.")
        return

    consumer, producer = connect_kafka()
    if consumer is None:
        print("[ERREUR] Impossible de connecter Kafka. Arrêt.")
        return

    print(f"[OK] Worker prêt — en attente de frames sur '{TOPIC_IN}'...")
    frame_count = 0

    try:
        for message in consumer:
            frame_bytes = message.value   # bytes JPEG bruts
            frame_idx   = message.key.decode() if message.key else str(frame_count)
            frame_count += 1

            t0 = time.time()
            label, score = detect(model, frame_bytes)
            level        = classify_risk(label)
            latency_ms   = round((time.time() - t0) * 1000, 1)

            # Publier le résultat vers Streamlit
            result = {
                "frame_idx": frame_idx,
                "label":     label,
                "score":     round(score, 3),
                "level":     level,
                "heure":     time.strftime("%H:%M:%S"),
                "latency_ms": latency_ms
            }
            producer.send(TOPIC_OUT, value=result)

            # Publier une alerte si critique ou alerte
            if level in ("CRITIQUE", "ALERTE"):
                alert = {
                    "alerte": level,
                    "objet":  label,
                    "score":  round(score, 2),
                    "heure":  time.strftime("%H:%M:%S"),
                    "source": "KafkaWorker"
                }
                producer.send(TOPIC_ALERTS, value=alert)
                print(f"[🚨 ALERTE] Frame {frame_idx} — {label.upper()} ({score:.2f}) — {latency_ms}ms")
            else:
                print(f"[✅ OK]     Frame {frame_idx} — {label} ({score:.2f}) — {latency_ms}ms")

            producer.flush()

    except KeyboardInterrupt:
        print(f"\n[INFO] Worker arrêté. {frame_count} frames traitées.")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    main()
