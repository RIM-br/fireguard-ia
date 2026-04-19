import json
import cv2
from ultralytics import YOLO
from kafka import KafkaProducer

# 1. Connexion Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 2. Charger ton IA
import os
# On construit le chemin vers le fichier dans le même dossier que le script
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.pt')
model = YOLO(model_path)

# 3. Charger la vidéo
video_path = 'video-fire1.mp4'
cap = cv2.VideoCapture(video_path)

print("🚀 Analyse vidéo en cours... Appuie sur 'q' pour arrêter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Lancer l'IA sur l'image actuelle (on ignore les prédictions faibles)
    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        if len(r.boxes) > 0:  # Si on détecte quelque chose
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # Préparer le message
                alert_data = {
                    "statut": "ALERTE_DETECTION",
                    "objet": label,
                    "confiance": round(conf, 2),
                    "source": "Camera_Parking_01"
                }
                
                # Envoyer à Kafka
                producer.send('parking-alerts', value=alert_data)
                print(f"🔥 {label} détecté ({conf}) -> Message envoyé !")

    # Optionnel : Afficher la vidéo avec les détections
    annotated_frame = results[0].plot()
    cv2.imshow("IA Surveillance Parking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
producer.flush()