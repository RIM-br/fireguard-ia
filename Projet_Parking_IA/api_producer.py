import json
import time
from ultralytics import YOLO
from kafka import KafkaProducer

# 1. Connexion au serveur Kafka lancé par Docker
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 2. Chargement de ton modèle IA
model = YOLO('best.pt')

def run_api_detection(image_path):
    # Lancement de l'IA sur l'image
    results = model(image_path, conf=0.5)
    
    for r in results:
        for box in r.boxes:
            # On prépare la donnée de sortie (API format)
            detection_data = {
                "alerte": "FEU_DETECTE",
                "score": round(float(box.conf[0]), 2),
                "objet": model.names[int(box.cls[0])],
                "heure": time.strftime("%H:%M:%S")
            }
            
            # ENVOI VERS KAFKA sur le sujet 'parking-alerts'
            producer.send('parking-alerts', value=detection_data)
            print(f"📡 Donnée envoyée à Kafka : {detection_data}")

# Test sur ton image (vérifie qu'elle s'appelle bien test.jpg dans ton dossier)
try:
    run_api_detection('test.png')
    producer.flush()
    print("✅ Envoi terminé avec succès !")
except Exception as e:
    print(f"❌ Erreur : {e}. Vérifie que test.jpg est bien dans le dossier.")