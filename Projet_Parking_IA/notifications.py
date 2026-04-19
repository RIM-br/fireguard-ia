from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'parking-alerts',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("🔍 Système d'alerte en ligne. En attente de l'IA...")

for message in consumer:
    data = message.value
    print(f"⚠️ [NOTIFICATION] : {data['alerte']} ({data['objet']}) avec une confiance de {data['score']} à {data['heure']}")
    