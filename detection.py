import cv2
import requests
import random

def get_prediction(frame):
    return {
        "label": random.choice(["fire", "smoke", "normal"])
    }

#def get_prediction(frame):
    #url = "http://127.0.0.1:8000/predict"
    
    #_, img_encoded = cv2.imencode('.jpg', frame)
    #files = {"file": img_encoded.tobytes()}
    
    #response = requests.post(url, files=files)
    #return response.json()

def classify_risk(label):
    if label == "smoke":
        return "ALERTE", "level-medium"
    elif label == "fire":
        return "CRITIQUE", "level-high"
    else:
        return "NORMAL", "level-low"