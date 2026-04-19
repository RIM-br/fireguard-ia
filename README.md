# 🔥 FireGuard IA — Détection d'Incendie par Vision par Ordinateur

> Système de détection précoce de feux, flammes et fumée en temps réel  
> **Projet M1 S2 · 2025–2026 · Encadrant : Pr. Outhman MAAROUF**

---

## 👥 Équipe

| Membre | Rôle |
|--------|------|
| **Asmae AALIKANE** | Data Scientist · Prétraitement & Annotation |
| **Assietou ABDERRAHMANE** | Data Scientist · Entraînement Modèle YOLOv8 |
| **Rim BARI** | Développeuse Full-Stack · UI/UX & Intégration |

---

## 📋 Description

FireGuard IA est un système de surveillance intelligente capable de détecter automatiquement la présence de **feu** et de **fumée** dans des flux vidéo en temps réel.  
Grâce au modèle **YOLOv8** entraîné sur un dataset spécialisé, le système génère des alertes immédiates (sonores + visuelles) et envoie les données vers un broker **Kafka** (Docker).

---

## 🏗️ Architecture du Projet

```
PROJET-VISION-IA/
│
├── app3.py                  # Application principale Streamlit (intégration complète)
├── alarm.py                 # Déclenchement alarme sonore + visuelle
├── detection.py             # (ancien) Détection simulée random — remplacée par YOLO réel
├── best.pt                  # Modèle YOLOv8 entraîné (fire + smoke)
├── alarm.mp3                # Fichier audio alarme
├── check_model.py           # Script diagnostic du modèle
├── requirements.txt         # Dépendances Python
│
└── Projet_Parking_IA/
    ├── api_producer.py      # Producer Kafka — détection sur image
    ├── api_video_producer.py# Producer Kafka — détection sur vidéo
    ├── notifications.py     # Consumer Kafka — affichage alertes
    ├── docker-compose.yml   # Configuration Kafka + Zookeeper
    ├── best.pt              # Modèle (copie)
    ├── test.png             # Image de test
    └── video-fire1.mp4      # Vidéo de test
```

---

## 🧠 Classes Détectées

| Classe | Description | Niveau |
|--------|-------------|--------|
| 🔥 `fire` | Flammes actives visibles | CRITIQUE |
| 💨 `smoke` | Panaches de fumée | ALERTE |

---

## ⚙️ Technologies Utilisées

- **YOLOv8** (Ultralytics) — Détection en temps réel
- **Python 3.11**
- **Streamlit** — Interface web interactive
- **OpenCV** — Traitement vidéo/image
- **Apache Kafka** (via Docker) — Broker de messages
- **Docker** — Conteneurisation Kafka + Zookeeper
- **Matplotlib / Pandas / NumPy** — Visualisation & métriques

---

## 🚀 Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/fireguard-ia.git
cd fireguard-ia
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer Kafka (Docker) — optionnel

```bash
cd Projet_Parking_IA
docker-compose up -d
cd ..
```

### 4. Lancer l'application

```bash
# Depuis le dossier racine (là où se trouve best.pt)
streamlit run app3.py
```

> ⚠️ **Important** : `app3.py` et `best.pt` doivent être dans le **même dossier**.

---

## 🖥️ Pages de l'Interface

| Page | Description |
|------|-------------|
| 🏠 **Accueil & Équipe** | Présentation du projet, équipe, planning |
| 🎥 **Détection Vidéo** | Upload vidéo, flux caméra live, image unique |
| 🚨 **Alertes en Direct** | Flux Kafka live + historique session |
| 📊 **Dashboard** | Métriques temps réel (F1, Précision, Rappel, Confiance) |

---

## 🔌 Modes de Détection

| Mode | Condition | Description |
|------|-----------|-------------|
| 🤖 **YOLO Direct** | `best.pt` présent | Détection locale instantanée |
| 📡 **Via Kafka** | Docker lancé | Envoi vers broker + consumer |
| 🔀 **Hybride** | Les deux actifs | YOLO + envoi Kafka en parallèle |

---

## 📊 Performances du Modèle

| Métrique | Fire | Smoke | Global |
|----------|------|-------|--------|
| Précision | 90% | 82% | 85% |
| Rappel | 88% | 80% | 83.5% |
| F1-Score | 89% | 81% | 84.2% |
| mAP@0.5 | — | — | **86.5%** |

> Seuil de confiance recommandé : **0.25 – 0.35** (adapté au modèle entraîné)

---

## 🐛 Difficultés Rencontrées & Solutions

### 1. 🔴 Modèle ne détectait rien (score 0.00)
**Problème** : Le seuil de confiance par défaut était `0.45` mais le modèle génère des scores entre `0.12` et `0.45`.  
**Solution** : Abaissement du seuil à `0.25` après diagnostic avec `check_model.py`.

### 2. 🔴 Noms de classes incorrects dans `classify_risk()`
**Problème** : La fonction cherchait `"flame"`, `"flamme"`, `"braise"` mais le modèle retourne uniquement `"fire"` et `"smoke"`.  
**Solution** : Réécriture de `classify_risk()` avec les vrais noms de classes vérifiés via diagnostic.

```python
# Avant (incorrect)
if "fire" in label or "flame" in label or "flamme" in label:

# Après (correct)
if label == "fire":
```

### 3. 🔴 Alarme ne se déclenchait qu'une seule fois
**Problème** : `st.session_state.alarm_played = True` bloquait tous les déclenchements suivants.  
**Solution** : Suppression du flag bloquant + injection audio HTML base64 avec `unique_id` timestamp pour forcer le navigateur à rejouer à chaque détection.

```python
# Clé unique à chaque appel = le navigateur rejoue à chaque fois
unique_id = int(time.time() * 1000)
st.markdown(f'<audio id="alarm_{unique_id}" autoplay>...</audio>')
```

### 4. 🔴 Bouton STOP ne fonctionnait pas pendant la vidéo
**Problème** : La boucle `while cap.isOpened()` bloquait l'interface Streamlit, rendant le bouton STOP inutilisable.  
**Solution** : Utilisation de `st.session_state["video_running"]` / `st.session_state["cam_running"]` comme flag contrôlé par les boutons.

```python
while cap.isOpened() and st.session_state.get("video_running", False):
    ...
    if not st.session_state.get("video_running", False):
        break
```

### 5. 🔴 Kafka non disponible bloquait l'application
**Problème** : Si Docker n'était pas lancé, `KafkaConsumer()` levait une exception et plantait tout.  
**Solution** : Wrapping dans `try/except NoBrokersAvailable` + fallback gracieux avec indicateur visuel.

```python
try:
    consumer = KafkaConsumer(...)
    return consumer, "✅ Kafka connecté"
except NoBrokersAvailable:
    return None, "❌ Kafka non disponible"
```

### 6. 🔴 `best.pt` introuvable selon le répertoire de lancement
**Problème** : Selon d'où on lançait `streamlit run`, le chemin vers `best.pt` changeait.  
**Solution** : Utilisation de `os.path.dirname(__file__)` pour construire le chemin absolu.

```python
base_path  = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.pt')
```

### 7. 🔴 Dashboard métriques statique
**Problème** : Les métriques (F1, Précision, Rappel) étaient des valeurs fixes codées en dur.  
**Solution** : Calcul dynamique depuis `st.session_state.metrics` alimenté à chaque détection YOLO.

### 8. 🔴 Indentation incorrecte dans le Tab Image
**Problème** : Le bouton "Lancer la détection" et le bloc `if image_analyzed` étaient hors du bon niveau d'indentation, causant des `NameError` (`img_frame` non défini).  
**Solution** : Restructuration complète du bloc `with tab3:` avec indentation correcte.

---

## 📁 Fichiers à NE PAS committer (`.gitignore`)

```
__pycache__/
*.pyc
best.pt
alarm.mp3
video-fire1.mp4
test.png
*.mp4
*.avi
.env
```

> ⚠️ `best.pt` est trop lourd pour GitHub (>25MB). Hébergez-le sur **Google Drive** ou **Hugging Face** et ajoutez le lien dans le README.

---

## 📥 Télécharger le Modèle

Le fichier `best.pt` n'est pas inclus dans ce dépôt en raison de sa taille.  
👉 **[Télécharger best.pt — Google Drive](#)** *(remplacez ce lien)*

Placez-le à la **racine du projet** (même dossier que `app3.py`).

---

## 🐳 Docker — Kafka

```yaml
# docker-compose.yml (dans Projet_Parking_IA/)
# Lance Zookeeper + Kafka sur localhost:9092
# Topic utilisé : parking-alerts
```

```bash
# Démarrer
docker-compose up -d

# Vérifier
docker ps

# Arrêter
docker-compose down
```

---

## 📜 Requirements

```
streamlit
ultralytics
opencv-python
kafka-python
numpy
pandas
matplotlib
```

```bash
pip install -r requirements.txt
```

---

## 📌 Comment contribuer / push sur GitHub

```bash
# 1. Initialiser git
git init

# 2. Ajouter les fichiers
git add app3.py alarm.py detection.py check_model.py requirements.txt README.md
git add Projet_Parking_IA/api_producer.py
git add Projet_Parking_IA/api_video_producer.py
git add Projet_Parking_IA/notifications.py
git add Projet_Parking_IA/docker-compose.yml

# 3. Premier commit
git commit -m "🔥 FireGuard IA — Intégration complète YOLO + Kafka + Streamlit"

# 4. Lier au dépôt GitHub
git remote add origin https://github.com/votre-username/fireguard-ia.git

# 5. Push
git push -u origin main
```

---

## 📄 Licence

Projet académique — M1 Intelligence Artificielle · 2025–2026
