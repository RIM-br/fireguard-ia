"""
check_model.py — Diagnostic FireGuard IA
Lance : python check_model.py
"""
import os
from ultralytics import YOLO

# ── 1. Chargement ──
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.pt')

print("=" * 50)
print("🔍 DIAGNOSTIC FIREGUARD IA — best.pt")
print("=" * 50)

if not os.path.exists(model_path):
    print("❌ best.pt introuvable ! Vérifiez le dossier.")
    exit()

model = YOLO(model_path)

# ── 2. Classes du modèle ──
print("\n✅ Modèle chargé avec succès !")
print(f"\n📋 Classes détectées par best.pt ({len(model.names)}) :")
for idx, name in model.names.items():
    print(f"   [{idx}] → {name}")

# ── 3. Test sur test.png ──
test_img = os.path.join(base_path, 'Projet_Parking_IA', 'test.png')
if not os.path.exists(test_img):
    test_img = os.path.join(base_path, 'test.png')

if os.path.exists(test_img):
    print(f"\n🖼️  Test sur : {test_img}")
    # Seuil très bas pour voir ce qui est détecté
    for seuil in [0.1, 0.2, 0.3, 0.5]:
        results = model(test_img, conf=seuil, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({
                    "classe": model.names[int(box.cls[0])],
                    "score":  round(float(box.conf[0]), 3)
                })
        print(f"   Seuil {seuil:.1f} → {len(dets)} détection(s) : {dets}")
else:
    print("\n⚠️  test.png non trouvé — testez manuellement avec une image.")

print("\n" + "=" * 50)
print("💡 Copiez les noms de classes ci-dessus et envoyez-les")
print("   pour que app3.py soit corrigé automatiquement.")
print("=" * 50)
