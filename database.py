import mysql.connector
from datetime import datetime

# ── Configuration connexion MySQL ──
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",
    "database": "fireguard_ia",
    "charset":  "utf8mb4"      # ← AJOUTER cette ligne
}

def get_connection():
    """Retourne une connexion MySQL."""
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    """Vérifie la connexion au démarrage."""
    try:
        conn = get_connection()
        conn.close()
        return True, "✅ MySQL connecté"
    except Exception as e:
        return False, f"❌ MySQL erreur : {e}"

# ─────────────────────────────────────────────
# ALERTES
# ─────────────────────────────────────────────
def save_alerte(label, niveau, score, source, session_id):
    """Sauvegarde une alerte en base."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO alertes (date, heure, label, niveau, score, source, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            label, niveau, round(score, 3), source, session_id
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erreur save_alerte : {e}")
        return False

def get_all_alertes(limit=100):
    """Récupère les dernières alertes."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT date, heure, label, niveau, score, source, session_id
            FROM alertes
            ORDER BY id DESC
            LIMIT %s
        """, (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Erreur get_alertes : {e}")
        return []

def get_stats_globales():
    """Statistiques globales toutes sessions confondues."""
    try:
        conn = get_connection()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM alertes")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM alertes WHERE label='fire'")
        fire = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM alertes WHERE label='smoke'")
        smoke = c.fetchone()[0]

        c.execute("SELECT AVG(score) FROM alertes")
        avg_score = c.fetchone()[0] or 0

        c.execute("SELECT COUNT(DISTINCT session_id) FROM alertes")
        nb_sessions = c.fetchone()[0]

        conn.close()
        return {
            "total":       total,
            "fire":        fire,
            "smoke":       smoke,
            "avg_score":   round(float(avg_score), 3),
            "nb_sessions": nb_sessions
        }
    except Exception as e:
        print(f"❌ Erreur stats : {e}")
        return {"total": 0, "fire": 0, "smoke": 0, "avg_score": 0, "nb_sessions": 0}

def get_alertes_par_jour():
    """Nombre d'alertes par jour pour le graphique."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT date, COUNT(*) as nb
            FROM alertes
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """)
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Erreur alertes_par_jour : {e}")
        return []

# ─────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────
def save_session(session_id, fire, smoke, frames,
                 avg_fire, avg_smoke, kafka_sent, duree):
    """Sauvegarde le résumé d'une session."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO sessions
            (session_id, date, total_frames, fire_count, smoke_count,
             avg_conf_fire, avg_conf_smoke, kafka_sent, duree_sec)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frames, fire, smoke,
            round(avg_fire, 3), round(avg_smoke, 3),
            kafka_sent, duree
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erreur save_session : {e}")
        return False

def get_all_sessions():
    """Récupère toutes les sessions."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT session_id, date, total_frames,
                   fire_count, smoke_count,
                   avg_conf_fire, avg_conf_smoke,
                   kafka_sent, duree_sec
            FROM sessions
            ORDER BY id DESC
        """)
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Erreur get_sessions : {e}")
        return []

# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────
def save_metriques(session_id, precision_fire, precision_smoke,
                   recall_fire, recall_smoke, f1_fire, f1_smoke, map_global):
    """Sauvegarde les métriques YOLO d'une session."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO metriques
            (session_id, date, precision_fire, precision_smoke,
             recall_fire, recall_smoke, f1_fire, f1_smoke, map_global)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(precision_fire,  3), round(precision_smoke, 3),
            round(recall_fire,     3), round(recall_smoke,    3),
            round(f1_fire,         3), round(f1_smoke,        3),
            round(map_global,      3)
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erreur save_metriques : {e}")
        return False

# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────
def delete_all():
    """Vide toutes les tables."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM alertes")
        c.execute("DELETE FROM sessions")
        c.execute("DELETE FROM metriques")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erreur delete_all : {e}")
        return False