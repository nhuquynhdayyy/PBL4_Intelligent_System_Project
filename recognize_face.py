# File: recognize_face.py
from deepface import DeepFace
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cosine

# --- CẤU HÌNH ---
MODEL_NAME = "Facenet512"
DB_PATH = "face_db.pkl"
THRESHOLD = 0.5 # Nới lỏng ngưỡng một chút (gốc là 0.4) để dễ nhận diện hơn

print("[INFO] Loading Facenet model...")
model = DeepFace.build_model(MODEL_NAME)

try:
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    print(f"[INFO] Đã tải {len(face_db)} ID từ database.")
except:
    face_db = {}
    print("[WARN] Chưa có database khuôn mặt.")

def recognize_face(frame):
    """
    Hàm nhận diện cũ: Dùng enforce_detection=False và so khớp Cosine.
    Input: Ảnh đã được resize to (Zoom Face).
    """
    try:
        # 1. Lấy embedding (enforce_detection=False là chìa khóa để không bị lỗi No Face)
        objs = DeepFace.represent(frame, model_name=MODEL_NAME, enforce_detection=False)
        emb = objs[0]['embedding']
    except Exception as e:
        # print(f"DeepFace Error: {e}")
        return "Unknown", 0.0

    best_match = "Unknown"
    best_conf = 0.0

    # 2. So sánh với DB
    for student_id, emb_list in face_db.items():
        for ref_emb in emb_list:
            dist = cosine(emb, ref_emb)
            conf = max(0, 1 - dist / 0.4) # 0.4 là ngưỡng gốc của Facenet
            
            if conf > best_conf:
                best_conf = conf
                best_match = student_id

    # 3. Lọc ngưỡng tin cậy
    if best_conf < THRESHOLD:
        return "Unknown", round(best_conf, 2)

    return best_match, round(best_conf, 2)