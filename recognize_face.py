# File: recognize_face.py
from deepface import DeepFace
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cosine

MODEL_NAME = "Facenet512"
DB_PATH = "face_db.pkl"

print("[INFO] Loading Facenet model...")
model = DeepFace.build_model(MODEL_NAME)

try:
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
except:
    face_db = {}

def recognize_face(frame):
    try:
        # Tăng cường độ sắc nét/tương phản nhẹ
        frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15) 
        # TỐI ƯU: YOLO đã cắt mặt rồi, nên detector_backend để 'skip' để chạy cực nhanh
        objs = DeepFace.represent(frame, model_name="Facenet512", enforce_detection=False, detector_backend='skip')
        if not objs: return "Unknown", 1.0
        emb = objs[0]['embedding']
    except:
        return "Unknown", 1.0

    best_match = "Unknown"
    min_dist = 1.0 

    for student_id, emb_list in face_db.items():
        for ref_emb in emb_list:
            dist = cosine(emb, ref_emb)
            if dist < min_dist:
                min_dist = dist
                best_match = student_id

    # Với Facenet512 và Cosine:
    # < 0.3: Rất giống (Chủ nhân)
    # 0.3 - 0.4: Khá giống
    # > 0.4: Người lạ
    return best_match, min_dist