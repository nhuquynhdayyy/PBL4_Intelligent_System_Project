# File: create_face_db.py (version TOI UU TOC DO)
import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# --- CAU HINH ---
DATASET_PATH = "datasets/faces"
DB_PATH = "face_db.pkl"
MODEL_NAME = "Facenet512"   # nen dung Facenet512 cho do chinh xac cao

print("[INFO] Dang tai model DeepFace...")
model = DeepFace.build_model(MODEL_NAME)

def preprocess_face_clahe(img):
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    final = cv2.filter2D(final, -1, kernel)
    return final

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print("Khong tim thay thu muc datasets/faces")
        exit()

    face_db = {}

    # CHI LAY THU MUC, BO FILE .pkl, .txt...
    students = [
        f for f in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, f))
    ]

    for student_id in tqdm(students, desc="Xu ly du lieu"):
        student_path = os.path.join(DATASET_PATH, student_id)

        embeddings = []

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # xu ly sang & net nhe
            processed_img = preprocess_face_clahe(img)

            try:
                # KHONG detect mat, lay truc tiep embedding
                res = DeepFace.represent(
                    img_path=processed_img,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )
                if res:
                    embeddings.append(res[0]["embedding"])
            except:
                pass

        if embeddings:
            face_db[student_id] = embeddings

    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

    print(f"\n[INFO] DA XONG! DB luu tai {DB_PATH}")
