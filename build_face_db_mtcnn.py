# build_face_db_mtcnn.py
from deepface import DeepFace
import os, pickle, numpy as np, cv2
from tqdm import tqdm

# --- 1️⃣ Cấu hình ---
DATASET_DIR = "datasets"
OUTPUT_PATH = "face_db.pkl"
MODEL_NAME = "Facenet"

print(f"[INFO] Loading {MODEL_NAME} model...")
model = DeepFace.build_model(MODEL_NAME)
print("[INFO] Model loaded successfully.")

# --- 2️⃣ Khởi tạo DB ---
face_db = {}

# --- 3️⃣ Duyệt qua từng thư mục học sinh ---
for student_id in tqdm(os.listdir(DATASET_DIR), desc="[BUILDING DB]"):
    student_path = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(student_path):
        continue

    embeddings = []
    for img_file in os.listdir(student_path):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(student_path, img_file)

        try:
            # --- 4️⃣ Dò khuôn mặt bằng MTCNN ---
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend='mtcnn',
                enforce_detection=False
            )
            if len(faces) == 0:
                print(f"[WARN] No face found in {img_file}")
                continue

            face_crop = np.array(faces[0]["face"])
            face_crop = cv2.resize(face_crop, (160, 160))  # chuẩn kích thước Facenet

            # --- 5️⃣ Lấy embedding ---
            emb = DeepFace.represent(
                face_crop,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(emb)

        except Exception as e:
            print(f"[ERROR] {img_file}: {e}")
            continue

    if len(embeddings) > 0:
        face_db[student_id] = embeddings
        print(f"[OK] {student_id}: {len(embeddings)} embeddings saved.")
    else:
        print(f"[SKIP] {student_id}: No valid faces found.")

# --- 6️⃣ Lưu database ---
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(face_db, f)

print(f"\n✅ DONE! Face database saved to '{OUTPUT_PATH}'")
print(f"Total students: {len(face_db)}")
