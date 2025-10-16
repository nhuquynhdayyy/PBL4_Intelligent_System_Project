# recognize_face_json.py
from deepface import DeepFace
import cv2, numpy as np, pickle, os, json, datetime
from scipy.spatial.distance import cosine

# --- 1️⃣ Load model và DB ---
print("[INFO] Loading Facenet model...")
model = DeepFace.build_model("Facenet")

with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

threshold = 0.4

# --- 2️⃣ Hàm nhận diện khuôn mặt ---
def recognize_face(frame):
    try:
        embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]['embedding']
    except:
        return []

    detections = []
    best_match, best_conf = None, 0

    for student_id, embeddings in face_db.items():
        for ref_emb in embeddings:
            dist = cosine(embedding, ref_emb)
            conf = max(0, 1 - dist / threshold)
            if conf > best_conf:
                best_conf = conf
                best_match = student_id

    detections.append({
        "student_id": best_match if best_conf >= 0.5 else "unknown",
        "confidence": round(best_conf, 2)
    })

    return detections

# --- 3️⃣ Nhận diện real-time ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ Không thể mở camera.")
    exit()

print("[INFO] Bắt đầu nhận diện (ESC để thoát)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (480, 360))
    detections = recognize_face(frame_small)

    # --- 4️⃣ Xuất JSON mỗi frame ---
    frame_json = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "detections": detections
    }
    print(json.dumps(frame_json, ensure_ascii=False))
    with open("face_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(frame_json, ensure_ascii=False) + "\n")

    # --- 5️⃣ Hiển thị lên màn hình ---
    if detections:
        d = detections[0]
        label = f"{d['student_id']} ({d['confidence']*100:.0f}%)"
        color = (0, 255, 0) if d["student_id"] != "unknown" else (0, 0, 255)
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
