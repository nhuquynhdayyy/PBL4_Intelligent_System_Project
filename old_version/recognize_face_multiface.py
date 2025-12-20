# File recognize_face_multiface.py 
# Nhận diện nhiều khuôn mặt với Facenet (sửa lỗi căn chỉnh)
from deepface import DeepFace
import cv2, pickle, numpy as np, time
from scipy.spatial.distance import cosine

print("[INFO] Loading Facenet model...")
model = DeepFace.build_model("Facenet")

with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

threshold = 0.4
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] Multi-face recognition running... (ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)

    for det in detections:
        fa = det["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
        face_crop = np.array(det["face"])
        face_crop = cv2.resize(face_crop, (160,160))  # fix kích thước Facenet

        emb_test = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]

        best_match, best_score = "unknown", 1.0
        for sid, emb_list in face_db.items():
            for emb_ref in emb_list:
                dist = cosine(emb_test, np.array(emb_ref))
                if dist < best_score:
                    best_score = dist
                    best_match = sid

        label = f"{best_match} ({best_score:.3f})" if best_score < threshold else "Unknown"
        color = (0,255,0) if best_score < threshold else (0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"[DEBUG] {best_match} → cosine = {best_score:.3f}")

    cv2.imshow("Face Recognition - MultiFace (Aligned Fix)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
