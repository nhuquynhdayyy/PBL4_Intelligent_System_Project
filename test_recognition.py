from deepface import DeepFace
import cv2, pickle, numpy as np
from scipy.spatial.distance import cosine

# Load face_db.pkl
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

model_name = "Facenet"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy embedding từ frame hiện tại
    try:
        result = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
        emb_test = np.array(result[0]["embedding"])
    except:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # So sánh với từng học sinh
    best_match = None
    best_score = 1.0
    for student, embeddings in face_db.items():
        for emb_ref in embeddings:
            score = cosine(emb_test, np.array(emb_ref))
            if score < best_score:
                best_score = score
                best_match = student

    if best_match and best_score < 0.5:  # ngưỡng nhận diện
        cv2.putText(frame, f"{best_match} ({best_score:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "Unknown", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
