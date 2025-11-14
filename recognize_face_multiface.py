# from deepface import DeepFace
# import cv2, numpy as np, pickle, time
# from scipy.spatial.distance import cosine

# # --- 1️⃣ Load model & DB ---
# print("[INFO] Loading Facenet model...")
# model = DeepFace.build_model("Facenet")

# with open("face_db.pkl", "rb") as f:
#     face_db = pickle.load(f)

# threshold = 0.4  # Ngưỡng nhận diện

# # --- 2️⃣ Hàm so khớp khuôn mặt ---
# def match_face(face_img):
#     try:
#         emb = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]['embedding']
#     except:
#         return "unknown", 0.0

#     best_match, best_conf = "unknown", 0
#     for student_id, emb_list in face_db.items():
#         for ref_emb in emb_list:
#             dist = cosine(emb, ref_emb)
#             conf = max(0, 1 - dist / threshold)
#             if conf > best_conf:
#                 best_conf = conf
#                 best_match = student_id
#     if best_conf < 0.5:
#         best_match = "unknown"
#     return best_match, round(best_conf, 2)

# # --- 3️⃣ Mở camera ---
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("❌ Không thể mở camera.")
#     exit()

# print("[INFO] Bắt đầu nhận diện nhiều khuôn mặt... (ESC để thoát)")

# # --- 4️⃣ Vòng lặp nhận diện ---
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # --- 5️⃣ Dò tất cả khuôn mặt ---
#     detections = DeepFace.extract_faces(
#         img_path=frame,
#         detector_backend='retinaface',  # có thể đổi sang 'mtcnn' nếu muốn nhẹ hơn
#         enforce_detection=False
#     )

#     results = []
#     for det in detections:
#         fa = det["facial_area"]
#         x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

#         face_crop = np.array(det["face"])
#         student_id, conf = match_face(face_crop)

#         results.append({
#             "student_id": student_id,
#             "confidence": conf,
#             "box": (x, y, w, h)
#         })

#         # Vẽ bounding box
#         color = (0, 255, 0) if student_id != "unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(frame, f"{student_id} ({conf*100:.0f}%)",
#                     (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     # --- 6️⃣ Hiển thị & in JSON ---
#     cv2.imshow("Face Recognition - MultiFace", frame)
#     if results:
#         print({
#             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
#             "detections": results
#         })

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()

## Nhận diện nhiều khuôn mặt với Facenet (sửa lỗi căn chỉnh)
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
