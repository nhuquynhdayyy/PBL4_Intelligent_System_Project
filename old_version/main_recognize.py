# main_recognize.py
import cv2
from deepface import DeepFace
# Dòng mới
from recognize_face_v2 import recognize_face

DETECTOR = "retinaface"
DETECT_CONF = 0.95

print("[INFO] Loading DeepFace...")
print("[INFO] Opening camera...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera cannot be opened!")
    exit()

print("[INFO] Camera opened successfully.")
print("[INFO] Face recognition started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Cannot read frame!")
        break

    cv2.imshow("Camera Only", frame)  # ← dòng TEST QUAN TRỌNG NHẤT
    cv2.waitKey(1)

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
    except Exception as e:
        print("DeepFace detect error:", e)
        faces = []

    for f in faces:
        if f["confidence"] < DETECT_CONF:
            continue

        x, y, w, h = f["facial_area"].values()
        crop = frame[y:y+h, x:x+w]

        student, conf = recognize_face(crop)

        color = (0,255,0) if student != "Unknown" else (0,0,255)
        label = f"{student} ({conf:.2f})" if student!="Unknown" else "Unknown"

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - Press Q", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
