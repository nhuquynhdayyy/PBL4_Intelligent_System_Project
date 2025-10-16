from deepface import DeepFace
import cv2

# 1. Nạp camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Nhận diện khuôn mặt (tìm ID gần nhất trong cơ sở dữ liệu)
    result = DeepFace.find(img_path = frame, db_path = "datasets/faces/",
                           model_name = "Facenet", detector_backend = "mtcnn")

    # 3. Hiển thị kết quả
    if len(result) > 0:
        name = result[0]['identity'].split("\\")[-2]  # tên thư mục
        cv2.putText(frame, name, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
