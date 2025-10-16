from deepface import DeepFace
import cv2

model_name = "Facenet"
cap = cv2.VideoCapture(0)
print("[INFO] Nhấn SPACE để chụp, ESC để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        cv2.imwrite("test_capture.jpg", frame)
        print("Đã chụp ảnh test_capture.jpg")
        result = DeepFace.verify(img1_path="datasets/faces/003_HoNguyenThaoNguyen/003_HoNguyenThaoNguyen_1.jpg",
                                 img2_path="test_capture.jpg", model_name=model_name)
        print(result)
        if result["verified"]:
            print("✅ Nhận diện đúng người!")
        else:
            print("❌ Không khớp.")
cap.release()
cv2.destroyAllWindows()
