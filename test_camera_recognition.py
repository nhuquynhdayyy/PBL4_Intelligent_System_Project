import cv2
from ultralytics import YOLO
import numpy as np
from recognize_face import recognize_face 

def test_combined_system():
    # 1. Nạp model YOLOv8 (Best từ train4)
    model_path = "runs/detect/train4/weights/best.pt"
    print(f"[INFO] Đang nạp YOLOv8 từ: {model_path}")
    yolo_model = YOLO(model_path)

    # 2. Định nghĩa bảng màu cho từng hành động (BGR)
    # Bạn có thể chỉnh lại màu ở đây theo ý thích
    color_map = {
        "hand-raising": (0, 255, 0),    # Xanh lá
        "sitting": (0, 255, 255),       # Vàng
        "standing": (255, 0, 255),      # Tím hồng
        "face_known": (255, 0, 0),      # Xanh dương (Khi nhận diện đúng tên)
        "face_unknown": (0, 0, 255)     # Đỏ (Khi không biết là ai)
    }

    # 3. Mở Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam.")
        return

    print("[BẮT ĐẦU] Hệ thống đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 4. YOLO phát hiện hành động
        results = yolo_model.predict(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                
                # Mặc định lấy màu theo tên lớp, nếu không có thì để màu trắng
                color = color_map.get(label, (255, 255, 255))
                display_name = label

                # 5. XỬ LÝ RIÊNG CHO LỚP "FACE"
                if label == "face":
                    face_crop = frame[max(0,y1):min(frame.shape[0],y2), 
                                      max(0,x1):min(frame.shape[1],x2)]

                    if face_crop.size > 0:
                        # Nhận diện danh tính qua Facenet512
                        name, face_conf = recognize_face(face_crop)

                        if name != "Unknown":
                            display_name = f"{name} ({face_conf:.2f})"
                            color = color_map["face_known"]
                        else:
                            display_name = "Unknown Person"
                            color = color_map["face_unknown"]

                # 6. Vẽ Bounding Box và Tên lên màn hình
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Tạo nền cho chữ để dễ đọc hơn
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(display_name)*18, y1), color, -1)
                cv2.putText(frame, display_name, (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hiển thị kết quả
        cv2.imshow("He thong nhan dien tich hop (PBL4)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_combined_system()