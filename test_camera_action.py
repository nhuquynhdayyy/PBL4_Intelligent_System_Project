import cv2
from ultralytics import YOLO

def test_my_model():
    # 1. Nạp model "xịn" nhất sau 50 Epoch của bạn
    # Đảm bảo đường dẫn này đúng với thư mục train4 của bạn
    model_path = "runs/detect/train4/weights/best.pt"
    
    print(f"[INFO] Đang nạp mô hình từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[LỖI] Không tìm thấy file model. Hãy kiểm tra lại đường dẫn! \nChi tiết: {e}")
        return

    # 2. Mở Camera của Laptop (thường là ID 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam laptop.")
        return

    print("[BẮT ĐẦU] Nhấn 'q' để thoát cửa sổ nhận diện.")

    # Định nghĩa màu sắc cho 4 hành động để nhìn cho đẹp
    # face, hand-raising, sitting, standing
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Cho AI dự đoán trên khung hình camera
        # conf=0.5: Chỉ hiện những gì AI chắc chắn trên 50%
        results = model.predict(frame, conf=0.5, verbose=False)

        # 4. Vẽ kết quả lên màn hình
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ khung hình
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Lấy tên lớp (face, hand-raising, sitting, standing)
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # Vẽ hình chữ nhật quanh đối tượng
                color = colors[cls_id] if cls_id < len(colors) else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Ghi tên hành động và độ tự tin
                display_text = f"{label} {conf:.2f}"
                cv2.putText(frame, display_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 5. Hiển thị cửa sổ
        cv2.imshow("Kiem tra nhan dien hanh dong (PBL4)", frame)

        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng Camera.")

if __name__ == "__main__":
    test_my_model()