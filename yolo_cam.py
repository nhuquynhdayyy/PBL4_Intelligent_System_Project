# yolo_cam.py
import cv2, time
from ultralytics import YOLO

# Load model (đổi đường dẫn tới best.pt của bạn)
model = YOLO("runs/detect/train2/weights/best.pt")

# Mở camera
cap = cv2.VideoCapture(0)  # nếu USB cam khác: 1,2,...
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Kiểm tra thông số camera thực tế
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = cap.get(cv2.CAP_PROP_FPS)
print(f"📷 Camera info: {width:.0f}x{height:.0f} @ {fps:.1f} FPS")

# Biến cấu hình
t0 = time.time(); frames = 0
show_fps = True
conf_thres = 0.5

# Biến đếm sự kiện
hand_raise_count = 0
prev_detected = False
no_hand_frames = 0
gap_threshold = 10     # cần ≥10 frame không có tay để reset (giảm/ tăng tuỳ FPS)
warmup_frames = 10     # bỏ qua 10 frame đầu tiên
frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_id += 1

    # Dự đoán
    res = model.predict(source=frame, imgsz=640, conf=conf_thres,
                        device="cpu", verbose=False)[0]
    detected = len(res.boxes) > 0

    # Logic đếm (sau khi qua warmup)
    if frame_id > warmup_frames:
        if detected:
            if not prev_detected and no_hand_frames >= gap_threshold:
                hand_raise_count += 1
            no_hand_frames = 0
        else:
            no_hand_frames += 1
    prev_detected = detected

    # Vẽ bbox + label
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # FPS
    frames += 1
    if time.time()-t0 > 1:
        fps = frames / (time.time()-t0)
        t0 = time.time(); frames = 0
        if show_fps:
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị thông tin
    cv2.putText(frame, f"Conf:{conf_thres:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Count:{hand_raise_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Raised-Hand", frame)

    # Điều khiển phím
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord("q")]:   # ESC hoặc q thoát
        break
    elif key == ord("f"):
        show_fps = not show_fps
    elif key == 82:  # ↑
        conf_thres = min(1.0, conf_thres + 0.05)
    elif key == 84:  # ↓
        conf_thres = max(0.05, conf_thres - 0.05)

cap.release()
cv2.destroyAllWindows()

