# yolo_cam_fixed.py
import cv2, time
from ultralytics import YOLO

# --- 1️⃣ Load model YOLO ---
model = YOLO("runs/detect/train2/weights/best.pt")  # Đổi đường dẫn tới model của bạn

# --- 2️⃣ Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = cap.get(cv2.CAP_PROP_FPS)
print(f"📷 Camera info: {width:.0f}x{height:.0f} @ {fps:.1f} FPS")

# --- 3️⃣ Cấu hình ---
conf_thres = 0.5
show_fps = True
hand_raise_count = 0

# --- 4️⃣ FPS counter ---
t0 = time.time()
frames = 0

# --- 5️⃣ Màu cho từng nhãn ---
COLOR = {
    "hand": (0, 255, 0),        # xanh lá
    "maybehand": (0, 165, 255), # cam
    "nohand": (0, 0, 255),      # đỏ
}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- 6️⃣ Chạy YOLO ---
    res = model.predict(source=frame, imgsz=640, conf=conf_thres,
                        device="cpu", verbose=False)[0]

    # --- 7️⃣ Lưu nhãn trong frame ---
    labels_in_frame = set()

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Vẽ bbox với màu theo nhãn
        color = COLOR.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Thêm nhãn vào tập hợp frame hiện tại
        labels_in_frame.add(label)

    # --- 8️⃣ Logic đếm tay thật ---
    if "hand" in labels_in_frame:
        hand_raise_count += 1   # chỉ tăng khi có tay thật
        print(f"[INFO] Hand detected! Count = {hand_raise_count}")

    # --- 9️⃣ FPS tính mỗi giây ---
    frames += 1
    if time.time() - t0 > 1:
        fps_val = frames / (time.time() - t0)
        frames = 0
        t0 = time.time()
        if show_fps:
            cv2.putText(frame, f"FPS:{fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 🔟 Hiển thị thông tin ---
    cv2.putText(frame, f"Conf:{conf_thres:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Count:{hand_raise_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Raised-Hand (Fixed)", frame)

    # --- 11️⃣ Điều khiển bàn phím ---
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord("q")]:   # ESC hoặc q thoát
        break
    elif key == ord("f"):       # bật/tắt FPS
        show_fps = not show_fps
    elif key == 82:             # ↑ tăng ngưỡng
        conf_thres = min(1.0, conf_thres + 0.05)
    elif key == 84:             # ↓ giảm ngưỡng
        conf_thres = max(0.05, conf_thres - 0.05)

cap.release()
cv2.destroyAllWindows()
