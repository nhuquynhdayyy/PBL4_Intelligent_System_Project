import cv2, time, csv
from ultralytics import YOLO

# ---- Config ----
model_path = "runs/detect/train2/weights/best.pt"
video_path = "classroom.mp4"   # đổi thành đường dẫn video lớp học
output_path = "output.mp4"
csv_log = "hand_raise_log.csv"
conf_thres = 0.25
gap_threshold = 10   # cần ≥10 frame không có tay để tính sự kiện mới

# ---- Load model ----
model = YOLO(model_path)

# ---- Open video ----
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writer để lưu video kết quả
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # hoặc 'avc1', 'MJPG'
out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))


# ---- Biến đếm ----
hand_raise_count = 0
prev_detected = False
no_hand_frames = 0

# Log CSV
with open(csv_log, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Time(s)", "Count"])

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_id += 1

        # Dự đoán
        res = model.predict(source=frame, imgsz=640, conf=conf_thres,
                            device="cpu", verbose=False)[0]
        detected = len(res.boxes) > 0

        # Logic đếm
        if detected:
            if not prev_detected and no_hand_frames >= gap_threshold:
                hand_raise_count += 1
                time_s = frame_id / fps
                writer.writerow([frame_id, f"{time_s:.2f}", hand_raise_count])
            no_hand_frames = 0
        else:
            no_hand_frames += 1
        prev_detected = detected

        # Vẽ bbox
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Hiển thị count
        cv2.putText(frame, f"Count:{hand_raise_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Ghi vào file video
        out.write(frame)

    cap.release()
    out.release()
