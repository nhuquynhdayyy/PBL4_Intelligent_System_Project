import cv2, time, numpy as np
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face  # dùng hàm có sẵn

# --- Load model ---
hand_model = YOLO("runs/detect/train2/weights/best.pt")   # model phát hiện tay
person_model = YOLO("yolov8n.pt")  # model detect người (pretrained)

# --- Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Bộ nhớ theo dõi ---
tracks = {}  # key=(cx,cy) -> {'heights','baseline','stand_frames','time_stand_start','last_recog'}

def match_hand_to_person(hand_boxes, person_boxes):
    pairs = []
    for hb in hand_boxes:
        hx = (hb[0]+hb[2])//2
        hy = (hb[1]+hb[3])//2
        best_i, best_d = None, 1e9
        for i, pb in enumerate(person_boxes):
            px = (pb[0]+pb[2])//2
            py = (pb[1]+pb[3])//2
            # Nếu tay nằm trong người thì match ngay
            if pb[0] <= hx <= pb[2] and pb[1] <= hy <= pb[3]:
                best_i = i
                break
            d = (hx-px)**2 + (hy-py)**2
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            pairs.append((hb, person_boxes[best_i]))
    return pairs

print("[INFO] Bắt đầu theo dõi... (Nhấn ESC để thoát)")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- Detect hand ---
    res_hand = hand_model.predict(frame, imgsz=640, conf=0.5, device='cpu', verbose=False)[0]
    hand_boxes = [list(map(int, box.xyxy[0].cpu().numpy())) for box in res_hand.boxes]

    # --- Detect person ---
    res_person = person_model.predict(frame, imgsz=640, conf=0.3, device='cpu', verbose=False)[0]
    person_boxes = []
    for box in res_person.boxes:
        cls = int(box.cls[0])
        if person_model.names[cls] == "person":
            person_boxes.append(list(map(int, box.xyxy[0].cpu().numpy())))

    # --- Match hand ↔ person ---
    pairs = match_hand_to_person(hand_boxes, person_boxes)

    # --- Xử lý từng người ---
    for pb in person_boxes:
        x1, y1, x2, y2 = pb
        h = y2 - y1
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        key = (cx//20, cy//20)

        if key not in tracks:
            tracks[key] = {
                'heights': deque(maxlen=5),
                'baseline': h,
                'stand_frames': 0,
                'time_stand_start': None,
                'last_recog': 0
            }

        # Cập nhật baseline chiều cao
        tracks[key]['heights'].append(h)
        if len(tracks[key]['heights']) == 5:
            tracks[key]['baseline'] = np.median(tracks[key]['heights'])

        baseline = tracks[key]['baseline']
        standing = h > 1.1 * baseline  # ngưỡng nhẹ hơn

        # Đếm số frame và thời điểm đứng
        if standing:
            if tracks[key]['time_stand_start'] is None:
                tracks[key]['time_stand_start'] = time.time()
            tracks[key]['stand_frames'] += 1
        else:
            tracks[key]['stand_frames'] = 0
            tracks[key]['time_stand_start'] = None

        elapsed = (time.time() - tracks[key]['time_stand_start']) if tracks[key]['time_stand_start'] else 0
        print(f"[DEBUG] h={h:.1f}, baseline={baseline:.1f}, stand_frames={tracks[key]['stand_frames']}, elapsed={elapsed:.1f}s")

        # Kiểm tra có tay match không
        matched = any(pb == pair[1] for pair in pairs)

        # Nếu đứng ≥1.5s + có tay → nhận diện
        if matched and elapsed >= 5:
            now = time.time()
            if now - tracks[key]['last_recog'] > 5:
                # --- Crop cận khuôn mặt ---
                fy1 = y1 + int(0.05 * (y2 - y1))
                fy2 = y1 + int(0.35 * (y2 - y1))
                fx1 = x1 + int(0.15 * (x2 - x1))
                fx2 = x2 - int(0.15 * (x2 - x1))

                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size > 0:
                    zoom_face = cv2.resize(face_crop, (480, 480))
                    cv2.imshow("Zoom Face", zoom_face)

                    student_id, conf = recognize_face(zoom_face)
                    print(f"[INFO] Giơ tay + Đứng ổn định: {student_id} ({conf:.2f})")
                    cv2.putText(frame, f"{student_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracks[key]['last_recog'] = now

        # --- Vẽ khung người ---
        color = (0, 255, 255) if standing else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if standing:
            cv2.putText(frame, "STAND", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- Vẽ khung tay ---
    for hb in hand_boxes:
        cv2.rectangle(frame, (hb[0], hb[1]), (hb[2], hb[3]), (255, 0, 0), 2)
        cv2.putText(frame, "HAND", (hb[0], hb[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Hand + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
