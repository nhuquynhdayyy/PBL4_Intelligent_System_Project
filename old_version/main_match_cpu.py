# main_match_cpu.py
import cv2, time, numpy as np
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face  

# --- Load 2 model ---
hand_model = YOLO("runs/detect/train2/weights/best.pt")   
person_model = YOLO("yolov8n.pt")  # model detect người, rất nhẹ

# --- Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Bộ nhớ theo dõi đơn giản ---
tracks = {}  # key=(x,y,w,h) gần nhất; value={'heights':deque(),'last_recog':float}

def match_hand_to_person(hand_boxes, person_boxes):
    pairs = []
    for hb in hand_boxes:
        hx = (hb[0]+hb[2])//2
        hy = (hb[1]+hb[3])//2
        best_i, best_d = None, 1e9
        for i, pb in enumerate(person_boxes):
            px = (pb[0]+pb[2])//2
            py = (pb[1]+pb[3])//2
            if pb[0] <= hx <= pb[2] and pb[1] <= hy <= pb[3]:
                best_i = i; break
            d = (hx-px)**2 + (hy-py)**2
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            pairs.append((hb, person_boxes[best_i]))
    return pairs

while True:
    ok, frame = cap.read()
    if not ok: break

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

    for pb in person_boxes:
        x1,y1,x2,y2 = pb
        h = y2 - y1
        cx,cy = (x1+x2)//2, (y1+y2)//2

        # cập nhật lịch sử chiều cao
        key = (cx//20, cy//20)  # làm tròn để tránh rung
        if key not in tracks:
            tracks[key] = {'heights': deque(maxlen=5), 'last_recog': 0, 'baseline': h}
        tracks[key]['heights'].append(h)

        # cập nhật baseline sau vài frame đầu
        if len(tracks[key]['heights']) == 5:
            tracks[key]['baseline'] = np.median(tracks[key]['heights'])

        baseline = tracks[key]['baseline']
        standing = h > 1.2 * baseline

        # Kiểm tra xem người này có tay match không
        matched = any(pb == pair[1] for pair in pairs)

        if matched and standing:
            now = time.time()
            if now - tracks[key]['last_recog'] > 5:
                # crop mặt (vùng top 35% của bbox)
                fy1, fy2 = y1, y1 + int(0.35 * (y2 - y1))
                face_crop = frame[fy1:fy2, x1:x2]
                if face_crop.size > 0:
                    student_id, conf = recognize_face(cv2.resize(face_crop, (480,360)))
                    print(f"[INFO] Giơ tay + Đứng lên: {student_id} ({conf:.2f})")
                    cv2.putText(frame, f"{student_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    tracks[key]['last_recog'] = now

        # vẽ khung người
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        if standing: 
            cv2.putText(frame, "STAND", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # vẽ khung tay
    for hb in hand_boxes:
        cv2.rectangle(frame, (hb[0],hb[1]), (hb[2],hb[3]), (255,0,0), 2)
        cv2.putText(frame, "HAND", (hb[0], hb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Hand + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
