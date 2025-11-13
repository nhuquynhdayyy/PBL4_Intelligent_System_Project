# file: hand_face_match.py  (skeleton)
import cv2, time, numpy as np
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face     # dùng hàm bạn có sẵn

# models
hand_model = YOLO("runs/detect/train2/weights/best.pt")   # bạn đang dùng
person_model = YOLO("yolov8n.pt")  # pretrained để detect 'person'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# simple tracker storage: id -> bbox, history heights
next_id = 0
tracks = {}  # id: {bbox, last_seen_frame, heights deque, last_recog_time}

FRAME = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    FRAME += 1

    # 1) detect hands
    res_hand = hand_model.predict(source=frame, imgsz=640, conf=0.5, device='cpu', verbose=False)[0]
    hands = []
    for box in res_hand.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        hands.append(((x1,y1,x2,y2), float(box.conf[0])))

    # 2) detect persons
    res_person = person_model.predict(source=frame, imgsz=640, conf=0.3, device='cpu', verbose=False)[0]
    persons = []
    for box in res_person.boxes:
        cls = int(box.cls[0])
        if person_model.names[cls] != 'person': continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        persons.append((x1,y1,x2,y2))

    # 3) naive match: for each hand, find person whose bbox contains hand centroid or nearest centroid
    matched = {}  # hand_idx -> person_idx
    for hi, (hb, hc) in enumerate(hands):
        hx = (hb[0]+hb[2])//2; hy = (hb[1]+hb[3])//2
        best_i = None; best_dist = 1e9
        for pi, pb in enumerate(persons):
            px = (pb[0]+pb[2])//2; py = (pb[1]+pb[3])//2
            # if hand centroid inside person bbox -> match immediately
            if pb[0] <= hx <= pb[2] and pb[1] <= hy <= pb[3]:
                best_i = pi; break
            d = (hx-px)**2 + (hy-py)**2
            if d < best_dist:
                best_dist = d; best_i = pi
        matched[hi] = best_i

    # 4) update simple per-person state & detect "stand-up"
    for pi, pb in enumerate(persons):
        x1,y1,x2,y2 = pb
        h = y2-y1
        # find track (use nearest by centroid). Simplify: create new track for each person (production: use SORT)
        # For demo: use pb tuple as track key
        key = (x1,y1,x2,y2)
        if key not in tracks:
            tracks[key] = {'heights': deque(maxlen=10), 'last_recog': 0, 'last_seen': FRAME}
        tracks[key]['heights'].append(h)
        tracks[key]['last_seen'] = FRAME

        # check if any hand matched to this person
        hand_assigned = any(matched[hi]==pi for hi in matched)
        mean_h = np.mean(tracks[key]['heights'])
        baseline = mean_h  # in prod you keep baseline from earlier frames

        # If hand assigned AND current height > 1.15 * median(previous few frames) -> treat as stand+recognize
        if hand_assigned and h > 1.15 * np.median(list(tracks[key]['heights'])):
            # throttle recognition (1 per 5s)
            if time.time() - tracks[key]['last_recog'] > 5:
                # crop face region (top 35% of person bbox)
                fy1 = y1; fy2 = y1 + int(0.35 * (y2-y1))
                face_crop = frame[fy1:fy2, x1:x2]
                student_id, conf = recognize_face(cv2.resize(face_crop, (480,360)))
                print("Recognized:", student_id, conf)
                tracks[key]['last_recog'] = time.time()

    # Rendering + exit
    for hb,conf in hands:
        cv2.rectangle(frame, (hb[0],hb[1]), (hb[2],hb[3]), (255,0,0), 2)
    for pb in persons:
        cv2.rectangle(frame, (pb[0],pb[1]), (pb[2],pb[3]), (0,255,0), 2)
    cv2.imshow("Match", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release(); cv2.destroyAllWindows()
