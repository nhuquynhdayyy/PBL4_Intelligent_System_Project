# File: pc_ai_server_integrated.py
# Phiên bản: Tối ưu hiệu năng + Tích hợp ArcFace mới

import asyncio
import websockets
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import requests
import threading

# --- IMPORT BỘ NHẬN DIỆN MỚI ---
from recognize_face_arcface import recognize_face 

# --- CẤU HÌNH ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8765
BACKEND_API_URL = "http://127.0.0.1:5000/api/recognize"
VIDEO_PUSH_URL = "http://127.0.0.1:5000/api/video_stream/push"
LOG_DEBOUNCE_SECONDS = 5
REQUEST_TIMEOUT = 1.0 

# --- LOAD MODEL YOLO ---
print("[INFO] Loading YOLO model...")
person_model = YOLO("yolov8n.pt")
print("[INFO] YOLO loaded.")

# --- BỘ NHỚ ---
tracks = {}
last_logged_time = {}

# --- HÀM GỬI REQUEST KHÔNG GÂY LAG ---
def send_request_in_thread(url, data=None, json_data=None, headers=None):
    def send():
        try:
            requests.post(url, data=data, json=json_data, headers=headers, timeout=REQUEST_TIMEOUT)
        except: pass
    threading.Thread(target=send).start()

async def ai_processing_handler(websocket):
    print(f"[INFO] Kết nối mới: {websocket.remote_address}")
    try:
        async for message in websocket:
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue

            # 1. Phát hiện người
            res_person = person_model.predict(frame, imgsz=320, conf=0.4, device='cpu', verbose=False)[0]
            person_boxes = [list(map(int, box.xyxy[0].cpu().numpy())) for box in res_person.boxes if int(box.cls[0]) == 0]

            for pb in person_boxes:
                x1, y1, x2, y2 = pb
                h, w = y2 - y1, x2 - x1
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                key = (cx//20, cy//20)

                if key not in tracks: tracks[key] = {'heights': deque(maxlen=5), 'baseline': h, 'stand_frames': 0, 'time_stand_start': None, 'last_recog': 0, 'confirmed_stand': False, 'time_confirmed': 0}
                
                tracks[key]['heights'].append(h)
                baseline = tracks[key]['baseline']
                standing = h > 1.15 * baseline
                if len(tracks[key]['heights']) == 5 and not standing: tracks[key]['baseline'] = np.median(tracks[key]['heights'])

                if standing:
                    if tracks[key]['time_stand_start'] is None: tracks[key]['time_stand_start'] = time.time()
                else:
                    tracks[key]['time_stand_start'] = None; tracks[key]['confirmed_stand'] = False
                
                elapsed = (time.time() - tracks[key]['time_stand_start']) if tracks[key]['time_stand_start'] else 0

                if standing and elapsed >= 1.0:
                    if not tracks[key]['confirmed_stand']: tracks[key]['time_confirmed'] = time.time()
                    tracks[key]['confirmed_stand'] = True
                
                # 2. Nếu đứng, tiến hành nhận diện
                if tracks[key]['confirmed_stand']:
                    now = time.time()
                    stable_time = now - tracks[key].get('time_confirmed', 0)
                    if stable_time >= 1.0 and now - tracks[key]['last_recog'] > 3: # Giảm thời gian chờ xuống 3s
                        # Cắt vùng mặt (ước lượng)
                        fy1, fy2 = y1 + int(0.05 * h), y1 + int(0.45 * h)
                        fx1, fx2 = x1 + int(0.10 * w), x2 - int(0.10 * w)
                        face_crop = frame[fy1:fy2, fx1:fx2]

                        if face_crop.size > 0:
                            # GỌI HÀM NHẬN DIỆN MỚI (ARCFACE)
                            student_id, conf = recognize_face(face_crop)
                            
                            # Ngưỡng tin cậy cho ArcFace (Cosine Similarity)
                            # 0.4 - 0.5 là mức trung bình khá
                            if student_id and student_id != "Unknown" and conf >= 0.4: 
                                label = f"{student_id} ({conf:.2f})"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                current_time = time.time()
                                if (student_id not in last_logged_time or current_time - last_logged_time.get(student_id, 0) > LOG_DEBOUNCE_SECONDS):
                                    print(f"[RECOGNIZED] {student_id} ({conf:.2f})")
                                    send_request_in_thread(url=BACKEND_API_URL, json_data={"student_code": student_id, "confidence": float(conf)})
                                    last_logged_time[student_id] = current_time
                            
                            tracks[key]['last_recog'] = now
                
                color = (0, 255, 255) if tracks[key]['confirmed_stand'] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 3. Gửi video stream
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                send_request_in_thread(url=VIDEO_PUSH_URL, data=buffer.tobytes(), headers={'Content-Type': 'image/jpeg'})
            
    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Ngắt kết nối.")

async def main():
    print(f"[INFO] Server AI (ArcFace) chạy tại ws://{SERVER_HOST}:{SERVER_PORT}")
    async with websockets.serve(ai_processing_handler, SERVER_HOST, SERVER_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("Server tắt.")