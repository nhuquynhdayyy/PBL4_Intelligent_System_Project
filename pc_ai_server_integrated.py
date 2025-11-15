# File: pc_ai_server_integrated.py
# Phiên bản nâng cấp để giao tiếp với Web Server Flask

import asyncio
import websockets
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face
import requests # <-- Thư viện mới để gửi request

# --- CẤU HÌNH ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8765
BACKEND_API_URL = "http://127.0.0.1:5000/api/recognize" # <-- URL của backend
LOG_DEBOUNCE_SECONDS = 5 # Chờ 5s trước khi ghi nhận lại cùng 1 người

# --- LOAD MODEL ---
print("[INFO] Loading models...")
person_model = YOLO("yolov8n.pt")
print("[INFO] Models loaded.")

# --- BỘ NHỚ THEO DÕI ---
tracks = {}
last_display = {"img": None, "label": "", "time": 0}
last_logged_time = {} # <-- Biến mới để chống spam log

async def ai_processing_handler(websocket):
    print(f"[INFO] Raspberry Pi đã kết nối từ: {websocket.remote_address}")
    try:
        async for message in websocket:
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue

            res_person = person_model.predict(frame, imgsz=320, conf=0.4, device='cpu', verbose=False)[0]
            person_boxes = [list(map(int, box.xyxy[0].cpu().numpy())) for box in res_person.boxes if person_model.names[int(box.cls[0])] == "person"]

            for pb in person_boxes:
                x1, y1, x2, y2 = pb
                h, w = y2 - y1, x2 - x1
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                key = (cx//20, cy//20)

                if key not in tracks:
                    tracks[key] = {'heights': deque(maxlen=5), 'baseline': h, 'stand_frames': 0, 'time_stand_start': None, 'last_recog': 0, 'confirmed_stand': False, 'time_confirmed': 0}
                
                tracks[key]['heights'].append(h)
                baseline = tracks[key]['baseline']
                standing = h > 1.15 * baseline

                if len(tracks[key]['heights']) == 5 and not standing:
                    tracks[key]['baseline'] = np.median(tracks[key]['heights'])

                if standing:
                    if tracks[key]['time_stand_start'] is None: tracks[key]['time_stand_start'] = time.time()
                else:
                    tracks[key]['time_stand_start'] = None
                    tracks[key]['confirmed_stand'] = False
                
                elapsed = (time.time() - tracks[key]['time_stand_start']) if tracks[key]['time_stand_start'] else 0

                if standing and elapsed >= 1.0:
                    if not tracks[key]['confirmed_stand']: tracks[key]['time_confirmed'] = time.time()
                    tracks[key]['confirmed_stand'] = True
                
                if tracks[key]['confirmed_stand']:
                    now = time.time()
                    stable_time = now - tracks[key].get('time_confirmed', 0)

                    if stable_time >= 1.0 and now - tracks[key]['last_recog'] > 5:
                        fy1, fy2 = y1 + int(0.05 * h), y1 + int(0.45 * h)
                        fx1, fx2 = x1 + int(0.10 * w), x2 - int(0.10 * w)
                        
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            zoom_face = cv2.resize(face_crop, (480, 480))
                            cv2.imshow("Zoom Face", zoom_face) 
                            
                            student_id, conf = recognize_face(zoom_face)
                            
                            # =======================================================
                            # === PHẦN TÍCH HỢP - THAY ĐỔI QUAN TRỌNG NHẤT LÀ ĐÂY ===
                            # =======================================================
                            if student_id and student_id != "Unknown" and conf >= 0.5:
                                # 1. Vẫn vẽ lên màn hình OpenCV để debug
                                label = f"{student_id} ({conf*100:.0f}%)"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                # 2. Kiểm tra logic debounce (chống spam)
                                current_time = time.time()
                                if (student_id not in last_logged_time or \
                                    current_time - last_logged_time.get(student_id, 0) > LOG_DEBOUNCE_SECONDS):
                                    
                                    print(f"[INFO] Phát hiện {student_id}. Đang gửi yêu cầu xác nhận tới backend...")
                                    try:
                                        # 3. Gửi HTTP POST Request tới server Flask
                                        response = requests.post(BACKEND_API_URL, json={
                                            "student_code": student_id,
                                            "confidence": conf
                                        }, timeout=3)

                                        if response.status_code == 200:
                                            print(f"[SUCCESS] Backend đã nhận yêu cầu cho {student_id}.")
                                            last_logged_time[student_id] = current_time
                                        else:
                                            print(f"[BACKEND ERROR] Server phản hồi: {response.status_code} - {response.text}")

                                    except requests.exceptions.RequestException as e:
                                        print(f"[CONNECTION ERROR] Không thể kết nối tới backend: {e}")
                            
                            tracks[key]['last_recog'] = now
                
                color = (0, 255, 255) if tracks[key]['confirmed_stand'] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("AI Processing Server (Integrated)", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break
            
    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Raspberry Pi đã ngắt kết nối.")
    finally:
        cv2.destroyAllWindows()

# --- Phần main giữ nguyên ---
async def main():
    print(f"[INFO] WebSocket AI Server (Integrated) đang khởi động tại ws://{SERVER_HOST}:{SERVER_PORT}")
    async with websockets.serve(ai_processing_handler, SERVER_HOST, SERVER_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Server đã tắt.")