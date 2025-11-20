# File: pc_server_final.py
# Phiên bản: V6 Logic + ArcFace V2 + Flask Integration

import asyncio
import websockets
import cv2
import numpy as np
import time
import threading
import requests
from collections import deque
from ultralytics import YOLO

# --- IMPORT ENGINE MỚI ---
from ai_engine import FaceRecognitionSystem

# --- CẤU HÌNH ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8765
BACKEND_URL = "http://127.0.0.1:5000/api/recognize" # API ghi nhận
VIDEO_PUSH_URL = "http://127.0.0.1:5000/api/video_stream/push" # API stream video
LOG_TIMEOUT = 5.0  # Giây chờ giữa các lần log cùng 1 người

# --- KHỞI TẠO ---
print("[INFO] Đang tải YOLOv8...")
person_model = YOLO("yolov8n.pt") # Model nhẹ để detect người

print("[INFO] Đang khởi tạo ArcFace System...")
# Đảm bảo bạn có đủ file: face_detector.onnx, w600k_r50.onnx, face_embeddings.npz
face_system = FaceRecognitionSystem(threshold=0.45) 

# --- BIẾN TOÀN CỤC ---
tracks = {}
last_logged = {}

def send_log_request(student_id, confidence):
    """Gửi kết quả nhận diện về Backend (chạy ngầm)"""
    def _send():
        try:
            requests.post(BACKEND_URL, json={
                "student_code": student_id,
                "confidence": float(confidence)
            }, timeout=1)
            print(f"[>>> SENT API] {student_id}")
        except Exception as e:
            print(f"[API ERROR] {e}")
    threading.Thread(target=_send).start()

def send_video_push(buffer_bytes):
    """Gửi frame ảnh về Backend để hiển thị trên Web (chạy ngầm)"""
    def _push():
        try:
            requests.post(VIDEO_PUSH_URL, data=buffer_bytes, 
                          headers={'Content-Type': 'image/jpeg'}, timeout=0.2)
        except: pass
    threading.Thread(target=_push).start()

async def ai_handler(websocket):
    print(f"[CONN] Client kết nối: {websocket.remote_address}")
    try:
        async for message in websocket:
            # 1. Decode ảnh từ Pi
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue

            # 2. Detect Người (YOLO)
            results = person_model.predict(frame, imgsz=320, conf=0.4, verbose=False, device='cpu')[0]
            boxes = [box for box in results.boxes if int(box.cls[0]) == 0] # Class 0 = Person

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w = y2 - y1, x2 - x1
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                key = (cx//50, cy//50) # Grid key để tracking đơn giản

                # --- LOGIC V6: THEO DÕI TRẠNG THÁI ---
                if key not in tracks:
                    tracks[key] = {
                        'heights': deque(maxlen=5),
                        'baseline': h,
                        'sit_start': None,
                        'stand_frames': 0,
                        'confirmed_stand': False,
                        'time_confirmed': 0,
                        'last_recog_time': 0
                    }
                
                t_data = tracks[key]
                t_data['heights'].append(h)
                
                # Kiểm tra đứng (ngưỡng 1.15 lần chiều cao ngồi)
                standing = h > 1.15 * t_data['baseline']

                # --- LOGIC CHỐNG TRÔI BASELINE (V6) ---
                if not standing:
                    # Nếu đang ngồi
                    if t_data['sit_start'] is None: t_data['sit_start'] = time.time()
                    sit_duration = time.time() - t_data['sit_start']
                    
                    # Chỉ cập nhật baseline khi ngồi ổn định > 1s
                    if sit_duration >= 1.0 and len(t_data['heights']) == 5:
                        t_data['baseline'] = np.median(t_data['heights'])
                    
                    # Reset trạng thái đứng
                    t_data['stand_frames'] = 0
                    t_data['confirmed_stand'] = False
                else:
                    # Nếu đang đứng -> Reset thời gian ngồi
                    t_data['sit_start'] = None
                    t_data['stand_frames'] += 1

                # --- XÁC NHẬN ĐỨNG (Debounce) ---
                # Phải detect đứng liên tiếp 5 frame (để tránh nhiễu nháy)
                if t_data['stand_frames'] > 5: 
                    if not t_data['confirmed_stand']:
                        t_data['time_confirmed'] = time.time() # Thời điểm bắt đầu đứng hẳn
                    t_data['confirmed_stand'] = True

                # --- NHẬN DIỆN (ArcFace) ---
                color = (0, 255, 0)
                label = ""
                
                if t_data['confirmed_stand']:
                    color = (0, 255, 255) # Vàng = Đang đứng
                    
                    # Đợi đứng ổn định 0.5s mới chụp
                    time_since_stand = time.time() - t_data['time_confirmed']
                    time_since_last_recog = time.time() - t_data['last_recog_time']
                    
                    if time_since_stand >= 0.5 and time_since_last_recog > 2.0:
                        # Cắt vùng đầu (Mở rộng lên trên một chút để lấy trọn đầu)
                        fy1 = max(0, y1 - int(h*0.1))
                        fy2 = min(frame.shape[0], y1 + int(h*0.5)) # Lấy 50% phần trên cơ thể
                        fx1 = max(0, x1)
                        fx2 = min(frame.shape[1], x2)
                        
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        
                        if face_crop.size > 0:
                            # GỌI AI ENGINE
                            name, conf = face_system.recognize(face_crop)
                            
                            if name not in ["Unknown", "NoFace", "AlignError"]:
                                label = f"{name} ({conf:.2f})"
                                color = (0, 0, 255) # Đỏ = Đã nhận diện
                                
                                # Gửi API (Kiểm tra debounce log)
                                curr_time = time.time()
                                if (name not in last_logged or 
                                    curr_time - last_logged[name] > LOG_TIMEOUT):
                                    print(f"[RECOGNIZED] {name} - Conf: {conf:.2f}")
                                    send_log_request(name, conf)
                                    last_logged[name] = curr_time
                                
                                t_data['last_recog_time'] = curr_time

                # Vẽ UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if label:
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                elif t_data['confirmed_stand']:
                    cv2.putText(frame, "STANDING", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                # Debug Baseline (Comment nếu không cần)
                # cv2.putText(frame, f"Base: {int(t_data['baseline'])} H: {h}", (x1, y1-30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 3. Hiển thị & Stream
            cv2.imshow("PC AI Server (V6 + ArcFace)", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
            # Nén và gửi stream đi (tùy chọn)
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret: send_video_push(buf.tobytes())

    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Client ngắt kết nối.")
    finally:
        cv2.destroyAllWindows()

async def main():
    print(f"[START] Server lắng nghe tại ws://{SERVER_HOST}:{SERVER_PORT}")
    async with websockets.serve(ai_handler, SERVER_HOST, SERVER_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass