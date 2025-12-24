# File: pc_ai_server_integrated.py

import asyncio
import websockets
import cv2
import numpy as np
import time
import requests
import threading
from flask import Flask, Response
from ultralytics import YOLO
from collections import deque, Counter
from recognize_face import recognize_face 

# --- CẤU HÌNH ---
AI_SERVER_HOST = '0.0.0.0'
WEBSOCKET_PORT = 8765
STREAM_PORT = 5001
MAIN_WEB_API_URL = "http://127.0.0.1:5000/api/recognize"

# --- BIẾN TOÀN CỤC ---
global_latest_frame = None
global_ai_results = []
lock = threading.Lock()
tracks_data = {}

# --- FLASK STREAMING ---
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global global_latest_frame, global_ai_results
    while True:
        with lock:
            if global_latest_frame is None:
                time.sleep(0.05); continue
            display_frame = global_latest_frame.copy()
            current_results = list(global_ai_results)

        for (x1, y1, x2, y2, label, color) in current_results:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            if label: cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        (flag, encodedImage) = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if flag: yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

def run_flask():
    app.run(host=AI_SERVER_HOST, port=STREAM_PORT, debug=False, use_reloader=False)

# --- AI WORKER ---
print("[INFO] Loading YOLOv8-Pose...")
pose_model = YOLO("yolov8n-pose.pt") 
# # Sửa đường dẫn cũ thành đường dẫn tới model 50 Epoch mới nhất
# pose_model = YOLO("runs/detect/train4/weights/best.pt")
def ai_worker_loop():
    global global_latest_frame, global_ai_results, tracks_data
    
    while True:
        frame_to_process = None
        with lock:
            if global_latest_frame is not None:
                frame_to_process = global_latest_frame.copy()
        
        if frame_to_process is None: time.sleep(0.1); continue

        # Resize nhỏ để Pose chạy nhanh
        h_orig, w_orig = frame_to_process.shape[:2]
        ai_w = 320
        ai_h = int(ai_w * h_orig / w_orig)
        ai_frame = cv2.resize(frame_to_process, (ai_w, ai_h))
        scale_x = w_orig / ai_w
        scale_y = h_orig / ai_h

        # 1. Chạy Pose 
        results = pose_model.track(ai_frame, persist=True, conf=0.5, verbose=False)
        temp_results = [] 

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()

            for box, track_id, kps in zip(boxes, track_ids, keypoints):
                # Tọa độ Body
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)

                if track_id not in tracks_data:
                    tracks_data[track_id] = {'state': 'SIT', 'state_start': time.time(), 'last_api_sent': 0, 'identified_name': None, 'face_buffer': []}

                # Logic Sit/Stand
                ratio = check_pose_ratio(kps)
                state = "STANDING" if ratio > 0.65 else "SITTING"
                
                if state != tracks_data[track_id]['state']:
                    tracks_data[track_id]['state'] = state
                    tracks_data[track_id]['state_start'] = time.time()
                
                is_standing = (state == "STANDING" and time.time() - tracks_data[track_id]['state_start'] > 0.8)
                
                color = (0, 255, 0); label = f"SIT {track_id}"

                if is_standing:
                    color = (0, 0, 255)
                    name_disp = tracks_data[track_id]['identified_name'] if tracks_data[track_id]['identified_name'] else f"STAND {track_id}"
                    label = name_disp
                    
                    # === 2. NHẬN DIỆN KHUÔN MẶT (ZOOM FACE) ===
                    # Chỉ nhận diện nếu chưa biết tên và đã lâu chưa thử
                    if tracks_data[track_id]['identified_name'] is None:
                        if time.time() - tracks_data[track_id]['last_api_sent'] > 1.0:
                            
                            # --- CÔNG THỨC CROP (Theo tỷ lệ phần trăm) ---
                            h_box = y2 - y1
                            w_box = x2 - x1
                            
                            fy1 = y1 + int(0.05 * h_box)
                            fy2 = y1 + int(0.45 * h_box)
                            fx1 = x1 + int(0.10 * w_box)
                            fx2 = x2 - int(0.10 * w_box)

                            # Đảm bảo không crop ra ngoài ảnh
                            fy1 = max(0, fy1); fy2 = min(h_orig, fy2)
                            fx1 = max(0, fx1); fx2 = min(w_orig, fx2)
                            
                            face_crop = frame_to_process[fy1:fy2, fx1:fx2]
                            
                            if face_crop.size > 0:
                                # --- ZOOM LÊN 480x480 ---
                                # Việc phóng to này giúp DeepFace nhìn rõ hơn trên ảnh mờ
                                zoom_face = cv2.resize(face_crop, (480, 480))
                                
                                # Gọi hàm nhận diện
                                name, conf = recognize_face(zoom_face)
                                print(f"[DEBUG] ID:{track_id} | Name:{name} | Conf:{conf}")

                                if name != "Unknown":
                                    tracks_data[track_id]['face_buffer'].append(name)
                                    
                                    # Voting 2 phiếu cho chắc
                                    if len(tracks_data[track_id]['face_buffer']) >= 2:
                                        final_name = Counter(tracks_data[track_id]['face_buffer']).most_common(1)[0][0]
                                        print(f"[SUCCESS] Nhận diện: {final_name}")
                                        try:
                                            requests.post(MAIN_WEB_API_URL, json={"student_code": final_name}, timeout=0.5)
                                            tracks_data[track_id]['identified_name'] = final_name
                                            tracks_data[track_id]['last_api_sent'] = time.time()
                                        except: pass
                                        tracks_data[track_id]['face_buffer'] = []
                                else:
                                    # Nếu Unknown, thử lại nhanh hơn
                                    tracks_data[track_id]['last_api_sent'] = time.time() - 0.5
                else:
                    tracks_data[track_id]['identified_name'] = None
                    tracks_data[track_id]['face_buffer'] = []

                temp_results.append((x1, y1, x2, y2, label, color))

        with lock: global_ai_results = temp_results
        time.sleep(0.01)

def check_pose_ratio(kps):
    sy = (kps[5][1] + kps[6][1]) / 2; hy = (kps[11][1] + kps[12][1]) / 2; ky = (kps[13][1] + kps[14][1]) / 2
    torso = abs(hy - sy); thigh = abs(ky - hy)
    return thigh / torso if torso != 0 else 0

# --- RECEIVER ---
async def receive_loop():
    global global_latest_frame
    async with websockets.serve(handler, AI_SERVER_HOST, WEBSOCKET_PORT, ping_interval=None): await asyncio.Future()

async def handler(websocket):
    global global_latest_frame
    try:
        async for message in websocket:
            nparr = np.frombuffer(message, np.uint8)
            with lock: global_latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: pass

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=ai_worker_loop, daemon=True).start()
    print(f"System Ready. Stream: http://127.0.0.1:{STREAM_PORT}/video_feed")
    try: asyncio.run(receive_loop())
    except: pass