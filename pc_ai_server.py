# File: pc_ai_server.py (chạy trên máy tính Local)
import asyncio
import websockets
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
from recognize_face import recognize_face  # Import hàm nhận diện khuôn mặt của bạn

# --- CẤU HÌNH SERVER ---
SERVER_HOST = '0.0.0.0'  # Để lắng nghe trên tất cả các IP của máy
SERVER_PORT = 8765

# --- LOAD MODEL (giữ nguyên từ code gốc của bạn) ---
print("[INFO] Loading models...")
hand_model = YOLO("runs/detect/train2/weights/best.pt")
person_model = YOLO("yolov8n.pt")
print("[INFO] Models loaded.")

# --- BỘ NHỚ THEO DÕI (giữ nguyên từ code gốc của bạn) ---
tracks = {}
last_display = {"img": None, "label": "", "time": 0}

async def ai_processing_handler(websocket):
    """
    Hàm này sẽ được gọi mỗi khi có một client (Raspberry Pi) kết nối.
    Nó nhận frame, xử lý AI và hiển thị kết quả.
    """
    print(f"[INFO] Raspberry Pi đã kết nối từ: {websocket.remote_address}")
    try:
        async for message in websocket:
            # 1. Nhận và giải nén frame ảnh từ Pi
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 2. CHẠY TOÀN BỘ LOGIC AI TỪ FILE main_match_cpu_v5.py CỦA BẠN
            # (Copy và paste code xử lý từ file gốc vào đây)
            # --- Detect person ---
            res_person = person_model.predict(frame, imgsz=640, conf=0.3, device='cpu', verbose=False)[0]
            person_boxes = [list(map(int, box.xyxy[0].cpu().numpy())) for box in res_person.boxes if person_model.names[int(box.cls[0])] == "person"]

            # --- Xử lý từng người ---
            for pb in person_boxes:
                x1, y1, x2, y2 = pb
                h, w = y2 - y1, x2 - x1
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                key = (cx//20, cy//20)

                if key not in tracks:
                    tracks[key] = {'heights': deque(maxlen=5), 'baseline': h, 'stand_frames': 0, 'time_stand_start': None, 'last_recog': 0, 'confirmed_stand': False, 'time_confirmed': 0}
                
                tracks[key]['heights'].append(h)
                baseline = tracks[key]['baseline']
                standing = h > 1.1 * baseline

                if len(tracks[key]['heights']) == 5 and not standing:
                    tracks[key]['baseline'] = np.median(tracks[key]['heights'])

                if standing:
                    if tracks[key]['time_stand_start'] is None: tracks[key]['time_stand_start'] = time.time()
                    tracks[key]['stand_frames'] += 1
                else:
                    tracks[key]['stand_frames'] = 0
                    tracks[key]['time_stand_start'] = None
                    tracks[key]['confirmed_stand'] = False
                
                elapsed = (time.time() - tracks[key]['time_stand_start']) if tracks[key]['time_stand_start'] else 0

                if standing and elapsed >= 1.0:
                    if not tracks[key]['confirmed_stand']: tracks[key]['time_confirmed'] = time.time()
                    tracks[key]['confirmed_stand'] = True
                
                if tracks[key]['confirmed_stand']:
                    now = time.time()
                    stable_time = now - tracks[key].get('time_confirmed', 0)

                    # --- Nhận diện khuôn mặt (lưu ý: không còn cap.read() nữa) ---
                    if stable_time >= 1.0 and now - tracks[key]['last_recog'] > 5:
                        fy1, fy2 = y1 + int(0.05 * h), y1 + int(0.45 * h)
                        fx1, fx2 = x1 + int(0.10 * w), x2 - int(0.10 * w)
                        
                        face_crop = frame[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            zoom_face = cv2.resize(face_crop, (480, 480))
                            cv2.imshow("Zoom Face", zoom_face) 
                            student_id, conf = recognize_face(zoom_face)
                            print(f"[INFO] Đứng ổn định: {student_id} ({conf:.2f})")

                            if student_id and student_id != "unknown" and conf >= 0.5:
                                label = f"{student_id} ({conf*100:.0f}%)"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                face_thumb = cv2.resize(zoom_face, (120, 120))
                                last_display.update({"img": face_thumb, "label": label, "time": now})
                            
                            tracks[key]['last_recog'] = now
                
                color = (0, 255, 255) if tracks[key]['confirmed_stand'] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # --- Hiển thị overlay ---
            if last_display["img"] is not None:
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (160, 160), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                frame[20:140, 20:140] = last_display["img"]
                cv2.putText(frame, last_display["label"], (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # 3. Hiển thị kết quả lên màn hình máy tính Local
            cv2.imshow("AI Processing Server", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Raspberry Pi đã ngắt kết nối.")
    finally:
        cv2.destroyAllWindows()


async def main():
    print(f"[INFO] WebSocket AI Server đang khởi động tại ws://{SERVER_HOST}:{SERVER_PORT}")
    async with websockets.serve(ai_processing_handler, SERVER_HOST, SERVER_PORT):
        await asyncio.Future()  # Chạy mãi mãi

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Server đã tắt.")