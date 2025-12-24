# pc_ai_server_integrated.py
import asyncio
import websockets
import cv2
import numpy as np
import time
import requests
import threading
from flask import Flask, Response
from ultralytics import YOLO
from collections import Counter
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
# Lưu trữ tên đã nhận diện được cho từng Track ID
tracks_info = {} 
FACE_THRESHOLD = 0.32
last_api_sent_log = {}

app = Flask(__name__)

# --- GIAO DIỆN WEB STREAMING ---
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
            cv2.putText(display_frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        (flag, encodedImage) = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if flag: yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

def run_flask():
    app.run(host=AI_SERVER_HOST, port=STREAM_PORT, debug=False, use_reloader=False)

# --- AI WORKER (TRÁI TIM CỦA HỆ THỐNG) ---
print("[INFO] Đang nạp mô hình YOLOv8 từ train4/best.pt...")
# Nạp model bạn vừa train xong
model = YOLO("runs/detect/train4/weights/best.pt")

tracks_data = {} # Khởi tạo biến lưu lịch sử nhận diện

def ai_worker_loop():
    global global_latest_frame, global_ai_results, tracks_info, tracks_data
    
    # Định nghĩa bảng màu cho các lớp
    color_map = {
        "face": (255, 0, 0),          # Xanh dương
        "hand-raising": (0, 255, 0),  # Xanh lá
        "standing": (255, 0, 255),    # Tím
        "sitting": (0, 255, 255)      # Vàng
    }

    while True:
        frame_to_process = None
        with lock:
            if global_latest_frame is not None:
                frame_to_process = global_latest_frame.copy()
        
        if frame_to_process is None: time.sleep(0.1); continue

        # Dự đoán bằng model train4
        # Dùng track để giữ ID ổn định cho từng đối tượng
        results = model.track(frame_to_process, persist=True, conf=0.4, verbose=False)
        
        temp_results = []
        current_faces = []   # Lưu danh sách các mặt phát hiện được
        current_actions = [] # Lưu danh sách các hành động (đứng/giơ tay)

        # if results[0].boxes.id is not None:
        #     boxes = results[0].boxes.xyxy.cpu().numpy()
        #     cls_ids = results[0].boxes.cls.int().cpu().numpy()
        #     track_ids = results[0].boxes.id.int().cpu().numpy()

        #     for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
        #         x1, y1, x2, y2 = map(int, box)
        #         label_name = model.names[cls_id]
        #         color = color_map.get(label_name, (255, 255, 255))

        #         # --- TRONG AI_WORKER_LOOP ---

        #         # Thêm Deque để lưu lịch sử nhận diện (đặt ở đầu file hoặc đầu hàm)
        #         # tracks_data[track_id]['face_history'] = deque(maxlen=10)

        #         # 1. Nếu là KHUÔN MẶT
        #         if label_name == "face":
        #             face_w = x2 - x1
        #             face_h = y2 - y1
                    
        #             # CHỈ NHẬN DIỆN KHI MẶT ĐỦ LỚN (Tránh nhận diện nhầm người ở quá xa)
        #             if face_w > 40 and face_h > 40: 
        #                 face_crop = frame_to_process[y1:y2, x1:x2]
        #                 if face_crop.size > 0:
        #                     # Zoom to để khử nhiễu nhẹ
        #                     face_zoom = cv2.resize(face_crop, (224, 224)) 
        #                     name, conf = recognize_face(face_zoom)

        #                     # NGƯỠNG THẮT CHẶT: Giả sử Facenet dùng Cosine Distance (thường < 0.4 là khớp)
        #                     # Nếu conf của bạn càng nhỏ càng đúng, hãy chỉnh lại con số này
        #                     if name != "Unknown" and conf < 0.35: 
        #                         if track_id not in tracks_data:
        #                             tracks_data[track_id] = {'history': []}
                                
        #                         tracks_data[track_id]['history'].append(name)
                                
        #                         # CHỈ XÁC NHẬN KHI CÓ 7/10 KHUNG HÌNH TRÙNG TÊN NHAU
        #                         if len(tracks_data[track_id]['history']) >= 7:
        #                             occurence_count = Counter(tracks_data[track_id]['history'])
        #                             final_name, count = occurence_count.most_common(1)[0]
                                    
        #                             if count >= 5: # Ít nhất 5 lần xuất hiện cùng 1 tên
        #                                 tracks_info[track_id] = final_name
        #                                 display_label = f"{final_name}"
        #                             else:
        #                                 display_label = "Scanning..."
        #                         else:
        #                             display_label = "Scanning..."
        #                     else:
        #                         display_label = "Searching..."
        #             else:
        #                 display_label = "Too far to recognize"

        #         # 2. Nếu là HÀNH ĐỘNG PHÁT BIỂU (Đứng hoặc Giơ tay)
        #         elif label_name in ["hand-raising", "standing"]:
        #             current_actions.append({'box': (x1, y1, x2, y2), 'label': label_name})
        #             temp_results.append((x1, y1, x2, y2, label_name.upper(), color))
                
        #         # 3. Nếu là ngồi
        #         else:
        #             temp_results.append((x1, y1, x2, y2, label_name, color))
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            current_faces = []
            current_actions = []

            for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                x1, y1, x2, y2 = map(int, box)
                label_name = model.names[cls_id]
                color = (255, 255, 255)
                display_label = label_name

                # 1. XỬ LÝ LỚP FACE
                # Trong ai_worker_loop

                if label_name == "face":
                    # LUÔN KHỞI TẠO tracks_data cho ID mới, bất kể xa hay gần
                    if track_id not in tracks_data:
                        tracks_data[track_id] = {'history': []}

                    face_w = x2 - x1
                    identified_name = "Unknown" # Mặc định là Unknown

                    if face_w > 45: # Tăng lên 55 để bỏ qua các mặt quá nhỏ/xa
                        face_crop = frame_to_process[y1:y2, x1:x2]
                        name, dist = recognize_face(face_crop)

                        # DÙNG NGƯỠNG TẬP TRUNG FACE_THRESHOLD
                        if dist < FACE_THRESHOLD: 
                            tracks_data[track_id]['history'].append(name)
                        else:
                            tracks_data[track_id]['history'].append("Unknown")

                        if len(tracks_data[track_id]['history']) > 20:
                            tracks_data[track_id]['history'].pop(0)

                        # LOGIC BẦU CHỌN "THIẾU SỐ" (Strict Voting)
                        history = tracks_data[track_id]['history']
                        if len(history) >= 5:
                            occurence = Counter(history)
                            final_name, count = occurence.most_common(1)[0]

                            # Tên đó phải chiếm ít nhất 80% lịch sử nhận diện gần đây
                            if final_name != "Unknown" and count > (len(history) * 0.6):
                                tracks_info[track_id] = final_name
                                identified_name = final_name
                                display_label = f"Confirmed: {final_name}"
                                color = (255, 0, 0)
                            else:
                                display_label = "Analyzing..." # Không đoán bừa
                                color = (0, 0, 255)
                    else:
                        display_label = "Too far"
                        color = (128, 128, 128)

                    current_faces.append({'box': (x1, y1, x2, y2), 'name': identified_name})
                    temp_results.append((x1, y1, x2, y2, display_label, color))

                # 2. XỬ LÝ LỚP KHÁC (Standing, Hand-raising...)
                elif label_name in ["standing", "hand-raising"]:
                    current_actions.append({'box': (x1, y1, x2, y2), 'label': label_name})
                    temp_results.append((x1, y1, x2, y2, label_name.upper(), (0, 255, 0)))
                
                else: # Lớp Sitting hoặc lớp khác
                    temp_results.append((x1, y1, x2, y2, label_name, (0, 255, 255)))

            # --- LOGIC KẾT HỢP: XÁC NHẬN AI ĐANG PHÁT BIỂU (Bản tối ưu) ---
            current_time = time.time()
            
            for action in current_actions:
                ax1, ay1, ax2, ay2 = action['box']
                action_center_x = (ax1 + ax2) / 2 # Tâm của khung hình hành động

                for face in current_faces:
                    fx1, fy1, fx2, fy2 = face['box']
                    face_center_x = (fx1 + fx2) / 2 # Tâm của khuôn mặt
                    
                    # 1. Kiểm tra tâm mặt có nằm giữa chiều rộng của Box hành động không
                    # 2. Kiểm tra khoảng cách theo chiều dọc (Mặt phải nằm ở nửa trên của Box hành động)
                    is_x_aligned = ax1 - 20 <= face_center_x <= ax2 + 20
                    is_y_top = fy1 < ay1 + (ay2 - ay1) * 0.5 # Mặt phải nằm ở nửa trên cơ thể
                    
                    # if is_x_aligned and is_y_top:
                    #     if face['name'] != "Unknown":
                    #         # Đã xác định được ĐÚNG người này đang thực hiện hành động
                    #         print(f"[XÁC NHẬN] {face['name']} đang {action['label']}")
                    #         try:
                    #             requests.post(MAIN_WEB_API_URL, 
                    #                         json={"student_code": face['name'], "action": action['label']}, 
                    #                         timeout=0.1)
                    #         except: pass

                    if is_x_aligned and is_y_top:
                        student_name = face['name']
                        
                        if student_name != "Unknown":
                            # KIỂM TRA COOLDOWN: 
                            # Nếu chưa gửi bao giờ HOẶC đã gửi cách đây hơn 30 giây (1 lượt phát biểu thường > 30s)
                            last_sent = last_api_sent_log.get(student_name, 0)
                            
                            if current_time - last_sent > 30: # 30 giây cooldown
                                print(f"[GỬI API] Xác nhận {student_name} phát biểu.")
                                
                                # Cập nhật thời gian gửi ngay lập tức để chặn các vòng lặp sau
                                last_api_sent_log[student_name] = current_time
                                
                                try:
                                    # Gửi API sang Web Server
                                    requests.post(MAIN_WEB_API_URL, 
                                                json={"student_code": student_name, "action": action['label']}, 
                                                timeout=0.1)
                                except Exception as e:
                                    print(f"[LỖI API] {e}")
                            else:
                                # Đang trong thời gian chờ, không gửi trùng lặp
                                pass
                            
        with lock: global_ai_results = temp_results
        time.sleep(0.01)

# --- PHẦN NHẬN DỮ LIỆU TỪ WEB ---
async def receive_loop():
    async with websockets.serve(handler, AI_SERVER_HOST, WEBSOCKET_PORT, ping_interval=None):
        await asyncio.Future()

async def handler(websocket):
    global global_latest_frame
    try:
        async for message in websocket:
            nparr = np.frombuffer(message, np.uint8)
            with lock:
                global_latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: pass

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=ai_worker_loop, daemon=True).start()
    print(f"Hệ thống PBL4 sẵn sàng. Stream tại: http://localhost:{STREAM_PORT}/video_feed")
    try:
        asyncio.run(receive_loop())
    except: pass