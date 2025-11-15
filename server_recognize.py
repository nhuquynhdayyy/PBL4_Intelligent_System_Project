# File: server_recognize.py
# CHẠY TRÊN MÁY TÍNH LOCAL CỦA BẠN
import asyncio
import websockets
import cv2
import numpy as np
from recognize_face import recognize_face, face_db # Import hàm và db đã có

# --- CẤU HÌNH SERVER ---
SERVER_HOST = '0.0.0.0'  # Lắng nghe trên tất cả các IP của máy
SERVER_PORT = 8766      # Sử dụng một port khác để tránh xung đột

async def recognition_handler(websocket):
    """
    Hàm này được gọi mỗi khi Raspberry Pi kết nối.
    Nó nhận frame, chạy nhận diện và hiển thị kết quả.
    """
    print(f"[INFO] Client Raspberry Pi đã kết nối từ: {websocket.remote_address}")
    try:
        # Vòng lặp liên tục nhận message (frame ảnh) từ client
        async for message in websocket:
            # 1. Nhận và giải mã frame ảnh từ Pi
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 2. Chạy hàm nhận diện khuôn mặt trên frame nhận được
            # Chúng ta có thể resize để xử lý nhanh hơn nếu cần
            # frame_small = cv2.resize(frame, (480, 360))
            student_id, conf = recognize_face(frame)

            # 3. Hiển thị kết quả lên frame
            label = "Unknown"
            color = (0, 0, 255) # Màu đỏ cho người lạ

            if student_id and student_id != "unknown":
                label = f"{student_id} ({conf*100:.0f}%)"
                color = (0, 255, 0) # Màu xanh cho người quen

            # Vẽ label lên góc trên bên trái của frame
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # 4. Hiển thị frame kết quả lên màn hình máy tính Local
            cv2.imshow("Face Recognition Server", frame)

            # Nhấn 'q' hoặc ESC để thoát
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Client Raspberry Pi đã ngắt kết nối.")
    finally:
        # Đóng cửa sổ khi client ngắt kết nối
        cv2.destroyAllWindows()


async def main():
    print(f"[INFO] Server nhận diện khuôn mặt đang khởi động tại ws://{SERVER_HOST}:{SERVER_PORT}")
    # Đảm bảo bạn đã có file recognize_face.py và face_db.pkl trong cùng thư mục
    if not face_db:
        print("[ERROR] Không thể tải face_db.pkl. Hãy chắc chắn file tồn tại.")
        return
        
    print(f"[INFO] Đã tải thành công cơ sở dữ liệu của {len(face_db)} người.")
    
    async with websockets.serve(recognition_handler, SERVER_HOST, SERVER_PORT, max_size=None):
        await asyncio.Future()  # Chạy mãi mãi

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Server đã tắt.")