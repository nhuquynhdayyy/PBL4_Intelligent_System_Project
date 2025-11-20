# File: extract_loose.py
# Tool cắt ảnh "Thả lỏng" - Lấy cả tóc và khung đầu
import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def main(args):
    output_path = Path(args.output_dir) / args.name
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path) # Xóa cũ làm lại cho sạch
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- XỬ LÝ VIDEO: {args.video} (CHẾ ĐỘ LẤY CẢ ĐẦU) ---")

    if not os.path.exists(args.model):
        print(f"[LỖI] Thiếu file {args.model}")
        return

    # Ngưỡng tin cậy cao (0.8) để chỉ lấy ảnh rõ
    detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), 0.8, 0.3, 5000)
    cap = cv2.VideoCapture(args.video)
    
    saved_count = 0
    frame_count = 0
    
    # Tỷ lệ mở rộng khung hình (0.4 = mở rộng 40% mỗi bên) -> Lấy được cả tóc
    PADDING_RATIO = 0.3 

    while saved_count < args.max_images:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % args.skip_frames != 0: continue

        h_img, w_img = frame.shape[:2]
        detector.setInputSize((w_img, h_img))
        _, faces = detector.detect(frame)
        
        if faces is None or len(faces) == 0: continue
        
        # Lấy mặt to nhất
        face = faces[np.argmax([f[2]*f[3] for f in faces])]
        x, y, w, h = face[0:4].astype(int)
        
        # --- KỸ THUẬT MỞ RỘNG KHUNG HÌNH (PADDING) ---
        # Tính toán để lấy rộng ra, bao trọn cả đầu và tóc
        new_x = max(0, x - int(w * PADDING_RATIO))
        new_y = max(0, y - int(h * PADDING_RATIO * 1.5)) # Mở bên trên nhiều hơn để lấy tóc
        new_w = min(w_img, x + w + int(w * PADDING_RATIO)) - new_x
        new_h = min(h_img, y + h + int(h * PADDING_RATIO)) - new_y
        
        # Cắt ảnh
        face_crop = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        
        if face_crop.size == 0: continue
        
        # Lọc ảnh quá mờ
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < args.blur_threshold:
            continue

        # Resize về 112x112 nhưng giữ nguyên tỉ lệ hình ảnh (thêm viền đen nếu cần)
        # Tuy nhiên để đơn giản cho ArcFace, ta resize thẳng luôn, chấp nhận hơi méo xíu nhưng giữ đủ chi tiết
        final_face = cv2.resize(face_crop, (112, 112))

        filename = f"{saved_count:04d}.jpg"
        cv2.imwrite(str(output_path / filename), final_face)
        saved_count += 1
        
        print(f"✅ Đã lưu: {filename}")

    cap.release()
    print(f"[XONG] Đã có {saved_count} ảnh mới cho {args.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="datasets/faces")
    parser.add_argument("--model", type=str, default="face_detector.onnx")
    parser.add_argument("--max-images", type=int, default=200) # Lấy 200 ảnh
    parser.add_argument("--skip-frames", type=int, default=2)
    parser.add_argument("--blur-threshold", type=float, default=80.0) # Giảm ngưỡng nét xuống xíu cho dễ bắt
    args = parser.parse_args()
    main(args)