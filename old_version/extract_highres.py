# File: extract_highres_debug.py

import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def main(args):
    output_path = Path(args.output_dir) / args.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- XỬ LÝ VIDEO: {args.video} ---")
    print(f"-> Chế độ xoay: {args.rotate}")

    if not os.path.exists(args.model):
        print(f"[LỖI] Thiếu file model: {args.model}")
        return

    # HẠ NGƯỠNG XUỐNG 0.6 ĐỂ DỄ BẮT MẶT HƠN
    detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), 0.6, 0.3, 5000)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở video. Kiểm tra đường dẫn hoặc format (nên dùng .mp4)")
        return

    saved_count = 0
    frame_count = 0

    while saved_count < args.max_images:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % args.skip_frames != 0: continue

        # --- XỬ LÝ XOAY ẢNH (QUAN TRỌNG CHO PI) ---
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        h_img, w_img = frame.shape[:2]
        detector.setInputSize((w_img, h_img))
        _, faces = detector.detect(frame)
        
        # DEBUG: Tại sao không thấy mặt?
        if faces is None or len(faces) == 0: 
            # print(f"Frame {frame_count}: Không thấy mặt (thử xoay video xem?)")
            continue
        
        # Lấy mặt to nhất
        face = faces[np.argmax([f[2]*f[3] for f in faces])]
        x, y, w, h = face[0:4].astype(int)
        confidence = face[-1]

        # Cắt vùng mặt để kiểm tra độ nét
        face_roi = frame[max(0,y):min(h_img,y+h), max(0,x):min(w_img,x+w)]
        if face_roi.size == 0: continue
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # DEBUG: Tại sao thấy mặt mà không lưu?
        if blur_score < args.blur_threshold:
            print(f"Frame {frame_count}: Bỏ qua do mờ (Score: {blur_score:.1f} < {args.blur_threshold})")
            continue

        # Lưu ảnh gốc
        filename = f"{saved_count:04d}.jpg"
        cv2.imwrite(str(output_path / filename), frame)
        saved_count += 1
        
        print(f"✅ Đã lưu: {filename} | Độ nét: {blur_score:.1f} | Độ tin cậy: {confidence:.2f}")

    cap.release()
    print(f"\n[HOÀN TẤT] Đã lưu {saved_count} ảnh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="datasets/faces")
    parser.add_argument("--model", type=str, default="face_detector.onnx")
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument("--skip-frames", type=int, default=2)
    
    # GIẢM NGƯỠNG MẶC ĐỊNH XUỐNG 40.0 CHO CAM PI
    parser.add_argument("--blur-threshold", type=float, default=40.0) 
    
    # THÊM TÙY CHỌN XOAY
    # 0: Không xoay, 90: Xoay phải, -90: Xoay trái, 180: Lộn ngược
    parser.add_argument("--rotate", type=int, default=0, help="Xoay video: 0, 90, -90, 180")
    
    args = parser.parse_args()
    main(args)