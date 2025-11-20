# # File: extract_highres.py
# # Phiên bản: FULL FRAME - Lưu nguyên khung hình video gốc (Không cắt)
# # Chỉ lưu những frame có khuôn mặt rõ nét.

# import cv2
# import os
# import argparse
# import numpy as np
# from pathlib import Path

# def main(args):
#     output_path = Path(args.output_dir) / args.name
#     if output_path.exists():
#         import shutil
#         shutil.rmtree(output_path) # Xóa thư mục cũ làm lại cho sạch
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     print(f"--- XỬ LÝ VIDEO: {args.video} (CHẾ ĐỘ FULL KHUNG HÌNH) ---")
#     print(f"-> Lưu tại: {output_path}")

#     if not os.path.exists(args.model):
#         print(f"[LỖI] Thiếu file model: {args.model}")
#         return

#     # Khởi tạo bộ phát hiện mặt (YuNet)
#     # score_threshold=0.85: Chỉ chấp nhận khi AI rất chắc chắn đó là mặt
#     detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), 0.85, 0.3, 5000)
    
#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         print(f"[LỖI] Không thể mở video {args.video}")
#         return

#     saved_count = 0
#     frame_count = 0

#     while saved_count < args.max_images:
#         ret, frame = cap.read()
#         if not ret: break
        
#         frame_count += 1
#         # Bỏ qua frame để tránh ảnh quá giống nhau (mặc định mỗi 3 frame lấy 1)
#         if frame_count % args.skip_frames != 0: continue

#         h_img, w_img = frame.shape[:2]
        
#         # Resize input cho detector (không ảnh hưởng ảnh gốc)
#         detector.setInputSize((w_img, h_img))
#         _, faces = detector.detect(frame)
        
#         if faces is None or len(faces) == 0: continue
        
#         # Lấy khuôn mặt to nhất trong khung hình (Chủ thể chính)
#         face = faces[np.argmax([f[2]*f[3] for f in faces])]
#         x, y, w, h = face[0:4].astype(int)
        
#         # --- KIỂM TRA CHẤT LƯỢNG ---
#         # Cắt vùng mặt ra chỉ để kiểm tra độ nét (Blur check)
#         # Chứ không lưu vùng cắt này.
#         face_roi = frame[max(0,y):min(h_img,y+h), max(0,x):min(w_img,x+w)]
        
#         if face_roi.size == 0: continue
        
#         # Kiểm tra độ nét bằng Laplacian
#         gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
#         blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
#         if blur_score < args.blur_threshold:
#             # Ảnh mờ -> Bỏ qua
#             continue

#         # --- LƯU ẢNH ---
#         # Lưu NGUYÊN GỐC frame (Full HD/HD...)
#         filename = f"{saved_count:04d}.jpg"
#         cv2.imwrite(str(output_path / filename), frame)
#         saved_count += 1
        
#         # In tiến độ mỗi 10 ảnh
#         if saved_count % 10 == 0:
#             print(f"✅ Đã lưu: {filename} | Độ nét mặt: {blur_score:.1f}")

#     cap.release()
#     print(f"\n[HOÀN TẤT] Đã lưu {saved_count} ảnh Full Frame chất lượng cao.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video", type=str, required=True, help="Đường dẫn file video")
#     parser.add_argument("--name", type=str, required=True, help="Tên thư mục (Mã sinh viên + Tên)")
#     parser.add_argument("--output-dir", type=str, default="datasets/faces")
#     parser.add_argument("--model", type=str, default="face_detector.onnx")
#     parser.add_argument("--max-images", type=int, default=150, help="Số lượng ảnh tối đa cần lấy")
#     parser.add_argument("--skip-frames", type=int, default=3, help="Số frame bỏ qua")
#     parser.add_argument("--blur-threshold", type=float, default=100.0, help="Ngưỡng độ nét (Càng cao càng khắt khe)")
    
#     args = parser.parse_args()
#     main(args)



# File: extract_highres_debug.py
# Phiên bản: Sửa lỗi cho Camera Pi (Xoay ảnh + Hạ ngưỡng + Debug tin nhắn)

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