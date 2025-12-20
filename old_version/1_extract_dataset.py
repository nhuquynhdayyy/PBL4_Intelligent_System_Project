# Tên file: 1_extract_dataset.py
import cv2
import os
import argparse
import numpy as np
from pathlib import Path

def align_face(frame, landmarks):
    """Căn chỉnh khuôn mặt dựa trên vị trí 2 mắt."""
    pts = landmarks.copy()
    order = pts[np.argsort(pts[:, 0])]
    left_eye, right_eye = order[0], order[-1]
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int((left_eye[1] + right_eye[1]) / 2)
    eye_center = (center_x, center_y)
    
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

def main(args):
    person_output_path = Path(args.output_dir) / args.name
    person_output_path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Đang xử lý video: {args.video}")
    print(f"[*] Dữ liệu sẽ được lưu tại: {person_output_path}")

    # Khởi tạo bộ phát hiện mặt (Yêu cầu file face_detector.onnx)
    if not os.path.exists(args.model):
        print(f"[ERROR] Không tìm thấy file model: {args.model}. Hãy tải về và để cùng thư mục.")
        return

    detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), args.score_threshold, 0.3, 5000)
    cap = cv2.VideoCapture(args.video)
    saved_count, frame_count = 0, 0

    while saved_count < args.max_images:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % args.skip_frames != 0: continue

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        
        if faces is None or len(faces) == 0: continue
        
        # Lấy mặt lớn nhất
        face = faces[np.argmax([f[2]*f[3] for f in faces])]
        x, y, w, h = face[0:4].astype(int)
        
        if w < args.min_size or h < args.min_size: continue

        landmarks = face[4:14].reshape((5, 2)).astype(np.int32)
        aligned = align_face(frame, landmarks)
        
        # Detect lại trên ảnh đã align
        h_a, w_a = aligned.shape[:2]
        detector.setInputSize((w_a, h_a))
        _, faces2 = detector.detect(aligned)

        crop = None
        if faces2 is not None and len(faces2) > 0:
            f2 = faces2[np.argmax([f[2]*f[3] for f in faces2])]
            ax, ay, aw, ah = f2[0:4].astype(int)
            crop = aligned[ay:ay+ah, ax:ax+aw]
        
        if crop is None or crop.size == 0: continue

        # Lọc ảnh mờ
        blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if blur < args.blur_threshold: continue

        final_face = cv2.resize(crop, (112, 112))
        cv2.imwrite(str(person_output_path / f"{saved_count:04d}.jpg"), final_face)
        saved_count += 1
        if saved_count % 10 == 0:
            print(f" -> Đã lưu {saved_count}/{args.max_images} ảnh (độ nét: {blur:.1f})")

    cap.release()
    print(f"[DONE] Hoàn thành. Đã lưu {saved_count} ảnh cho {args.name}.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="datasets/faces")
    p.add_argument("--max-images", type=int, default=150) # Giới hạn 150 ảnh cho cân bằng
    p.add_argument("--min-size", type=int, default=60)
    p.add_argument("--blur-threshold", type=float, default=60.0)
    p.add_argument("--skip-frames", type=int, default=5)
    p.add_argument("--model", type=str, default="face_detector.onnx")
    p.add_argument("--score-threshold", type=float, default=0.6)
    main(p.parse_args())