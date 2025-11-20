# File: 2_face_training.py
# Phiên bản: High-Res Training (Tự tìm và cắt mặt chuẩn từ ảnh gốc chất lượng cao)

import cv2
import numpy as np
import os
import onnxruntime
import argparse
from pathlib import Path

# Cấu hình Model (Đảm bảo 2 file này đang nằm cạnh file .py)
MODEL_DETECTOR = "face_detector.onnx"
MODEL_ARCFACE = "w600k_r50.onnx"

def main(args):
    dataset_dir = Path(args.dataset_dir)
    
    # 1. Kiểm tra model
    if not os.path.exists(MODEL_DETECTOR) or not os.path.exists(MODEL_ARCFACE):
        print(f"[LỖI] Thiếu file model. Cần có '{MODEL_DETECTOR}' và '{MODEL_ARCFACE}' trong thư mục.")
        return

    print("[INFO] Đang khởi tạo hệ thống Training...")
    
    # Khởi tạo bộ phát hiện khuôn mặt (Để tìm mặt trong ảnh to)
    # score_threshold=0.7: Chỉ học những ảnh mặt thực sự rõ nét
    detector = cv2.FaceDetectorYN.create(MODEL_DETECTOR, "", (320, 320), 0.7, 0.3, 5000)
    
    # Khởi tạo bộ trích xuất đặc trưng (ArcFace)
    session = onnxruntime.InferenceSession(MODEL_ARCFACE)
    input_name = session.get_inputs()[0].name

    embeddings = []
    labels = []
    
    # Lấy danh sách thư mục học sinh
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print(f"[LỖI] Không thấy thư mục dữ liệu nào trong '{dataset_dir}'.")
        return

    print(f"[INFO] Tìm thấy {len(person_dirs)} người. Bắt đầu xử lý...")

    for person_dir in person_dirs:
        print(f" -> Đang học: {person_dir.name}")
        
        # Lấy tất cả ảnh trong thư mục
        image_paths = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
        
        # Giới hạn số lượng ảnh học để tránh mất cân bằng (Lấy tối đa 150 ảnh/người)
        image_paths = image_paths[:150] 

        for img_path in image_paths:
            # Đọc ảnh gốc (Ảnh to, chất lượng cao)
            img = cv2.imread(str(img_path))
            if img is None: continue

            h, w = img.shape[:2]

            # --- BƯỚC QUAN TRỌNG: TÌM MẶT TRONG ẢNH TO ---
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            
            if faces is None or len(faces) == 0:
                # Ảnh không tìm thấy mặt -> Bỏ qua (để tránh học rác)
                continue
            
            # Lấy khuôn mặt to nhất trong ảnh (thường là mặt học sinh)
            face = faces[np.argmax([f[2]*f[3] for f in faces])]
            x, y, fw, fh = face[0:4].astype(int)
            
            # Cắt chính xác khuôn mặt ra
            # Mở rộng vùng cắt một chút xíu (padding) để không bị mất cằm/trán
            pad = 5
            crop = img[max(0, y-pad):min(h, y+fh+pad), max(0, x-pad):min(w, x+fw+pad)]
            
            if crop.size == 0: continue

            # --- BƯỚC CHUẨN BỊ CHO ARCFACE ---
            # 1. Resize về chuẩn 112x112
            face_blob = cv2.resize(crop, (112, 112))
            
            # 2. Chuẩn hóa màu sắc và pixel
            face_blob = cv2.cvtColor(face_blob, cv2.COLOR_BGR2RGB)
            face_blob = (face_blob.astype(np.float32) - 127.5) / 128.0
            face_blob = np.transpose(face_blob, [2, 0, 1])[np.newaxis, ...]

            # 3. Trích xuất Vector đặc trưng (Embedding)
            emb = session.run(None, {input_name: face_blob})[0]
            
            # 4. Chuẩn hóa Vector (L2 Norm)
            emb_norm = emb / np.linalg.norm(emb)
            
            embeddings.append(emb_norm.flatten())
            labels.append(person_dir.name)

    if not embeddings:
        print("[ERROR] Không trích xuất được dữ liệu nào. Kiểm tra lại thư mục ảnh.")
        return

    # Lưu kết quả vào file .npz
    np.savez_compressed(args.output_file, embeddings=np.array(embeddings, dtype=np.float32), labels=np.array(labels))
    print(f"\n[THÀNH CÔNG] Đã học xong {len(embeddings)} khuôn mặt.")
    print(f"[LƯU TẠI] {args.output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, default="datasets/faces")
    p.add_argument("--output-file", type=str, default="face_embeddings.npz")
    main(p.parse_args())