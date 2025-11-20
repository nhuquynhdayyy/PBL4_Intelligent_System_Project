# Tên file: 2_face_training.py
import cv2
import numpy as np
import os
import onnxruntime
import argparse
from pathlib import Path

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 128.0
    return np.transpose(img, [2, 0, 1])[np.newaxis, ...]

def main(args):
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"[ERROR] Thư mục dataset không tồn tại: {dataset_dir}")
        return

    if not os.path.exists(args.recognizer_model):
        print(f"[ERROR] Không tìm thấy model ArcFace: {args.recognizer_model}")
        return

    print("[INFO] Đang tải model ArcFace...")
    session = onnxruntime.InferenceSession(args.recognizer_model)
    input_name = session.get_inputs()[0].name
    
    embeddings, labels = [], []
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"[INFO] Tìm thấy {len(person_dirs)} người. Bắt đầu trích xuất đặc trưng...")

    for person_dir in person_dirs:
        print(f" -> Đang xử lý: {person_dir.name}")
        image_paths = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
        
        # Cân bằng dữ liệu: Chỉ lấy tối đa 150 ảnh mỗi người nếu lỡ có nhiều hơn
        image_paths = image_paths[:150] 

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # Resize về chuẩn 112x112 nếu chưa đúng
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))

            emb = session.run(None, {input_name: preprocess(img)})[0]
            emb_norm = emb / np.linalg.norm(emb)
            
            embeddings.append(emb_norm.flatten())
            labels.append(person_dir.name)

    if not embeddings:
        print("[ERROR] Không tìm thấy dữ liệu.")
        return

    np.savez_compressed(args.output_file, embeddings=np.array(embeddings, dtype=np.float32), labels=np.array(labels))
    print(f"\n[SUCCESS] Đã huấn luyện xong! Dữ liệu lưu tại: {args.output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, default="datasets/faces")
    p.add_argument("--recognizer-model", type=str, default="w600k_r50.onnx")
    p.add_argument("--output-file", type=str, default="face_embeddings.npz")
    main(p.parse_args())