# Tên file: recognize_face_arcface.py
import cv2
import numpy as np
import onnxruntime
import os

def preprocess_for_recognition(face_image):
    """Chuẩn hóa ảnh cho model ArcFace."""
    if face_image.shape[:2] != (112, 112):
        face_image = cv2.resize(face_image, (112, 112))
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    normalized_image = (rgb_image.astype(np.float32) - 127.5) / 128.0
    transposed_image = np.transpose(normalized_image, [2, 0, 1])
    return np.expand_dims(transposed_image, axis=0)

class ArcFaceRecognizer:
    def __init__(self, model_path="w600k_r50.onnx", embeddings_path="face_embeddings.npz", threshold=0.4):
        print("[ArcFace] Đang khởi tạo...")
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            
            if not os.path.exists(embeddings_path):
                print(f"[CẢNH BÁO] Không tìm thấy file {embeddings_path}. Hãy chạy 2_face_training.py trước.")
                self.known_embeddings = None
                self.known_labels = None
            else:
                data = np.load(embeddings_path)
                self.known_embeddings = data['embeddings']
                self.known_labels = data['labels']
                print(f"[ArcFace] Đã tải dữ liệu của {len(np.unique(self.known_labels))} người.")
            
            self.threshold = threshold
        except Exception as e:
            print(f"[LỖI] Khởi tạo ArcFace thất bại: {e}")
            self.known_embeddings = None

    def recognize(self, face_crop):
        if self.known_embeddings is None or face_crop is None or face_crop.size == 0:
            return "Unknown", 0.0
        
        try:
            input_tensor = preprocess_for_recognition(face_crop)
            emb = self.session.run(None, {self.input_name: input_tensor})[0]
            emb_norm = emb / np.linalg.norm(emb)
            
            # Tính độ tương đồng Cosine
            scores = np.dot(emb_norm, self.known_embeddings.T).flatten()
            best_match_index = np.argmax(scores)
            best_score = scores[best_match_index]
            
            if best_score >= self.threshold:
                return self.known_labels[best_match_index], float(best_score)
            else:
                return "Unknown", float(best_score)
        except Exception:
            return "Error", 0.0

# Tạo instance toàn cục
recognizer_instance = ArcFaceRecognizer()

def recognize_face(face_crop):
    """Hàm wrapper để các file khác gọi."""
    return recognizer_instance.recognize(face_crop)