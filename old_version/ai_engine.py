# File: ai_engine.py
import cv2
import numpy as np
import onnxruntime
import os

class FaceRecognitionSystem:
    def __init__(self, 
                 detector_path="face_detector.onnx", 
                 recognizer_path="w600k_r50.onnx", 
                 embeddings_path="face_embeddings.npz",
                 threshold=0.4): # Ngưỡng nhận diện (0.4 - 0.5 là ổn)
        
        print("[AI ENGINE] Đang khởi tạo các model...")
        
        # 1. Load Face Detector (YuNet)
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Thiếu file model: {detector_path}")
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), 0.8)
        
        # 2. Load ArcFace Recognizer (ONNX)
        if not os.path.exists(recognizer_path):
            raise FileNotFoundError(f"Thiếu file model: {recognizer_path}")
        self.session = onnxruntime.InferenceSession(recognizer_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # 3. Load Embeddings Database
        self.known_embeddings = None
        self.known_labels = None
        self.load_embeddings(embeddings_path)
        
        self.threshold = threshold
        print("[AI ENGINE] Hệ thống sẵn sàng!")

    def load_embeddings(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.known_embeddings = data['embeddings']
            self.known_labels = data['labels']
            print(f"[AI ENGINE] Đã tải {len(self.known_labels)} khuôn mặt.")
        else:
            print("[WARNING] Chưa có file embeddings. Hãy chạy training trước.")

    def align_face(self, frame, landmarks):
        """Xoay khuôn mặt cho thẳng dựa trên mắt."""
        pts = landmarks.copy()
        left_eye = pts[0]  # YuNet: điểm 0 là mắt trái
        right_eye = pts[1] # YuNet: điểm 1 là mắt phải
        
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        (h, w) = frame.shape[:2]
        aligned = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return aligned

    def preprocess(self, face_img):
        """Chuẩn hóa ảnh 112x112 về [-1, 1]"""
        face_img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = (img.astype(np.float32) - 127.5) / 128.0
        return img

    def recognize(self, frame_crop):
        """
        Nhận vào một vùng ảnh crop (ví dụ vùng đầu), 
        Tự detect lại khuôn mặt bên trong -> Align -> Recognize
        """
        if self.known_embeddings is None:
            return "SystemNotReady", 0.0

        h, w = frame_crop.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_crop)
        
        if faces is None or len(faces) == 0:
            return "NoFace", 0.0

        # Lấy khuôn mặt to nhất trong vùng crop
        face = faces[np.argmax([f[2]*f[3] for f in faces])]
        landmarks = face[4:14].reshape(5, 2).astype(np.int32)
        
        # Căn chỉnh (Alignment)
        aligned_crop = self.align_face(frame_crop, landmarks)
        
        # detect lại trên ảnh đã align để cắt chính xác bbox
        self.detector.setInputSize((w, h))
        _, faces2 = self.detector.detect(aligned_crop)
        if faces2 is None: return "AlignError", 0.0
        
        face2 = faces2[0]
        x, y, w_f, h_f = face2[0:4].astype(int)
        final_face = aligned_crop[y:y+h_f, x:x+w_f]
        
        if final_face.size == 0: return "CropError", 0.0
        
        # Inference ArcFace
        input_tensor = self.preprocess(final_face)
        embedding = self.session.run(None, {self.input_name: input_tensor})[0]
        embedding = embedding / np.linalg.norm(embedding) # Chuẩn hóa vector
        
        # So sánh Cosine Similarity
        scores = np.dot(embedding, self.known_embeddings.T).flatten()
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score >= self.threshold:
            return self.known_labels[best_idx], float(best_score)
        else:
            return "Unknown", float(best_score)