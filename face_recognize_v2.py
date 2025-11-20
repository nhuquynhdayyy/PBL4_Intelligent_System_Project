import cv2
import numpy as np
import onnxruntime
import argparse

# --- CÁC HÀM HỖ TRỢ (Copy từ các file trước) ---

def preprocess_for_recognition(face_image):
    """Tiền xử lý ảnh khuôn mặt trước khi đưa vào model ArcFace."""
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    normalized_image = (rgb_image.astype(np.float32) - 127.5) / 128.0
    transposed_image = np.transpose(normalized_image, [2, 0, 1])
    input_tensor = np.expand_dims(transposed_image, axis=0)
    return input_tensor

def align_face(frame, landmarks):
    """Căn chỉnh khuôn mặt dựa trên vị trí hai mắt."""
    pts = landmarks.copy()
    order = pts[np.argsort(pts[:, 0])]
    left_eye = order[0]
    right_eye = order[-1]
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int((left_eye[1] + right_eye[1]) / 2)
    eye_center = (center_x, center_y)
    
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    (h, w) = frame.shape[:2]
    aligned = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return aligned

# --- HÀM CHÍNH ---

def main(args):
    # 1. Tải cơ sở dữ liệu khuôn mặt đã "huấn luyện"
    try:
        data = np.load(args.embeddings_file)
        known_embeddings = data['embeddings']
        known_labels = data['labels']
        print(f"[INFO] Đã tải {len(known_labels)} embeddings từ '{args.embeddings_file}'.")
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file embeddings: {args.embeddings_file}. Hãy chạy 'face_training_v2.py' trước.")
        return

    # 2. Khởi tạo các model ONNX
    print("[INFO] Đang tải các model ONNX...")
    try:
        detector = cv2.FaceDetectorYN.create(args.detector_model, "", (320, 320), 0.8)
        recognizer_session = onnxruntime.InferenceSession(args.recognizer_model)
        recognizer_input_name = recognizer_session.get_inputs()[0].name
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải model ONNX: {e}")
        return
    print("[INFO] Các model đã sẵn sàng.")

    # 3. Mở camera
    cap = cv2.VideoCapture(0) # 0 là webcam mặc định
    if not cap.isOpened():
        print("[ERROR] Không thể mở camera.")
        return

    # --- VÒNG LẶP XỬ LÝ REAL-TIME ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Không nhận được frame từ camera, có thể camera đã bị ngắt kết nối.")
            break

        h_frame, w_frame = frame.shape[:2]
        detector.setInputSize((w_frame, h_frame))

        # 4. Phát hiện tất cả khuôn mặt trong frame
        _, faces = detector.detect(frame)
        faces = faces if faces is not None else []

        # 5. Với mỗi khuôn mặt phát hiện được...
        for face_data in faces:
            box = face_data[0:4].astype(np.int32)
            landmarks = face_data[4:14].reshape(5, 2).astype(np.int32)
            (x, y, w, h) = box

            # Cắt và căn chỉnh khuôn mặt
            aligned_face = align_face(frame, landmarks)
            
            # Detect lại trên ảnh đã align để lấy bbox chuẩn
            h_aligned, w_aligned = aligned_face.shape[:2]
            detector.setInputSize((w_aligned, h_aligned))
            _, faces2 = detector.detect(aligned_face)
            if faces2 is None or len(faces2) == 0:
                continue

            box2 = faces2[0][0:4].astype(np.int32)
            (ax, ay, aw, ah) = box2
            face_crop = aligned_face[ay:ay+ah, ax:ax+aw]
            
            if face_crop.size == 0:
                continue
            
            # Resize về kích thước chuẩn của model nhận diện
            face_crop_resized = cv2.resize(face_crop, (112, 112))
            
            # 6. Trích xuất embedding của khuôn mặt hiện tại
            input_tensor = preprocess_for_recognition(face_crop_resized)
            current_embedding = recognizer_session.run(None, {recognizer_input_name: input_tensor})[0]
            current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)
            
            # 7. So sánh embedding hiện tại với toàn bộ cơ sở dữ liệu
            # Dùng phép nhân ma trận (dot product) vì các vector đã được chuẩn hóa L2
            # Kết quả sẽ là cosine similarity
            scores = np.dot(current_embedding_norm, known_embeddings.T)
            
            best_match_index = np.argmax(scores)
            best_score = scores[0][best_match_index]
            
            # 8. Ra quyết định
            if best_score >= args.threshold:
                name = known_labels[best_match_index]
                confidence = best_score
                display_text = f"{name}: {confidence:.2f}"
                color = (0, 255, 0) # Xanh lá cho người quen
            else:
                name = "Unknown"
                display_text = f"{name}: {best_score:.2f}"
                color = (0, 0, 255) # Đỏ cho người lạ

            # 9. Vẽ kết quả lên frame gốc
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 10. Hiển thị frame
        cv2.imshow("Face Recognition", frame)

        # Thoát bằng phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nhận diện khuôn mặt real-time bằng ArcFace.")
    parser.add_argument("--detector-model", type=str, default="face_detector.onnx", help="Đường dẫn tới model phát hiện mặt (YuNet/SCRFD).")
    parser.add_argument("--recognizer-model", type=str, default="w600k_r50.onnx", help="Đường dẫn tới model nhận diện mặt (ArcFace).")
    parser.add_argument("--embeddings-file", type=str, default="face_embeddings.npz", help="Đường dẫn tới file embeddings đã 'huấn luyện'.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhận diện (cosine similarity).")
    args = parser.parse_args()
    main(args)