import os
import pandas as pd
import cv2
from tqdm import tqdm
from recognize_face import recognize_face

# Tắt cảnh báo TensorFlow để nhìn log sạch hơn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def validate_system():
    test_dir = "datasets/faces" 
    results = []

    # Lấy danh sách tất cả ảnh trước để đếm tổng số
    all_images = []
    for student_id in os.listdir(test_dir):
        student_path = os.path.join(test_dir, student_id)
        if os.path.isdir(student_path):
            for img_name in os.listdir(student_path):
                all_images.append((student_id, img_name, os.path.join(student_path, img_name)))

    print(f"[INFO] Bắt đầu Validation cho {len(all_images)} ảnh...")

    # Sử dụng tqdm để hiện thanh tiến trình
    for actual_id, img_name, img_path in tqdm(all_images, desc="Đang kiểm tra"):
        img = cv2.imread(img_path)
        if img is None: continue

        # QUAN TRỌNG: Trong hàm recognize_face, hãy đảm bảo bạn dùng detector_backend='skip'
        pred_id, confidence = recognize_face(img)

        is_correct = (pred_id == actual_id)
        results.append({
            'Actual_ID': actual_id,
            'Predicted_ID': pred_id,
            'Confidence': confidence,
            'Result': 'Match' if is_correct else 'Mismatch'
        })

    # Xuất Excel (giữ nguyên đoạn xuất cũ)
    df = pd.DataFrame(results)
    df.to_excel("face_recognition_report.xlsx", index=False)
    accuracy = (df['Result'] == 'Match').mean()
    print(f"\n[SUCCESS] Hoàn tất! Độ chính xác: {accuracy*100:.2f}%")

if __name__ == "__main__":
    validate_system()