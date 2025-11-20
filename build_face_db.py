# File: build_face_db.py
import os
import pickle
from deepface import DeepFace

# --- CẤU HÌNH QUAN TRỌNG ---
DATASET_PATH = "datasets/faces"   # Thư mục chứa ảnh
OUTPUT_FILE = "face_db.pkl"       # Tên file kết quả
MODEL_NAME = "ArcFace"            # Model nhận diện tốt nhất hiện nay (thay vì Facenet)
DETECTOR_BACKEND = "opencv"       # Backend phát hiện khuôn mặt

def build_database():
    face_db = {}
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(DATASET_PATH):
        print(f"[LỖI] Không tìm thấy thư mục: {DATASET_PATH}")
        print("Hãy đảm bảo bạn đang đứng đúng thư mục gốc của dự án.")
        return

    print(f"[INFO] Bắt đầu xử lý dữ liệu từ: {DATASET_PATH}")
    print(f"[INFO] Sử dụng Model: {MODEL_NAME}")

    # Lấy danh sách các thư mục con (Tên sinh viên)
    student_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    total_students = len(student_folders)
    
    print(f"--> Tìm thấy {total_students} thư mục sinh viên.")

    for idx, student_id in enumerate(student_folders):
        student_path = os.path.join(DATASET_PATH, student_id)
        print(f"[{idx+1}/{total_students}] Đang xử lý: {student_id}...")
        
        embeddings = []
        
        # Duyệt qua từng file trong thư mục của sinh viên đó
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            
            # Chỉ nhận file ảnh
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            try:
                # --- ĐIỂM QUAN TRỌNG NHẤT ---
                # enforce_detection=True: Bắt buộc ảnh phải có mặt rõ ràng mới học.
                # Nếu ảnh mờ, tối, hoặc chụp cái lưng -> DeepFace sẽ báo lỗi -> Ta bỏ qua ảnh đó.
                results = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True, 
                    detector_backend=DETECTOR_BACKEND
                )
                
                # Lưu vector đặc trưng (embedding)
                if results:
                    embeddings.append(results[0]["embedding"])
                    
            except ValueError:
                print(f"    [Bỏ qua] {img_name}: Không tìm thấy khuôn mặt.")
            except Exception as e:
                print(f"    [Lỗi] {img_name}: {e}")

        # Chỉ lưu vào DB nếu học được ít nhất 1 ảnh
        if embeddings:
            face_db[student_id] = embeddings
            print(f"    -> [OK] Đã học {len(embeddings)} vector khuôn mặt.")
        else:
            print(f"    -> [CẢNH BÁO] Không học được ảnh nào từ {student_id}!")

    # Lưu kết quả ra file .pkl
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(face_db, f)
    
    print("\n" + "="*40)
    print(f"[HOÀN TẤT] Đã lưu cơ sở dữ liệu vào file: {OUTPUT_FILE}")
    print(f"Tổng số người trong DB: {len(face_db)}")
    print("="*40)

if __name__ == "__main__":
    build_database()