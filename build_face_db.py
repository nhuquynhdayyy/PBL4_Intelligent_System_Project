# from deepface import DeepFace
# import os, pickle

# dataset_path = "datasets/faces"
# face_db = {}

# print("[INFO] Bắt đầu trích xuất đặc trưng khuôn mặt...")

# for student_dir in os.listdir(dataset_path):
#     student_path = os.path.join(dataset_path, student_dir)
#     if not os.path.isdir(student_path):
#         continue
#     print(f"--> {student_dir}")
    
#     embeddings = []
#     for img_name in os.listdir(student_path):
#         img_path = os.path.join(student_path, img_name)
#         try:
#             emb = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
#             embeddings.append(emb[0]["embedding"])
#         except Exception as e:
#             print(f"[Lỗi] {img_name}: {e}")
#             continue
    
#     if embeddings:
#         face_db[student_dir] = embeddings

# # Lưu database vào file pkl
# with open("face_db.pkl", "wb") as f:
#     pickle.dump(face_db, f)

# print(f"[DONE] Đã tạo face_db.pkl với {len(face_db)} học sinh.")


from deepface import DeepFace
import os, pickle

dataset_path = "datasets/faces"
face_db = {}

print("[INFO] Trích xuất đặc trưng khuôn mặt học sinh...")

for student in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student)
    if not os.path.isdir(student_path):
        continue
    
    embeddings = []
    for img_file in os.listdir(student_path):
        img_path = os.path.join(student_path, img_file)
        try:
            emb = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
            embeddings.append(emb[0]["embedding"])
        except Exception as e:
            print(f"[Lỗi] {img_file}: {e}")
    
    if embeddings:
        face_db[student] = embeddings

with open("face_db.pkl", "wb") as f:
    pickle.dump(face_db, f)

print(f"[DONE] Đã lưu face_db.pkl với {len(face_db)} học sinh.")

