# build_model_face.py
from deepface import DeepFace
import torch

print("[INFO] Đang tải model pre-trained (Facenet)...")
model = DeepFace.build_model("Facenet")
print("✅ Model loaded successfully!")

# In ra tên lớp model để xác nhận
print(f"Model type: {type(model)}")

# Lưu model ra file .pt (chỉ lưu trạng thái trọng số nếu có)
try:
    torch.save(model.state_dict(), "model_face.pt")
    print("✅ Đã lưu model pre-trained vào file model_face.pt")
except Exception as e:
    print("⚠️ Không thể lưu bằng torch.save() (model không phải PyTorch).")
    print("→ DeepFace đang dùng backend TensorFlow/Keras, không có state_dict().")
    print("→ Không sao cả — model đã cache trong ~/.deepface/weights và có thể load lại trực tiếp.")
