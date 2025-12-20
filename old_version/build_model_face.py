# build_model_face.py
# Lưu model pre-trained của DeepFace (Facenet) ra file .pt
# Dùng để kiểm tra và tái sử dụng model sau này
# Lưu ý: DeepFace sử dụng TensorFlow/Keras làm backend, không phải PyTorch,
# nên việc lưu bằng torch.save() có thể không hoạt động như mong đợi.
# Tuy nhiên, ta vẫn có thể lưu model để tham khảo.
# Nếu cần sử dụng lại, DeepFace sẽ tự động tải từ cache.
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
