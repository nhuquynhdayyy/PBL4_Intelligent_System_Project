# check_model.py
import os
import onnxruntime as ort

model_file = "w600k_r50.onnx"

print(f"--- ĐANG KIỂM TRA FILE: {model_file} ---")

if not os.path.exists(model_file):
    print("❌ LỖI: Không tìm thấy file trong thư mục này!")
else:
    # 1. Kiểm tra dung lượng
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"📦 Dung lượng file: {size_mb:.2f} MB")

    if size_mb < 100:
        print("⚠️ CẢNH BÁO: File quá nhẹ! Model chuẩn phải > 160MB.")
        print("   -> Có thể bạn đang copy nhầm file shortcut hoặc file pointer của Git LFS.")
    
    # 2. Thử load bằng ONNX Runtime
    print("⚙️ Đang thử load model...")
    try:
        sess = ort.InferenceSession(model_file)
        print("✅ KẾT QUẢ: Model OK! File hoạt động tốt.")
    except Exception as e:
        print("❌ KẾT QUẢ: Model BỊ LỖI.")
        print("   -> Lỗi chi tiết:", e)