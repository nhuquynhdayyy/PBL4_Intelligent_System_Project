# test_deepface_basic.py
# Script kiểm tra mô hình nhận diện khuôn mặt cơ bản từ ảnh
# ---------------------------------------------

from deepface import DeepFace
import cv2

# 1️⃣ Đặt đường dẫn đến 2 ảnh cần so sánh
img1_path = "datasets/faces/006_NguyenNhuQuynh/006_NguyenNhuQuynh_1.jpg"   # ảnh khuôn mặt thứ 1
img2_path = "datasets/faces/003_HoNguyenThaoNguyen/003_HoNguyenThaoNguyen_1.jpg"   # ảnh khuôn mặt thứ 2

# 2️⃣ Hiển thị 2 ảnh để kiểm tra
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
cv2.imshow("Ảnh 1", img1)
cv2.imshow("Ảnh 2", img2)
cv2.waitKey(500)  # hiển thị 0.5s

# 3️⃣ So sánh 2 khuôn mặt
print("[INFO] Đang so sánh 2 khuôn mặt...")
result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name="Facenet")

# 4️⃣ In kết quả
if result["verified"]:
    print("✅ Hai ảnh là cùng một người!")
else:
    print("❌ Hai ảnh là người khác!")

print("Độ tương đồng (distance):", round(result["distance"], 3))
print("Ngưỡng xác nhận:", result["threshold"])
print("Chi tiết:", result)

cv2.destroyAllWindows()
