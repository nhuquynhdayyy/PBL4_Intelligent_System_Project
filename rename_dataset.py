import os

def rename_valid_set():
    # Đường dẫn tới thư mục valid
    image_dir = "action_data/valid/images"
    label_dir = "action_data/valid/labels"

    # Lấy danh sách file ảnh (hỗ trợ .jpg, .png, .jpeg)
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    images.sort() # Sắp xếp để đổi tên theo thứ tự

    print(f"[INFO] Đang đổi tên {len(images)} file...")

    for i, old_name in enumerate(images):
        # Tạo tên mới: 001, 002, 003...
        new_base_name = f"{i+1:03d}" 
        
        # Lấy đuôi mở rộng (ví dụ: .jpg)
        ext = os.path.splitext(old_name)[1]
        
        # 1. Đổi tên ảnh
        old_image_path = os.path.join(image_dir, old_name)
        new_image_path = os.path.join(image_dir, new_base_name + ext)
        os.rename(old_image_path, new_image_path)

        # 2. Đổi tên file nhãn .txt tương ứng
        old_label_name = os.path.splitext(old_name)[0] + ".txt"
        old_label_path = os.path.join(label_dir, old_label_name)
        new_label_path = os.path.join(label_dir, new_base_name + ".txt")

        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)

    print("[SUCCESS] Đã đổi tên xong toàn bộ thư mục Valid!")

if __name__ == "__main__":
    rename_valid_set()