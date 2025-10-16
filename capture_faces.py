import cv2, os

student_id = input("Nhập ID và tên học sinh (VD: 001_NguyenVanA): ")
save_dir = f"datasets/faces/{student_id}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
print("[INFO] Nhấn phím SPACE để chụp, ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Chup khuon mat", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        count += 1
        path = f"{save_dir}/{student_id}_{count}.jpg"
        cv2.imwrite(path, frame)
        print(f"[OK] Đã lưu: {path}")

cap.release()
cv2.destroyAllWindows()
