# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')  # nhỏ, nhanh; pip sẽ tự tải nếu cần
# model.predict(source=0, show=True, conf=0.25)  # source=0 là webcam

#Xem tên các lớp đối tượng mà mô hình YOLO đã được huấn luyện để nhận diện
from ultralytics import YOLO
model = YOLO("runs/detect/train2/weights/best.pt")
print(model.names)

