import cv2
print("cv2:", cv2.__version__)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
from ultralytics import YOLO
print("ultralytics import OK")
from deepface import DeepFace
print("deepface import OK")
