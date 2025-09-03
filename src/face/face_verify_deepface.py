#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepFace verify - Week 4 sample
So sánh 2 ảnh xem có phải cùng người.

Chạy:
python src/face_verify_deepface.py --img1 path/to/a.jpg --img2 path/to/b.jpg --model Facenet512 --detector opencv
"""
import argparse
from deepface import DeepFace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--model", type=str, default="Facenet512")
    parser.add_argument("--detector", type=str, default="opencv")
    args = parser.parse_args()

    print(f"[i] Comparing: {args.img1} vs {args.img2}")
    obj = DeepFace.verify(img1_path=args.img1, img2_path=args.img2,
                          model_name=args.model, detector_backend=args.detector, enforce_detection=False)
    # obj keys: verified, distance, threshold, model, detector_backend, time, etc.
    print("verified:", obj.get("verified"))
    print("distance:", obj.get("distance"))
    print("threshold:", obj.get("threshold"))
    print("model:", obj.get("model"))
    print("detector_backend:", obj.get("detector_backend"))

if __name__ == "__main__":
    main()
