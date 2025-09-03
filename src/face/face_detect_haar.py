#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face detection demo (OpenCV Haar) - Week 4 sample
Hiển thị số lượng khuôn mặt và vẽ bbox.
Yêu cầu: opencv-contrib-python

Chạy:
python src/face_detect_haar.py --camera 0
Nhấn 'q' để thoát.
"""
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.2, help="scaleFactor")
    parser.add_argument("--neighbors", type=int, default=5, help="minNeighbors")
    args = parser.parse_args()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[!] Không mở được camera")
        return

    print("[i] Nhấn 'q' để thoát.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=args.scale, minNeighbors=args.neighbors, minSize=(60,60))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Week4 - Face Haar", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
