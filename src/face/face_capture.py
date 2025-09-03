#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face sample capture - Week 4 helper
Chụp ảnh khuôn mặt để làm dữ liệu nhận diện.
Nhấn 'c' để chụp, 'q' để thoát.

Chạy:
python src/face_capture.py --out_dir ./data/faces --student_id HS_01 --camera 0
"""
import argparse
import os
import cv2
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./data/faces")
    parser.add_argument("--student_id", type=str, default="HS_01")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.student_id)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[!] Không mở được camera")
        return

    print("[i] Nhấn 'c' để chụp, 'q' để thoát.")
    counter = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, f"Save dir: {save_dir}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Face capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            cv2.imwrite(os.path.join(save_dir, fname), frame)
            counter += 1
            print(f"[+] Saved: {fname} (total {counter})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
