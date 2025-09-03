#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera smoke test
Mở webcam và hiển thị khung hình, FPS. Nhấn 'q' để thoát.
Chạy: python src/cam_test.py --camera 0
"""
import argparse, time, cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[!] Không mở được camera")
        return

    print("[i] Nhấn 'q' để thoát.")
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t = time.time()
        fps = frames / (t - t0 + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
