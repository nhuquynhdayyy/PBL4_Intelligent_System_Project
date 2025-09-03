#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera resolution & FPS test
Thử nhiều độ phân giải (320x240, 640x480, 1280x720, 1920x1080) và đo FPS.
Nhấn 'q' để thoát tại bất kỳ độ phân giải nào, script sẽ chuyển sang độ phân giải tiếp theo.

Chạy:
python src/cam_resolution_test.py --camera 0
"""
import argparse, time, cv2

def run_test(cam_index, width, height):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[!] Không mở được camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"[i] Testing {width}x{height}. Nhấn 'q' để chuyển sang cấu hình tiếp theo.")
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t = time.time()
        fps = frames / (t - t0 + 1e-9)
        cv2.putText(frame, f"{width}x{height} FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Camera Resolution Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    resolutions = [(320,240), (640,480), (1280,720), (1920,1080)]
    for (w,h) in resolutions:
        run_test(args.camera, w, h)

if __name__ == "__main__":
    main()
