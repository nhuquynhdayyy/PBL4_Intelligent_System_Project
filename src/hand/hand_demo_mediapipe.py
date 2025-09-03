#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand demo (MediaPipe) - Week 4 sample
Hiển thị số lượng bàn tay phát hiện được + vẽ landmarks.
Yêu cầu: pip install mediapipe opencv-contrib-python

Chạy:
python src/hand_demo_mediapipe.py --camera 0
Nhấn 'q' để thoát.
"""
import argparse
import cv2
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min_det", type=float, default=0.5)
    parser.add_argument("--min_track", type=float, default=0.5)
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[!] Không mở được camera")
        return

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track
    ) as hands:
        print("[i] Nhấn 'q' để thoát.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    count += 1
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            cv2.putText(frame, f"Hands: {count}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Week4 - MediaPipe Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
