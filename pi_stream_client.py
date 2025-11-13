# File: pi_stream_client.py
import asyncio
import websockets
import cv2
import time
from picamera2 import Picamera2

SERVER_IP = "10.131.231.201"
SERVER_URI = f"ws://{SERVER_IP}:8765"

async def video_streamer():
    print(f"Dang ket noi toi server: {SERVER_URI}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (480, 360)})
    picam2.configure(config)
    picam2.start()
    print("Pi Camera da khoi dong.")
    time.sleep(1.0)

    while True:
        try:
            async with websockets.connect(SERVER_URI, ping_interval=None) as websocket:
                print("Da ket noi toi WebSocket server.")
                while True:
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ret:
                        await websocket.send(buffer.tobytes())
                        await asyncio.sleep(0.05)
        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
            print(f"Mat ket noi: {e}. Thu lai sau 3 giay...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"Loi khong xac dinh: {e}")
            break

    picam2.stop()
    print("Dung chuong trinh.")

if __name__ == '__main__':
    asyncio.run(video_streamer())
