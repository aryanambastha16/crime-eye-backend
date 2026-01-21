import cv2
import requests
import time

CLOUD_URL = CLOUD_URL = "http://127.0.0.1:5000/upload_frame"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    _, buffer = cv2.imencode(".jpg", frame)
    response = requests.post(
        CLOUD_URL,
        files={"image": buffer.tobytes()}
    )

    try:
        print(response.json())
    except:
        print("No response")

    time.sleep(1)
