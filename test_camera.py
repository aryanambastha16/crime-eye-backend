import cv2

for i in range(5):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"Camera {i} is working")
        cap.release()
    else:
        print(f"Camera {i} not working")
