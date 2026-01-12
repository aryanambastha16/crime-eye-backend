import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
import face_recognition
import numpy as np

app = Flask(__name__)

CRIMINALS_DIR = "criminals"
CAPTURED_DIR = "captured"

known_encodings = []
known_names = []
known_images = {}

alerts = []

os.makedirs(CAPTURED_DIR, exist_ok=True)

print("[INFO] Loading criminal faces...")

for name in os.listdir(CRIMINALS_DIR):
    person_dir = os.path.join(CRIMINALS_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
            known_images[name] = f"/criminal_image/{name}/{img_name}"
            break

print(f"[INFO] Loaded criminals: {set(known_names)}")

def recognition_loop():
    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera not opened!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            distances = face_recognition.face_distance(known_encodings, encoding)

            if True in matches:
                best_match = np.argmin(distances)
                name = known_names[best_match]

                # FIXED: Convert numpy float to normal float
                confidence = float(round((1 - distances[best_match]) * 100, 2))

                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{time_str}.jpg"
                filepath = os.path.join(CAPTURED_DIR, filename)

                cv2.imwrite(filepath, frame)

                alert = {
                    "name": name,
                    "time": time_str,
                    "confidence": confidence,
                    "captured_image": f"/captured_image/{filename}",
                    "database_image": known_images.get(name, "")
                }

                alerts.append(alert)
                print("[ALERT]", alert)

        time.sleep(1)

@app.route("/alerts")
def get_alerts():
    return jsonify(alerts)

@app.route("/captured_image/<path:filename>")
def serve_captured(filename):
    return send_from_directory(CAPTURED_DIR, filename)

@app.route("/criminal_image/<name>/<path:filename>")
def serve_criminal(name, filename):
    return send_from_directory(os.path.join(CRIMINALS_DIR, name), filename)

def start_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    print("[INFO] Starting Flask server...")
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    print("[INFO] Starting recognition loop...")
    recognition_loop()
