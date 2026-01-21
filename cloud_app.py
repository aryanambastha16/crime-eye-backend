import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from datetime import datetime
from criminal_db import criminal_profiles

app = Flask(__name__)

CRIMINALS_DIR = "criminals"
THRESHOLD = 0.45

known_encodings = []
known_names = []

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
            break

print(f"[INFO] Loaded criminals: {set(known_names)}")

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    if len(encodings) == 0:
        return jsonify({"status": "no_face"})

    encoding = encodings[0]
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_match = np.argmin(distances)
    best_distance = distances[best_match]

    if best_distance >= THRESHOLD:
        return jsonify({"status": "unknown"})

    name = known_names[best_match]
    confidence = float(round((1 - best_distance) * 100, 2))
    profile = criminal_profiles.get(name, {})

    result = {
        "status": "criminal_detected",
        "name": name,
        "confidence": confidence,
        "age": profile.get("age", "N/A"),
        "blood_group": profile.get("blood_group", "N/A"),
        "crime_history": profile.get("crime_history", []),
        "level": profile.get("level", "N/A"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
