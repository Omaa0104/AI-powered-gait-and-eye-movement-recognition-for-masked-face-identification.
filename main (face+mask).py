import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
from tensorflow.keras.models import load_model

# ---------- LOAD MODELS ----------
print("Loading mask model...")
mask_model = load_model(r"D:\Gait and eye Recognition\mask_detector.model")
print("Mask model loaded!")

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

print("Opening camera...")
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

print("Starting loop...")

# ---------- VARIABLES ----------
frame_count = 0
name = "Unknown"
mask_label = "Detecting..."
mask_history = []

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    frame_count += 1

    # Convert for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Safe crop
            x = max(0, x)
            y = max(0, y)
            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            # ---------- MASK DETECTION ----------
            face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            face_resized = face_resized / 255.0
            face_resized = np.reshape(face_resized, (1, 224, 224, 3))

            pred = mask_model.predict(face_resized, verbose=0)[0]

            mask_prob = pred[0]
            no_mask_prob = pred[1]

            if mask_prob > 0.7:
                current_label = "Mask"
            elif no_mask_prob > 0.7:
                current_label = "No Mask"
            else:
                current_label = "Uncertain"

            # ---------- SMOOTHING ----------
            mask_history.append(current_label)
            if len(mask_history) > 10:
                mask_history.pop(0)

            mask_label = max(set(mask_history), key=mask_history.count)

            # ---------- FACE RECOGNITION ----------
            if frame_count % 30 == 0:
                try:
                    result = DeepFace.find(
                        img_path=face,
                        db_path="dataset/",
                        model_name="Facenet",
                        enforce_detection=False
                    )

                    if len(result[0]) > 0:
                        path = result[0]['identity'][0]
                        name = os.path.basename(os.path.dirname(path))
                    else:
                        name = "Unknown"

                except:
                    name = "Error"

            # ---------- DISPLAY ----------
            label = f"{name} ({mask_label})"

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    cv2.imshow("Final System: Face + Mask", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

