import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
from tensorflow.keras.models import load_model
import json

# ---------- PATHS ----------
MASK_MODEL_PATH = r"C:\Project\code\mask_detector.model"
DATASET_PATH = r"C:\Project\code\dataset"
OUTPUT_DIR = r"C:\Project\code\output_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD MODELS ----------
print("Loading mask model...")
mask_model = load_model(MASK_MODEL_PATH)
print("Mask model loaded!")

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---------- EYE LANDMARKS ----------
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_POINTS = [362, 385, 387, 263, 373, 380]

# ---------- SETTINGS ----------
FRAME_SKIP = 30
MATCH_THRESHOLD = 0.8

# ---------- HELPERS ----------
def get_iris(lm, eye, iris, w, h):
    x1 = int(lm[eye[0]].x * w)
    x2 = int(lm[eye[1]].x * w)
    ix = int(np.mean([lm[p].x for p in iris]) * w)
    iy = int(np.mean([lm[p].y for p in iris]) * h)
    return x1, x2, ix, iy

def gaze_dir(x1, x2, ix):
    c = (x1 + x2) / 2
    if ix < c - 5: return "LEFT"
    elif ix > c + 5: return "RIGHT"
    else: return "CENTER"

def ear(lm, pts, w, h):
    coords = [(int(lm[p].x*w), int(lm[p].y*h)) for p in pts]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A + B) / (2.0 * C)

def save_eye_features(name, features):
    folder = os.path.join(DATASET_PATH, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "eye.json")

    with open(path, "w") as f:
        json.dump(features, f, indent=4)

    print(f"✅ Saved eye features in {path}")

# ---------- FACE PROCESS ----------
def process_frame(frame, frame_count, name_state, mask_history):
    name = name_state

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

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            # MASK
            face_resized = cv2.resize(face, (224, 224)) / 255.0
            face_resized = np.reshape(face_resized, (1, 224, 224, 3))
            pred = mask_model.predict(face_resized, verbose=0)[0]

            label = "Mask" if pred[0] > 0.7 else "No Mask"
            mask_history.append(label)
            if len(mask_history) > 10:
                mask_history.pop(0)
            label = max(set(mask_history), key=mask_history.count)

            # FACE RECOG
            if frame_count % FRAME_SKIP == 0:
                try:
                    result = DeepFace.find(
                        img_path=frame,
                        db_path=DATASET_PATH,
                        model_name="Facenet",
                        enforce_detection=False
                    )
                    if len(result) > 0 and len(result[0]) > 0:
                        best = result[0].iloc[0]
                        if best['distance'] < MATCH_THRESHOLD:
                            name = os.path.basename(os.path.dirname(best['identity']))
                        else:
                            name = "Unknown"
                except:
                    name = "Error"

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({label})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, name

# ---------- OPTION 3: FACE + EYE ----------
def run_webcam():
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    frame_count = 0
    name = "Unknown"
    mask_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame, name = process_frame(frame, frame_count, name, mask_history)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_mesh = face_mesh.process(rgb)

        if res_mesh.multi_face_landmarks:
            lm = res_mesh.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            x1, x2, ix, iy = get_iris(lm, LEFT_EYE, LEFT_IRIS, w, h)
            rx1, rx2, rix, riy = get_iris(lm, RIGHT_EYE, RIGHT_IRIS, w, h)

            # draw both eyes
            cv2.circle(frame, (ix, iy), 3, (0,0,255), -1)
            cv2.circle(frame, (rix, riy), 3, (0,0,255), -1)

            cv2.line(frame, ((x1+x2)//2, iy), (ix, iy), (255,0,255), 2)
            cv2.line(frame, ((rx1+rx2)//2, riy), (rix, riy), (255,0,255), 2)
            
            # ===== DRAW LEFT EYE LANDMARKS =====
            for p in LEFT_EAR_POINTS:
                px = int(lm[p].x * w)
                py = int(lm[p].y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # ===== DRAW RIGHT EYE LANDMARKS =====
            for p in RIGHT_EAR_POINTS:
                px = int(lm[p].x * w)
                py = int(lm[p].y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)            

        cv2.imshow("Webcam Face + Eye", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- OPTION 4: FULL EYE DATASET ----------
def run_eye_dataset():
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    gaze_history = []
    blink_counter = 0
    blink_total = 0

    print("Press S to save")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            x1, x2, ix, iy = get_iris(lm, LEFT_EYE, LEFT_IRIS, w, h)
            rx1, rx2, rix, riy = get_iris(lm, RIGHT_EYE, RIGHT_IRIS, w, h)

            gazeL = gaze_dir(x1, x2, ix)
            gazeR = gaze_dir(rx1, rx2, rix)
            gaze = gazeL if gazeL == gazeR else "CENTER"

            gaze_history.append(gaze)
            if len(gaze_history) > 30:
                gaze_history.pop(0)

            e = (ear(lm, LEFT_EAR_POINTS, w, h) + ear(lm, RIGHT_EAR_POINTS, w, h)) / 2

            if e < 0.21:
                blink_counter += 1
            else:
                if blink_counter > 2:
                    blink_total += 1
                blink_counter = 0

            transitions = sum(1 for i in range(1,len(gaze_history)) if gaze_history[i]!=gaze_history[i-1])

            current_features = {
                "blink_count": blink_total,
                "transitions": transitions,
                "stability": 0
            }

            # visuals
            cv2.circle(frame, (ix, iy), 3, (0,0,255), -1)
            cv2.circle(frame, (rix, riy), 3, (0,0,255), -1)

            cv2.line(frame, ((x1+x2)//2, iy), (ix, iy), (255,0,255), 2)
            cv2.line(frame, ((rx1+rx2)//2, riy), (rix, riy), (255,0,255), 2)
            
            # ===== DRAW LEFT EYE LANDMARKS =====
            for p in LEFT_EAR_POINTS:
                px = int(lm[p].x * w)
                py = int(lm[p].y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

            # ===== DRAW RIGHT EYE LANDMARKS =====
            for p in RIGHT_EAR_POINTS:
                px = int(lm[p].x * w)
                py = int(lm[p].y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)            

            cv2.putText(frame, f"Gaze: {gaze}", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame, f"Blinks: {blink_total}", (30,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame, f"Transitions: {transitions}", (30,130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        cv2.imshow("Eye Dataset Capture", frame)

        key = cv2.waitKey(10)

        if key != -1:
            print("Key:", key)  # debug

        if key == ord('s') or key == ord('S'):
            print("S pressed!")
            name = input("Enter name: ")
            save_eye_features(name, current_features)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n===== SELECT MODE =====")
    print("1. Image")
    print("2. Video")
    print("3. Webcam")
    print("4. Capture Eye Dataset")
    print("0. Exit")

    choice = input("Enter your choice (0-4): ").strip()

    if choice == "3":
        run_webcam()
    elif choice == "4":
        run_eye_dataset()
    else:
        print("Image/Video unchanged from your original code")