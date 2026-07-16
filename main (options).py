import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
from tensorflow.keras.models import load_model

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

# ---------- SETTINGS ----------
FRAME_SKIP = 30        # face recognition every N frames
MATCH_THRESHOLD = 0.8  # relaxed threshold (IMPORTANT)

# ---------- CORE FUNCTION ----------
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

            x = max(0, x)
            y = max(0, y)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            # ---------- MASK DETECTION ----------
            face_resized = cv2.resize(face, (224, 224))
            face_resized = face_resized / 255.0
            face_resized = np.reshape(face_resized, (1, 224, 224, 3))

            pred = mask_model.predict(face_resized, verbose=0)[0]

            if pred[0] > 0.7:
                current_label = "Mask"
            elif pred[1] > 0.7:
                current_label = "No Mask"
            else:
                current_label = "Uncertain"

            # smoothing
            mask_history.append(current_label)
            if len(mask_history) > 10:
                mask_history.pop(0)

            mask_label = max(set(mask_history), key=mask_history.count)

            # ---------- FACE RECOGNITION (FIXED) ----------
            if frame_count % FRAME_SKIP == 0:
                try:
                    # IMPORTANT: use FULL FRAME (not cropped face)
                    result = DeepFace.find(
                        img_path=frame,
                        db_path=DATASET_PATH,
                        model_name="Facenet",
                        enforce_detection=False,
                        distance_metric="cosine"
                    )

                    if len(result) > 0 and len(result[0]) > 0:
                        best_match = result[0].iloc[0]

                        print("\n--- MATCH DEBUG ---")
                        print(best_match[['identity', 'distance']])

                        if best_match['distance'] < MATCH_THRESHOLD:
                            path = best_match['identity']
                            name = os.path.basename(os.path.dirname(path))
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"

                except Exception as e:
                    print("Recognition error:", e)
                    name = "Error"

            # ---------- DRAW ----------
            label = f"{name} ({mask_label})"
            thickness, font_scale = get_scale_factors(frame)

            # Draw box
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), thickness)

            # Background for text (optional but looks clean)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            cv2.rectangle(frame,
                        (x, y - text_h - 10),
                        (x + text_w, y),
                        (0, 255, 0), -1)

            # Draw text
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness)

    return frame, name


# ---------- IMAGE ----------
def run_image(image_path):
    print("Processing image...")

    frame = cv2.imread(image_path)
    name = "Unknown"
    mask_history = []

    frame, _ = process_frame(frame, 30, name, mask_history)

    out_path = os.path.join(OUTPUT_DIR, "output_image.jpg")
    cv2.imwrite(out_path, frame)

    display_frame = resize_for_display(frame)
    cv2.imshow("Image Result", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Saved:", out_path)


# ---------- VIDEO ----------
def run_video(video_path):
    print("Processing video...")

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(OUTPUT_DIR, "output_video.mp4")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    name = "Unknown"
    mask_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame, name = process_frame(frame, frame_count, name, mask_history)

        out.write(frame)
        display_frame = resize_for_display(frame)
        cv2.imshow("Video", display_frame)
        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved:", out_path)


# ---------- WEBCAM ----------
def run_webcam():
    print("Opening webcam...")
    # cap = cv2.VideoCapture(0)
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

        display_frame = resize_for_display(frame)
        cv2.imshow("Webcam", display_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def resize_for_display(frame, max_width=1000, max_height=700):
    h, w = frame.shape[:2]

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return frame

def get_scale_factors(frame):
    h, w = frame.shape[:2]

    scale = max(w, h) / 800   # base reference size

    thickness = max(2, int(2 * scale))
    font_scale = max(0.6, 0.8 * scale)

    return thickness, font_scale

# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n===== SELECT MODE =====")
    print("1. Image")
    print("2. Video")
    print("3. Webcam")
    print("0. Exit")

    choice = input("Enter your choice (0-3): ").strip()

    if choice == "1":
        path = input("Enter image path: ").strip()
        run_image(path)

    elif choice == "2":
        path = input("Enter video path: ").strip()
        run_video(path)

    elif choice == "3":
        run_webcam()

    elif choice == "0":
        print("Exiting...")

    else:
        print("Invalid choice! Please run again.")