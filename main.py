import os
import sys
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Try to use a local trained model `best.pt` if present, otherwise fall back to
# the small pretrained model `yolov8n.pt` (ultralytics will download it).
MODEL_FILES = ["best.pt", "yolov8n.pt"]
model_path = None
for m in MODEL_FILES:
    if os.path.exists(m):
        model_path = m
        break
if model_path is None:
    model_path = "yolov8n.pt"

model = None
if YOLO is not None:
    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Fehler beim Laden des Modells:", e)
        model = None

cap = cv2.VideoCapture(0)  # adjust camera index if needed
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

print("Drücke 'q' zum Beenden. Modell:", model_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if model is not None:
        # Run inference on the BGR frame (ultralytics accepts numpy arrays)
        try:
            results = model(frame, conf=0.25, verbose=False)
            r = results[0]
            if hasattr(r, 'boxes') and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"plate {conf:.2f}", (x1, max(
                        10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            # If inference fails, continue showing raw frames
            print("Inference error:", e)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
