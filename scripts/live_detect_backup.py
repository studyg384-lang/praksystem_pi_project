import threading
import time
import cv2
from flask import Flask, Response
from libcamera import Transform
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "helmetv1.onnx"
IMGSZ = 320        # must match the export size
CONF = 0.4         # lower = more detections, raise to cut false alarms

model = YOLO(MODEL_PATH, task="detect")

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=1)   # 180 degree rotation
))
cam.start()

lock = threading.Lock()
latest_frame = None
latest_boxes = []
det_fps = 0.0

def detect_loop():
    global latest_boxes, det_fps
    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        t = time.time()
        results = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
        boxes = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2, y2, model.names[int(b.cls)], float(b.conf)))
        with lock:
            latest_boxes = boxes
            det_fps = 1 / (time.time() - t)

threading.Thread(target=detect_loop, daemon=True).start()

app = Flask(__name__)

def generate():
    global latest_frame
    while True:
        frame = cam.capture_array()
        with lock:
            latest_frame = frame.copy()   # clean copy for the detector
            boxes = list(latest_boxes)
            fps = det_fps

        for x1, y1, x2, y2, label, conf in boxes:
            color = (0, 0, 255) if "without" in label else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Detect FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.03)

@app.route("/")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000, threaded=True)