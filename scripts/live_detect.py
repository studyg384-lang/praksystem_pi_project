import asyncio
import base64
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from libcamera import Transform
from picamera2 import Picamera2
from pydantic import BaseModel
from ultralytics import YOLO

# Optimize CPU threads for Raspberry Pi 4
cv2.setNumThreads(2)
os.environ["OMP_NUM_THREADS"] = "4"

BASE_DIR = "/home/praksystem/project_prak"
DEVICE_ID = os.getenv("DEVICE_ID", "pi-cam-01")
API_SECRET_TOKEN = os.getenv("EDGE_API_TOKEN", "prak-secret-key-2026")  # Change this!

# Placeholder GPS coordinates (hook up a NEO-6M GPS module here later)
CURRENT_LOCATION = {"lat": 28.6139, "lng": 77.2090, "address": "Checkpoint Alpha"}

# ================= DATA PACKAGE SCHEMA =================
class ViolationPackage(BaseModel):
    package_type: str = "violation_event"
    event_id: str
    device_id: str
    timestamp_iso: str
    track_id: int
    violation_type: str
    confidence: float
    plate_text: str = "unknown"
    location: dict
    evidence_frame_b64: str      # Full scene photo (Base64 JPEG)
    evidence_crop_b64: str       # Cropped violator/plate photo (Base64 JPEG)
# =======================================================

CANDIDATE_MODELS = [
    os.path.join(BASE_DIR, "helmetv2_ncnn_model"),
    os.path.join(BASE_DIR, "helmetv2.onnx"),
    os.path.join(BASE_DIR, "helmetv2.pt"),
]
MODEL_PATH = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), CANDIDATE_MODELS[0])
model = YOLO(MODEL_PATH, task="detect")

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=Transform(hflip=0, vflip=1),
    buffer_count=2
))
cam.start()

lock = threading.Lock()
new_frame_event = threading.Event()
latest_raw_frame = None
latest_jpg_bytes = None
latest_boxes = []
det_fps = 0.0
stream_fps = 0.0

unique_violator_ids = set()
recent_packages: List[dict] = []  # Stores last 25 violation packages in RAM
ws_clients: List[WebSocket] = []
main_loop: Optional[asyncio.AbstractEventLoop] = None


def encode_b64_jpeg(img, quality=75) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


async def broadcast_package(package_dict: dict):
    """Push package instantly to all connected frontend dashboards."""
    disconnected = []
    for ws in list(ws_clients):
        try:
            await ws.send_json(package_dict)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in ws_clients:
            ws_clients.remove(ws)


def detect_loop():
    global latest_boxes, det_fps
    while True:
        new_frame_event.wait(timeout=0.1)
        new_frame_event.clear()

        with lock:
            if latest_raw_frame is None:
                continue
            frame = latest_raw_frame.copy()

        t0 = time.perf_counter()
        results = model.track(
            frame, imgsz=320, conf=0.4, iou=0.5,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )

        boxes = []
        if results and len(results[0].boxes) > 0:
            h, w = frame.shape[:2]
            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                conf_val = float(b.conf[0])
                class_name = model.names[cls_id]
                track_id = int(b.id[0]) if b.id is not None else None

                boxes.append((x1, y1, x2, y2, class_name, conf_val, track_id))

                # Trigger instant package when a NEW violator ID is confirmed
                if track_id is not None and "without" in class_name.lower():
                    if track_id not in unique_violator_ids:
                        unique_violator_ids.add(track_id)

                        # Safely crop violator region
                        cx1, cy1 = max(0, x1), max(0, y1)
                        cx2, cy2 = min(w, x2), min(h, y2)
                        crop = frame[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else frame

                        now_iso = datetime.now(timezone.utc).isoformat()
                        pkg = ViolationPackage(
                            event_id=f"{DEVICE_ID}-{int(time.time()*1000)}-{track_id}",
                            device_id=DEVICE_ID,
                            timestamp_iso=now_iso,
                            track_id=track_id,
                            violation_type=class_name,
                            confidence=round(conf_val, 3),
                            plate_text="unknown",
                            location=CURRENT_LOCATION,
                            evidence_frame_b64=encode_b64_jpeg(frame, quality=70),
                            evidence_crop_b64=encode_b64_jpeg(crop, quality=80),
                        ).model_dump()

                        with lock:
                            recent_packages.append(pkg)
                            if len(recent_packages) > 25:
                                recent_packages.pop(0)

                        if main_loop is not None:
                            asyncio.run_coroutine_threadsafe(broadcast_package(pkg), main_loop)

        dt = time.perf_counter() - t0
        with lock:
            latest_boxes = boxes
            det_fps = 0.8 * det_fps + 0.2 * (1.0 / max(dt, 1e-4)) if det_fps > 0 else (1.0 / max(dt, 1e-4))


def camera_loop():
    global latest_raw_frame, latest_jpg_bytes, stream_fps
    prev_t = time.perf_counter()

    while True:
        frame = cam.capture_array()
        with lock:
            latest_raw_frame = frame
            boxes = list(latest_boxes)
            d_fps = det_fps
            total_v = len(unique_violator_ids)
        new_frame_event.set()

        vis = frame.copy()
        for x1, y1, x2, y2, class_name, conf, track_id in boxes:
            color = (0, 0, 255) if "without" in class_name.lower() else (0, 255, 0)
            label = f"ID:{track_id} {class_name} {conf:.2f}" if track_id else f"{class_name} {conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.perf_counter()
        stream_fps = 0.9 * stream_fps + 0.1 * (1.0 / max(now - prev_t, 1e-4))
        prev_t = now

        cv2.putText(vis, f"AI: {d_fps:.1f} FPS | Violations: {total_v}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            with lock:
                latest_jpg_bytes = jpg.tobytes()


app = FastAPI(title="Road Eye Secure Edge Node")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()


def verify_token(token: str):
    if token != API_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized token")


@app.get("/video_feed")
def video_feed(token: str = Query(...)):
    """Low-latency authenticated live stream."""
    verify_token(token)
    def gen():
        while True:
            with lock:
                frame_bytes = latest_jpg_bytes
            if frame_bytes:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/packages/history")
def get_recent_packages(x_api_key: str = Header(...)):
    """Fetch recent violation packages on initial page load."""
    verify_token(x_api_key)
    with lock:
        return {"device_id": DEVICE_ID, "packages": list(recent_packages)}


@app.websocket("/ws/packages")
async def websocket_packages(websocket: WebSocket, token: str = Query(...)):
    """Real-time encrypted push channel for violation packages + telemetry."""
    if token != API_SECRET_TOKEN:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            # Send lightweight heartbeat + FPS telemetry every 2 seconds
            with lock:
                telemetry = {
                    "package_type": "telemetry",
                    "device_id": DEVICE_ID,
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                    "detect_fps": round(det_fps, 1),
                    "unique_violations": len(unique_violator_ids),
                    "location": CURRENT_LOCATION,
                }
            await websocket.send_json(telemetry)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")