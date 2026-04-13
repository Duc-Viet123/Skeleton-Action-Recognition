import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import asyncio
import threading
import queue
import time
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import onnxruntime as ort
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
_HERE              = os.path.dirname(os.path.abspath(__file__))
ACTION_MODEL_PATH  = os.path.join(_HERE, "skateformer_int8.onnx")
YOLO_WEIGHTS       = os.path.join(_HERE, "yolo11s-pose.onnx")
SAVE_DIR           = os.path.join(_HERE, "alerts")
REPORT_DIR         = os.path.join(_HERE, "reports")
UPLOAD_DIR         = os.path.join(_HERE, "uploads")

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASSES            = ["Normal", "Fall", "Fight"]
WINDOW_SIZE        = 64
BOX_ALPHA          = 0.55
YOLO_SKIP          = 5
ACTION_SKIP        = 5
SAVE_COOLDOWN_SECS = 10
FALL_THRESHOLD     = 0.60
FIGHT_THRESHOLD    = 0.60
QUEUE_FRAME_MAX    = 2
QUEUE_SKELETON_MAX = 4
QUEUE_RENDER_MAX   = 4     


def _make_sess_opts():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = max(1, os.cpu_count() - 1)
    opts.inter_op_num_threads     = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts

def kp_to_sk18(kp, w, h):
    kp = kp.copy()
    kp[:, 0] /= w; kp[:, 1] /= h
    sk = np.zeros((18, 3))
    sk[0],sk[2],sk[3],sk[4] = kp[0],kp[6],kp[8],kp[10]
    sk[5],sk[6],sk[7]       = kp[5],kp[7],kp[9]
    sk[1]  = (kp[5]+kp[6])/2
    sk[8]  = (kp[11]+kp[12])/2
    sk[9],sk[10],sk[11]  = kp[12],kp[14],kp[16]
    sk[12],sk[13],sk[14] = kp[11],kp[13],kp[15]
    sk[15],sk[16],sk[17] = kp[2],kp[1],kp[4]
    return sk

#  THREADS

class CaptureThread(threading.Thread):
    def __init__(self, cap, frame_q, stop_event, fps_state):
        super().__init__(daemon=True)
        self.cap        = cap
        self.frame_q    = frame_q
        self.stop_event = stop_event
        self.fps_state  = fps_state
        self.fps_count  = 0
        self.fps_start  = time.time()

        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or src_fps > 120:
            src_fps = 30
        self.frame_delay = 1.0 / src_fps
        self.fps_state["target_fps"] = src_fps

    def run(self):
        next_frame_time = time.perf_counter()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break

            self.fps_count += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.fps_state["capture_fps"] = self.fps_count / elapsed
                self.fps_count = 0
                self.fps_start = time.time()

            try:
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass

            next_frame_time += self.frame_delay
            sleep_t = next_frame_time - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_frame_time = time.perf_counter()


class YoloThread(threading.Thread):
    def __init__(self, yolo_model, frame_q, skel_q, stop_event):
        super().__init__(daemon=True)
        self.model      = yolo_model
        self.frame_q    = frame_q
        self.skel_q     = skel_q
        self.stop_event = stop_event
        self.frame_idx  = 0
        self.last_res   = None

    def run(self):
        MAX_PEOPLE = 2
        while not self.stop_event.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self.frame_idx += 1
            h, w = frame.shape[:2]

            if self.frame_idx % YOLO_SKIP == 0 or self.last_res is None:
                self.last_res = self.model.track(
                    frame, persist=True, imgsz=480, conf=0.5, verbose=False
                )

            frame_data  = np.zeros((18, 3, MAX_PEOPLE))
            drawn_boxes = []
            drawn_kpts  = []
            r = self.last_res[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                kpts      = r.keypoints.data.cpu().numpy()
                centers   = np.column_stack((
                    (boxes[:,0]+boxes[:,2])/2,
                    (boxes[:,1]+boxes[:,3])/2
                ))
                drawn_boxes = list(zip(boxes.astype(int), track_ids))
                drawn_kpts  = list(zip(kpts, track_ids))
                if len(centers) == 1:
                    frame_data[:,:,0] = kp_to_sk18(kpts[0], w, h)
                elif len(centers) >= 2:
                    min_dist, pair = 1e9, (0,1)
                    for i in range(len(centers)):
                        for j in range(i+1, len(centers)):
                            d = np.linalg.norm(centers[i]-centers[j])
                            if d < min_dist:
                                min_dist = d; pair = (i,j)
                    i1, i2 = pair
                    frame_data[:,:,0] = kp_to_sk18(kpts[i1], w, h)
                    frame_data[:,:,1] = kp_to_sk18(kpts[i2], w, h)

            try:
                self.skel_q.put_nowait({
                    "frame": frame, "frame_data": frame_data,
                    "drawn_boxes": drawn_boxes, "frame_idx": self.frame_idx,
                    "drawn_kpts": drawn_kpts,
                })
            except queue.Full:
                pass


class ActionThread(threading.Thread):
    def __init__(self, action_sess, skel_q, render_q, result_state, stop_event, alert_manager):
        super().__init__(daemon=True)
        self.sess         = action_sess
        self.skel_q       = skel_q
        self.render_q     = render_q
        self.result_state = result_state
        self.stop_event   = stop_event
        self.alert_manager = alert_manager
        self.skel_buf     = deque(maxlen=WINDOW_SIZE)
        self.smooth_boxes = {}
        self.last_saved_time = {"Fight": 0.0, "Fall": 0.0}

    def _ema_box(self, tid, raw):
        if tid not in self.smooth_boxes:
            self.smooth_boxes[tid] = raw.astype(float)
        else:
            self.smooth_boxes[tid] = BOX_ALPHA*raw + (1-BOX_ALPHA)*self.smooth_boxes[tid]
        return self.smooth_boxes[tid].astype(int)

    def _tid_color(self, tid: int):
        # Stable vivid color per track id (BGR)
        hue = (tid * 37) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _kpt17_to_sk18_px(self, kpt17):
        """
        Convert Ultralytics COCO-17 keypoints -> project `sk18` (18,3) in pixel coords.

        Project convention is defined by `kp_to_sk18()`:
        - add neck (mid of shoulders) and mid-hip (mid of hips)
        - do not include COCO left_ear (index 3)
        """
        if kpt17 is None:
            return None
        kpt17 = np.asarray(kpt17, dtype=float)
        if kpt17.ndim != 2 or kpt17.shape[0] < 17 or kpt17.shape[1] < 2:
            return None

        sk = np.zeros((18, 3), dtype=float)

        # COCO-17 indices (Ultralytics): 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
        # 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb, 9 l_wri, 10 r_wri,
        # 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank
        def _get(i):
            x = float(kpt17[i, 0]); y = float(kpt17[i, 1])
            c = float(kpt17[i, 2]) if kpt17.shape[1] >= 3 else 1.0
            return np.array([x, y, c], dtype=float)

        # same mapping as kp_to_sk18(), but keep pixel coords
        sk[0] = _get(0)   # nose
        sk[2] = _get(6)   # r_shoulder
        sk[3] = _get(8)   # r_elbow
        sk[4] = _get(10)  # r_wrist
        sk[5] = _get(5)   # l_shoulder
        sk[6] = _get(7)   # l_elbow
        sk[7] = _get(9)   # l_wrist
        sk[9] = _get(12)  # r_hip
        sk[10] = _get(14) # r_knee
        sk[11] = _get(16) # r_ankle
        sk[12] = _get(11) # l_hip
        sk[13] = _get(13) # l_knee
        sk[14] = _get(15) # l_ankle
        sk[15] = _get(2)  # r_eye
        sk[16] = _get(1)  # l_eye
        sk[17] = _get(4)  # r_ear

        # synthetic joints: neck (mid shoulders), mid-hip (mid hips)
        ls, rs = _get(5), _get(6)
        lh, rh = _get(11), _get(12)
        sk[1] = (ls + rs) / 2.0
        sk[8] = (lh + rh) / 2.0
        return sk

    def _draw_sk18_skeleton(self, frame, kpt17, color):
        sk18 = self._kpt17_to_sk18_px(kpt17)
        if sk18 is None:
            return

        xy = sk18[:, :2].astype(int)
        conf = sk18[:, 2].astype(float)
        thr = 0.30

        # Must match training-time bone pairs (Feeders/feeder_finetune.py)
        bone_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
            (0, 15), (0, 16), (15, 17),
        ]

        for a, b in bone_pairs:
            if conf[a] >= thr and conf[b] >= thr:
                cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)

        for i in range(18):
            if conf[i] >= thr:
                cv2.circle(frame, tuple(xy[i]), 3, color, -1, lineType=cv2.LINE_AA)

    def run(self):
        while not self.stop_event.is_set():
            try:
                payload = self.skel_q.get(timeout=0.5)
            except queue.Empty:
                continue

            frame, frame_data = payload["frame"], payload["frame_data"]
            drawn_boxes, frame_idx = payload["drawn_boxes"], payload["frame_idx"]
            drawn_kpts = payload.get("drawn_kpts", [])

            smooth     = []
            active_ids = set()
            for box, tid in drawn_boxes:
                smooth.append((self._ema_box(tid, box), int(tid)))
                active_ids.add(int(tid))
            for old in list(self.smooth_boxes):
                if old not in active_ids:
                    del self.smooth_boxes[old]

            self.skel_buf.append(frame_data)

            if len(self.skel_buf) == WINDOW_SIZE and frame_idx % ACTION_SKIP == 0:
                inp    = np.array(self.skel_buf).transpose(2,0,1,3)[np.newaxis].astype(np.float32)
                logits = self.sess.run(None, {"input": inp})[0]
                e      = np.exp(logits - logits.max())
                probs  = (e / e.sum())[0]
                pred   = int(np.argmax(probs))

                label = CLASSES[pred]
                conf  = float(probs[pred])

                if label == "Fall"  and conf < FALL_THRESHOLD:  label, conf = "Normal", float(probs[0])
                if label == "Fight" and conf < FIGHT_THRESHOLD: label, conf = "Normal", float(probs[0])

                self.result_state.update({
                    "label": label,
                    "conf":  conf,
                    "probs": probs.tolist(),
                })

        
            label = self.result_state.get("label", "—")
            conf  = self.result_state.get("conf",  0.0)
            self._draw_frame(frame, smooth, drawn_kpts, label, conf)

            if label in ("Fight", "Fall"):
                now = time.time()
                if now - self.last_saved_time.get(label, 0) >= SAVE_COOLDOWN_SECS:
                    self.alert_manager.save_alert(frame.copy(), label)
                    self.last_saved_time[label] = now

            try:
                self.render_q.put_nowait({
                    "frame": frame,
                    "label": label,
                    "conf":  conf,
                })
            except queue.Full:
                pass

    def _draw_frame(self, frame, drawn_boxes, drawn_kpts, label, conf):
        BOX_COLOR = (0, 200, 100)
        for sbox, tid in drawn_boxes:
            x1, y1, x2, y2 = sbox
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), BOX_COLOR, 1)
            cv2.rectangle(frame, (x1, y1),     (x2, y2),     BOX_COLOR, 2)
            cv2.rectangle(frame, (x1, y1-22),  (x1+54, y1),  BOX_COLOR, -1)
            cv2.putText(frame, f"ID:{tid}", (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw pose skeletons for detected people
        for kpt, tid in drawn_kpts:
            self._draw_sk18_skeleton(frame, kpt, self._tid_color(int(tid)))

        if label in ("Fall", "Fight"):
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, h-46), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
            text = f"{label.upper()}  {conf*100:.1f}%"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            cv2.putText(frame, text, ((w-tw)//2, h-14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)


class AlertManager:
    def __init__(self):
        self.alerts = deque(maxlen=200)
        self.count  = {"Fight": 0, "Fall": 0}

    def save_alert(self, frame, label):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fname = f"{label}_{ts}.jpg"
        fpath = os.path.join(SAVE_DIR, fname)
        cv2.imwrite(fpath, frame)
        self.count[label] += 1
        self.alerts.appendleft({
            "label":     label,
            "file":      fname,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        print(f"[Alert] {label} saved → {fname}")

    def get_list(self):
        return list(self.alerts)

    def get_count(self):
        return dict(self.count)

class PipelineState:
    def __init__(self):
        self.cap          = None
        self.threads      = []
        self.stop_event   = threading.Event()
        self.frame_q      = None
        self.skel_q       = None
        self.render_q     = None
        self.result_state = {"label": "—", "conf": 0.0, "probs": [0, 0, 0]}
        self.fps_state    = {"capture_fps": 0.0, "target_fps": 30.0}
        self.session_start = None
        self.running      = False
        self.yolo_model   = None
        self.action_sess  = None
        self.models_ready = False

        self.alert_manager = AlertManager()

    def load_models(self):
        print("[Server] Loading YOLO...")
        self.yolo_model = YOLO(YOLO_WEIGHTS)
        print("[Server] Loading SkateFormer ONNX...")
        self.action_sess = ort.InferenceSession(
            ACTION_MODEL_PATH,
            sess_options=_make_sess_opts(),
            providers=["CPUExecutionProvider"],
        )
        self.models_ready = True
        print("[Server] Models ready.")

    def start(self, src):
        if not self.models_ready:
            raise RuntimeError("Models not loaded yet")
        self.stop()

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        self.stop_event   = threading.Event()
        self.frame_q      = queue.Queue(maxsize=QUEUE_FRAME_MAX)
        self.skel_q       = queue.Queue(maxsize=QUEUE_SKELETON_MAX)
        self.render_q     = queue.Queue(maxsize=QUEUE_RENDER_MAX)
        self.result_state = {"label": "—", "conf": 0.0, "probs": [0, 0, 0]}
        self.fps_state    = {"capture_fps": 0.0, "target_fps": 30.0}
        self.session_start = time.time()

        t_cap    = CaptureThread(self.cap, self.frame_q, self.stop_event, self.fps_state)
        t_yolo   = YoloThread(self.yolo_model, self.frame_q, self.skel_q, self.stop_event)
        t_action = ActionThread(
            self.action_sess, self.skel_q, self.render_q,
            self.result_state, self.stop_event, self.alert_manager
        )
        self.threads = [t_cap, t_yolo, t_action]
        for t in self.threads:
            t.start()

        self.running = True
        print(f"[Server] Pipeline started → {src}")

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)
        self.threads = []
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        print("[Server] Pipeline stopped.")

#  FASTAPI APP

app = FastAPI(title="AI Surveillance API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = PipelineState()

static_dir = os.path.join(_HERE, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static",  StaticFiles(directory=static_dir),  name="static")
app.mount("/alerts",  StaticFiles(directory=SAVE_DIR),    name="alerts")
app.mount("/reports", StaticFiles(directory=REPORT_DIR),  name="reports")


@app.on_event("startup")
def startup_event():
    """Load models khi server khởi động (chạy trong thread riêng để không block)"""
    t = threading.Thread(target=pipeline.load_models, daemon=True)
    t.start()


@app.on_event("shutdown")
def shutdown_event():
    pipeline.stop()



@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── STATUS
@app.get("/api/status")
def get_status():
    uptime = round(time.time() - pipeline.session_start, 1) if pipeline.session_start else 0
    return {
        "running":      pipeline.running,
        "models_ready": pipeline.models_ready,
        "detection":    pipeline.result_state,
        "fps":          pipeline.fps_state,
        "alerts":       pipeline.alert_manager.get_count(),
        "uptime_sec":   uptime,
    }


# ── UPLOAD VIDEO
@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not pipeline.models_ready:
        raise HTTPException(503, "Models are still loading, please wait...")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg"}:
        raise HTTPException(400, f"Unsupported format: {ext}")

    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline.start(dest)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return {"ok": True, "source": file.filename}


# ── START WEBCAM
@app.post("/api/start-webcam")
def start_webcam(camera_id: int = 0):
    if not pipeline.models_ready:
        raise HTTPException(503, "Models are still loading, please wait...")
    try:
        pipeline.start(camera_id)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "source": f"webcam:{camera_id}"}


# ── STOP PIPELINE
@app.post("/api/stop")
def stop_pipeline():
    pipeline.stop()
    return {"ok": True}


# ── ALERTS LIST
@app.get("/api/alerts")
def get_alerts():
    return {
        "count": pipeline.alert_manager.get_count(),
        "items": pipeline.alert_manager.get_list(),
    }


# ── WEBSOCKET
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            if not pipeline.running:
                await ws.send_json({"type": "idle"})
                await asyncio.sleep(0.5)
                continue

            try:
                payload = await loop.run_in_executor(
                    None, lambda: pipeline.render_q.get(timeout=0.1)
                )
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            frame = payload["frame"]
            h, w  = frame.shape[:2]
            scale = min(1.0, 640 / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            await ws.send_bytes(buf.tobytes())

            await ws.send_json({
                "type":  "meta",
                "label": payload["label"],
                "conf":  round(payload["conf"], 4),
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=8000, reload=False)