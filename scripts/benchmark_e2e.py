import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ULTRALYTICS_CONFIG_DIR = ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

try:
    import psutil
except ImportError:
    psutil = None

DEPLOY_DIR = ROOT / "deploy"
MODEL_WEIGHTS = DEPLOY_DIR / "skateformer.pt"
SKATE_ONNX = DEPLOY_DIR / "skateformer.onnx"
SKATE_INT8_ONNX = DEPLOY_DIR / "skateformer_int8.onnx"
YOLO_ONNX_PATH = DEPLOY_DIR / "yolo11s-pose.onnx"
DEFAULT_VIDEO_SOURCE = ROOT / "data/test_raw_videos/data_fall_detection/N3_buocgiay.mp4"
OUTPUT_CSV = ROOT / "results/benchmark_e2e.csv"

CLASSES = ["Normal", "Fall", "Fight"]
WINDOW_SIZE = 64


def get_rss_mb():
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def file_size_mb(path):
    return Path(path).stat().st_size / 1e6


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def require_file(path, label):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def state_size_mb(state_dict):
    total = 0
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            total += tensor.numel() * tensor.element_size()
    return total / 1e6


def make_ort_options(num_threads=None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads or max(1, (os.cpu_count() or 2) - 1)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return str(resolve_path(value))


def softmax(logits):
    logits = np.asarray(logits)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def kp_to_sk18(kp, w, h):
    kp = kp.copy()
    kp[:, 0] /= w
    kp[:, 1] /= h
    sk = np.zeros((18, 3), dtype=np.float32)
    sk[0], sk[2], sk[3], sk[4] = kp[0], kp[6], kp[8], kp[10]
    sk[5], sk[6], sk[7] = kp[5], kp[7], kp[9]
    sk[1] = (kp[5] + kp[6]) / 2
    sk[8] = (kp[11] + kp[12]) / 2
    sk[9], sk[10], sk[11] = kp[12], kp[14], kp[16]
    sk[12], sk[13], sk[14] = kp[11], kp[13], kp[15]
    sk[15], sk[16], sk[17] = kp[2], kp[1], kp[4]
    return sk


def frame_to_skeleton(yolo_model, frame, imgsz, conf, use_track=False):
    h, w = frame.shape[:2]
    if use_track:
        result = yolo_model.track(frame, persist=True, imgsz=imgsz, conf=conf, verbose=False)[0]
    else:
        result = yolo_model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    frame_data = np.zeros((18, 3, 2), dtype=np.float32)
    boxes = []

    if result.boxes is None or result.keypoints is None or result.boxes.xyxy is None:
        return frame_data, boxes

    raw_boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()
    if len(raw_boxes) == 0 or len(kpts) == 0:
        return frame_data, boxes

    centers = np.column_stack(
        ((raw_boxes[:, 0] + raw_boxes[:, 2]) / 2, (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2)
    )
    boxes = raw_boxes.astype(int).tolist()

    if len(centers) == 1:
        frame_data[:, :, 0] = kp_to_sk18(kpts[0], w, h)
    else:
        min_dist, pair = float("inf"), (0, 1)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    min_dist, pair = dist, (i, j)
        frame_data[:, :, 0] = kp_to_sk18(kpts[pair[0]], w, h)
        frame_data[:, :, 1] = kp_to_sk18(kpts[pair[1]], w, h)

    return frame_data, boxes


class TorchActionModel:
    name = "PyTorch PT"

    def __init__(self, path, device):
        from model.SkateFormer import Model as SkateFormer

        self.path = Path(path)
        self.device = torch.device(device)
        self.model = SkateFormer(
            num_classes=3,
            num_people=2,
            num_points=18,
            num_frames=64,
            in_channels=3,
            embed_dim=96,
            depths=(2, 2, 2, 2),
            channels=(96, 192, 192, 192),
            kernel_size=7,
            num_heads=32,
            mlp_ratio=4.0,
            attn_drop=0.1,
            drop=0.1,
            drop_path=0.2,
            rel=True,
            index_t=False,
            global_pool="avg",
            type_1_size=(8, 6),
            type_2_size=(8, 3),
            type_3_size=(8, 6),
            type_4_size=(8, 3),
        ).to(self.device)

        checkpoint = torch.load(self.path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.size_mb = state_size_mb(self.model.state_dict())

    @torch.no_grad()
    def __call__(self, x):
        tensor = torch.from_numpy(x).to(self.device)
        output = self.model(tensor, index_t=False)
        if isinstance(output, tuple):
            output = output[0]
        return output.detach().cpu().numpy()


class OnnxActionModel:
    def __init__(self, path, label, providers, num_threads):
        self.name = label
        self.path = Path(path)
        self.size_mb = file_size_mb(self.path)
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=make_ort_options(num_threads),
            providers=providers,
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def __call__(self, x):
        feed = {}
        for name in self.input_names:
            if "index" in name:
                feed[name] = np.zeros((x.shape[0], x.shape[2]), dtype=np.int64)
            else:
                feed[name] = x
        return self.session.run(None, feed)[0]


def draw_overlay(frame, boxes, label, conf):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
    cv2.putText(
        frame,
        f"{label} {conf * 100:.1f}%",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def percentile(values, pct):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def run_benchmark(action_model, yolo_model, args, baseline_rss=0.0, initial_max_rss=0.0):
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    if args.save_video:
        out_path = Path(args.save_video)
        if len(args.models) > 1:
            out_path = out_path.with_name(f"{out_path.stem}_{action_model.name.replace(' ', '_')}{out_path.suffix}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    else:
        writer = None

    skel_buf = deque(maxlen=WINDOW_SIZE)
    last_frame_data = np.zeros((18, 3, 2), dtype=np.float32)
    last_boxes = []
    last_label = "Normal"
    last_conf = 0.0

    frames = 0
    yolo_runs = 0
    action_runs = 0
    yolo_ms = []
    action_ms = []
    max_rss_mb = max(initial_max_rss, get_rss_mb())

    started = time.perf_counter()
    while True:
        if args.max_frames and frames >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        if frames % args.yolo_skip == 0 or frames == 1:
            t0 = time.perf_counter()
            last_frame_data, last_boxes = frame_to_skeleton(
                yolo_model,
                frame,
                imgsz=args.imgsz,
                conf=args.yolo_conf,
                use_track=args.track,
            )
            yolo_ms.append((time.perf_counter() - t0) * 1000)
            yolo_runs += 1
            max_rss_mb = max(max_rss_mb, get_rss_mb())

        skel_buf.append(last_frame_data)
        if len(skel_buf) == WINDOW_SIZE and frames % args.action_skip == 0:
            inp = np.asarray(skel_buf, dtype=np.float32).transpose(2, 0, 1, 3)[np.newaxis]
            t0 = time.perf_counter()
            logits = action_model(inp)
            action_ms.append((time.perf_counter() - t0) * 1000)
            action_runs += 1
            max_rss_mb = max(max_rss_mb, get_rss_mb())
            probs = softmax(logits)[0]
            pred = int(np.argmax(probs))
            last_label = CLASSES[pred]
            last_conf = float(probs[pred])

        if writer is not None:
            writer.write(draw_overlay(frame.copy(), last_boxes, last_label, last_conf))

        if args.show:
            cv2.imshow("benchmark_e2e", draw_overlay(frame.copy(), last_boxes, last_label, last_conf))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - started
    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    yolo_ms_mean = float(np.mean(yolo_ms)) if yolo_ms else 0.0
    action_ms_mean = float(np.mean(action_ms)) if action_ms else 0.0
    system_latency_ms = (elapsed / frames * 1000) if frames > 0 else 0.0

    return {
        "model": action_model.name,
        "action_size_mb": action_model.size_mb,
        "frames": frames,
        "elapsed_sec": elapsed,
        "system_latency_ms": system_latency_ms,
        "end_to_end_fps": frames / elapsed if elapsed > 0 else 0.0,
        "ram_delta_mb": max(0.0, max_rss_mb - baseline_rss) if baseline_rss else 0.0,
        "yolo_runs": yolo_runs,
        "action_runs": action_runs,
        "yolo_ms_mean": yolo_ms_mean,
        "yolo_ms_p95": percentile(yolo_ms, 95),
        "yolo_raw_fps": 1000.0 / yolo_ms_mean if yolo_ms_mean > 0 else 0.0,
        "action_ms_mean": action_ms_mean,
        "action_ms_p95": percentile(action_ms, 95),
        "action_raw_fps": 1000.0 / action_ms_mean if action_ms_mean > 0 else 0.0,
        "action_calls_per_sec": action_runs / elapsed if elapsed > 0 else 0.0,
    }


def build_model(kind, args):
    providers = args.providers.split(",")

    if kind == "pt":
        return TorchActionModel(require_file(args.pt, "PyTorch model"), args.device)
    if kind == "onnx":
        return OnnxActionModel(require_file(args.onnx, "ONNX FP32 model"), "ONNX FP32", providers, args.num_threads)
    if kind == "int8":
        return OnnxActionModel(require_file(args.int8, "ONNX INT8 model"), "ONNX INT8", providers, args.num_threads)
    raise ValueError(f"Unknown model kind: {kind}")


def print_system_summary(results):
    if not results:
        return

    baseline_latency = results[0]["system_latency_ms"]
    print("\nEND-TO-END SYSTEM SUMMARY")
    print("=" * 92)
    print(
        f"{'Pipeline':<14} {'Size (MB)':>10} {'Latency (ms)':>13} "
        f"{'FPS':>8} {'RAM delta (MB)':>15} {'Speedup':>10}"
    )
    print("-" * 92)
    for r in results:
        speedup = baseline_latency / r["system_latency_ms"] if r["system_latency_ms"] > 0 else 0.0
        print(
            f"{r['model']:<14} {r['action_size_mb']:>10.1f} "
            f"{r['system_latency_ms']:>13.1f} {r['end_to_end_fps']:>8.1f} "
            f"{r['ram_delta_mb']:>15.1f} {speedup:>9.2f}x"
        )
    print("=" * 92)
    print("Latency/FPS here are for the whole deployed pipeline, not only SkateFormer.")


def print_results(results):
    print_system_summary(results)
    print("\nEND-TO-END BENCHMARK")
    print("=" * 132)
    print(
        f"{'Model':<14} {'Frames':>8} {'Time(s)':>9} {'E2E FPS':>9} "
        f"{'YOLO n':>7} {'YOLO ms':>9} {'YOLO FPS':>9} {'YOLO p95':>9} "
        f"{'Act n':>7} {'Act ms':>9} {'Act FPS':>9} {'Act p95':>9} {'Act calls/s':>11}"
    )
    print("-" * 132)
    for r in results:
        print(
            f"{r['model']:<14} {r['frames']:>8} {r['elapsed_sec']:>9.2f} "
            f"{r['end_to_end_fps']:>9.2f} {r['yolo_runs']:>7} "
            f"{r['yolo_ms_mean']:>9.2f} {r['yolo_raw_fps']:>9.2f} {r['yolo_ms_p95']:>9.2f} "
            f"{r['action_runs']:>7} {r['action_ms_mean']:>9.2f} "
            f"{r['action_raw_fps']:>9.2f} {r['action_ms_p95']:>9.2f} {r['action_calls_per_sec']:>11.2f}"
        )
    print("=" * 132)
    print("E2E FPS = whole video pipeline. Act FPS = raw action-model throughput. Act calls/s = action calls over full pipeline time.")


def save_results(results, path):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark server-like YOLO + SkateFormer pipeline.")
    parser.add_argument("--source", default=str(DEFAULT_VIDEO_SOURCE), help="Video path or camera id, e.g. 0.")
    parser.add_argument("--models", nargs="+", default=["pt", "onnx", "int8"], choices=["pt", "onnx", "int8"])
    parser.add_argument("--pt", default=str(MODEL_WEIGHTS))
    parser.add_argument("--onnx", default=str(SKATE_ONNX))
    parser.add_argument("--int8", default=str(SKATE_INT8_ONNX))
    parser.add_argument("--yolo", default=str(YOLO_ONNX_PATH))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-skip", type=int, default=5)
    parser.add_argument("--action-skip", type=int, default=5)
    parser.add_argument("--track", action="store_true", help="Use YOLO.track like server.py. Requires lap.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-video", default="")
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    yolo_path = require_file(args.yolo, "YOLO pose model")
    if not str(args.source).isdigit():
        require_file(args.source, "Input video")

    print("MODEL PATHS")
    print("=" * 80)
    print(f"PyTorch PT : {require_file(args.pt, 'PyTorch model')}")
    print(f"ONNX FP32  : {require_file(args.onnx, 'ONNX FP32 model')}")
    print(f"ONNX INT8  : {require_file(args.int8, 'ONNX INT8 model')}")
    print(f"YOLO Pose  : {yolo_path}")
    print(f"Source     : {args.source}")
    print(f"Output     : {resolve_path(args.output)}")
    print("=" * 80)

    results = []
    for kind in args.models:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        baseline_rss = get_rss_mb()
        print(f"\n[INFO] Loading YOLO + {kind} backend...")
        yolo_model = YOLO(str(yolo_path))
        action_model = build_model(kind, args)
        loaded_rss = get_rss_mb()

        print(f"[INFO] Running {action_model.name}...")
        results.append(
            run_benchmark(
                action_model,
                yolo_model,
                args,
                baseline_rss=baseline_rss,
                initial_max_rss=loaded_rss,
            )
        )

        del action_model
        del yolo_model

    print_results(results)
    if args.output:
        save_results(results, args.output)
        print(f"[INFO] Saved results to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
