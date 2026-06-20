import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.SkateFormer import Model as SkateFormer


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize SkateFormer ONNX to INT8 and benchmark.")
    parser.add_argument("--weights", default="results/finetune/best.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--input", default="deploy/skateformer.onnx", help="Input FP32 ONNX path.")
    parser.add_argument("--output", default="deploy/skateformer_int8.onnx", help="Output INT8 ONNX path.")
    parser.add_argument("--yolo", default="deploy/yolo11s-pose.onnx", help="YOLO pose ONNX path.")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs.")
    return parser.parse_args()


def load_torch_model(weights):
    print("Loading PyTorch .pt model...")
    model_pt = SkateFormer(
        num_classes=3,
        num_people=2,
        num_points=18,
        num_frames=64,
        type_1_size=(8, 6),
        type_2_size=(8, 3),
        type_3_size=(8, 6),
        type_4_size=(8, 3),
    )
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    clean = {}
    for key, value in state_dict.items():
        name = key.replace("module.", "")
        if name.startswith("proj_head."):
            name = name.replace("proj_head.", "head.")
        clean[name] = value
    model_pt.load_state_dict(clean, strict=False)
    model_pt.eval()
    return model_pt


def make_feed(sess, dummy_x, dummy_it):
    feed = {}
    for item in sess.get_inputs():
        if "index" in item.name:
            feed[item.name] = dummy_it
        else:
            feed[item.name] = dummy_x
    return feed


def get_size_mb(path):
    return os.path.getsize(path) / 1e6


def get_pt_size_mb(model):
    tmp = ROOT / "_tmp_model_size.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return size


def measure_ram_mb(fn):
    proc = psutil.Process(os.getpid())
    fn()
    mem_before = proc.memory_info().rss / 1e6
    fn()
    mem_after = proc.memory_info().rss / 1e6
    return max(0.0, mem_after - mem_before)


def benchmark_fn(fn, n):
    for _ in range(5):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std(), 1000 / arr.mean()


def main():
    args = parse_args()
    weights = require_file(resolve_path(args.weights), "PyTorch checkpoint")
    skate_input = require_file(resolve_path(args.input), "SkateFormer ONNX FP32")
    skate_output = resolve_path(args.output)
    yolo_onnx = require_file(resolve_path(args.yolo), "YOLO pose ONNX")
    skate_output.parent.mkdir(parents=True, exist_ok=True)

    print("Quantizing SkateFormer to INT8")
    quantize_dynamic(
        str(skate_input),
        str(skate_output),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"Done! Saved to {skate_output}\n")

    dummy_np_x = np.random.randn(1, 3, 64, 18, 2).astype(np.float32)
    dummy_np_it = np.zeros((1, 64), dtype=np.int64)
    dummy_pt_x = torch.from_numpy(dummy_np_x)
    dummy_pt_it = torch.from_numpy(dummy_np_it)
    dummy_yolo = np.random.randn(1, 3, 480, 480).astype(np.float32)

    model_pt = load_torch_model(weights)

    sess_fp32 = ort.InferenceSession(str(skate_input), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(skate_output), providers=["CPUExecutionProvider"])
    sess_yolo = ort.InferenceSession(str(yolo_onnx), providers=["CPUExecutionProvider"])

    fp32_feed = make_feed(sess_fp32, dummy_np_x, dummy_np_it)
    int8_feed = make_feed(sess_int8, dummy_np_x, dummy_np_it)
    yolo_feed = {sess_yolo.get_inputs()[0].name: dummy_yolo}

    size_pt = get_pt_size_mb(model_pt)
    size_fp32 = get_size_mb(skate_input)
    size_int8 = get_size_mb(skate_output)
    size_yolo = get_size_mb(yolo_onnx)

    ram_pt = measure_ram_mb(lambda: model_pt(dummy_pt_x, dummy_pt_it))
    ram_fp32 = measure_ram_mb(lambda: sess_fp32.run(None, fp32_feed))
    ram_int8 = measure_ram_mb(lambda: sess_int8.run(None, int8_feed))
    ram_yolo = measure_ram_mb(lambda: sess_yolo.run(None, yolo_feed))

    print(f"\nBenchmarking ({args.runs} runs each, CPU)...\n")

    t_pt, s_pt, fps_pt = benchmark_fn(lambda: model_pt(dummy_pt_x, dummy_pt_it), args.runs)
    t_fp32, s_fp32, fps_fp32 = benchmark_fn(lambda: sess_fp32.run(None, fp32_feed), args.runs)
    t_int8, s_int8, fps_int8 = benchmark_fn(lambda: sess_int8.run(None, int8_feed), args.runs)
    t_yolo, s_yolo, fps_yolo = benchmark_fn(lambda: sess_yolo.run(None, yolo_feed), args.runs)

    col = 24
    print(f"\n{'=' * 90}")
    print("  SKATEFORMER BENCHMARK")
    print(f"{'=' * 90}")
    print(f"{'Model':<{col}} {'Mean (ms)':>10} {'Std':>7} {'FPS':>8} {'Speedup':>9} {'Size (MB)':>11} {'RAM (MB)':>10}")
    print("-" * 90)
    print(f"{'PyTorch .pt':<{col}} {t_pt:>9.1f}  {s_pt:>6.1f}  {fps_pt:>7.1f}  {'1.00x':>9}  {size_pt:>10.1f}  {ram_pt:>9.1f}")
    print(f"{'ONNX FP32':<{col}} {t_fp32:>9.1f}  {s_fp32:>6.1f}  {fps_fp32:>7.1f}  {t_pt/t_fp32:>8.2f}x  {size_fp32:>10.1f}  {ram_fp32:>9.1f}")
    print(f"{'ONNX INT8':<{col}} {t_int8:>9.1f}  {s_int8:>6.1f}  {fps_int8:>7.1f}  {t_pt/t_int8:>8.2f}x  {size_int8:>10.1f}  {ram_int8:>9.1f}")
    print("-" * 90)
    print(f"INT8 vs FP32 speedup : {t_fp32/t_int8:.2f}x  |  size reduction : {size_fp32/size_int8:.2f}x smaller")

    print(f"\n{'=' * 90}")
    print("  YOLOv11-POSE ONNX BENCHMARK")
    print(f"{'=' * 90}")
    print(f"{'Model':<{col}} {'Mean (ms)':>10} {'Std':>7} {'FPS':>8} {'Size (MB)':>11} {'RAM (MB)':>10}")
    print("-" * 90)
    print(f"{'YOLOv11-Pose ONNX':<{col}} {t_yolo:>9.1f}  {s_yolo:>6.1f}  {fps_yolo:>7.1f}  {size_yolo:>10.1f}  {ram_yolo:>9.1f}")
    print("-" * 90)

    print(f"\n{'=' * 90}")
    print("  TOM TAT CHO BANG BAO CAO")
    print(f"{'=' * 90}")
    print(f"  YOLOv11-Pose ONNX latency : {t_yolo:.1f} ms  ({fps_yolo:.1f} FPS)")
    print(f"  SkateFormer ONNX FP32     : {t_fp32:.1f} ms  ({fps_fp32:.1f} FPS)")
    print(f"  SkateFormer ONNX INT8     : {t_int8:.1f} ms  ({fps_int8:.1f} FPS)")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
