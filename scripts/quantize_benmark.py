from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import time
import os
import psutil
import torch
from model.SkateFormer import Model as SkateFormer


MODEL_WEIGHTS   = r"C:\Users\VietTruongDuc\Documents\datn_dataset\results\finetune\best.pt"
SKATE_INPUT     = "skateformer.onnx"
SKATE_OUTPUT    = "skateformer_int8.onnx"
YOLO_ONNX_PATH  = "yolo11s-pose.onnx"   


print("Quantizing SkateFormer to INT8")
quantize_dynamic(
    SKATE_INPUT,
    SKATE_OUTPUT,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)
print(f"Done! Saved to {SKATE_OUTPUT}\n")

dummy_np_x      = np.random.randn(1, 3, 64, 18, 2).astype(np.float32)
dummy_np_it     = np.zeros((1, 64), dtype=np.int64)
dummy_pt_x      = torch.from_numpy(dummy_np_x)
dummy_pt_it     = torch.from_numpy(dummy_np_it)
dummy_yolo = np.random.randn(1, 3, 480, 480).astype(np.float32)

print("Loading PyTorch .pt model...")
model_pt = SkateFormer(
    num_classes=3, num_people=2, num_points=18, num_frames=64,
    type_1_size=(8,6), type_2_size=(8,3),
    type_3_size=(8,6), type_4_size=(8,3)
)
checkpoint = torch.load(MODEL_WEIGHTS, map_location="cpu")
sd = checkpoint.get("model_state_dict", checkpoint)
clean = {}
for k, v in sd.items():
    n = k.replace("module.", "")
    if n.startswith("proj_head."): n = n.replace("proj_head.", "head.")
    clean[n] = v
model_pt.load_state_dict(clean, strict=False)
model_pt.eval()


sess_fp32 = ort.InferenceSession(SKATE_INPUT,   providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession(SKATE_OUTPUT,  providers=["CPUExecutionProvider"])
sess_yolo = ort.InferenceSession(YOLO_ONNX_PATH, providers=["CPUExecutionProvider"])

def make_feed(sess, dummy_x, dummy_it):
    feed = {}
    for i in sess.get_inputs():
        if "index" in i.name:
            feed[i.name] = dummy_it
        else:
            feed[i.name] = dummy_x
    return feed

fp32_feed = make_feed(sess_fp32, dummy_np_x, dummy_np_it)
int8_feed = make_feed(sess_int8, dummy_np_x, dummy_np_it)
yolo_feed = {sess_yolo.get_inputs()[0].name: dummy_yolo}


def get_size_mb(path):
    return os.path.getsize(path) / 1e6

def get_pt_size_mb(model):
    tmp = "_tmp_model_size.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return size

size_pt      = get_pt_size_mb(model_pt)
size_fp32    = get_size_mb(SKATE_INPUT)
size_int8    = get_size_mb(SKATE_OUTPUT)
size_yolo    = get_size_mb(YOLO_ONNX_PATH)


def measure_ram_mb(fn):
    proc = psutil.Process(os.getpid())
    fn()
    mem_before = proc.memory_info().rss / 1e6
    fn()
    mem_after  = proc.memory_info().rss / 1e6
    return max(0.0, mem_after - mem_before)

ram_pt   = measure_ram_mb(lambda: model_pt(dummy_pt_x, dummy_pt_it))
ram_fp32 = measure_ram_mb(lambda: sess_fp32.run(None, fp32_feed))
ram_int8 = measure_ram_mb(lambda: sess_int8.run(None, int8_feed))
ram_yolo = measure_ram_mb(lambda: sess_yolo.run(None, yolo_feed))
N = 100  

def benchmark_fn(fn, n=N):
    for _ in range(5):   
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std(), 1000 / arr.mean()

print(f"\nBenchmarking ({N} runs each, CPU)...\n")

t_pt,   s_pt,   fps_pt   = benchmark_fn(lambda: model_pt(dummy_pt_x, dummy_pt_it))
t_fp32, s_fp32, fps_fp32 = benchmark_fn(lambda: sess_fp32.run(None, fp32_feed))
t_int8, s_int8, fps_int8 = benchmark_fn(lambda: sess_int8.run(None, int8_feed))
t_yolo, s_yolo, fps_yolo = benchmark_fn(lambda: sess_yolo.run(None, yolo_feed))


col = 24
print(f"\n{'='*90}")
print(f"  SKATEFORMER BENCHMARK")
print(f"{'='*90}")
print(f"{'Model':<{col}} {'Mean (ms)':>10} {'Std':>7} {'FPS':>8} {'Speedup':>9} {'Size (MB)':>11} {'RAM (MB)':>10}")
print("-" * 90)
print(f"{'PyTorch .pt':<{col}} {t_pt:>9.1f}  {s_pt:>6.1f}  {fps_pt:>7.1f}  {'1.00x':>9}  {size_pt:>10.1f}  {ram_pt:>9.1f}")
print(f"{'ONNX FP32':<{col}} {t_fp32:>9.1f}  {s_fp32:>6.1f}  {fps_fp32:>7.1f}  {t_pt/t_fp32:>8.2f}x  {size_fp32:>10.1f}  {ram_fp32:>9.1f}")
print(f"{'ONNX INT8':<{col}} {t_int8:>9.1f}  {s_int8:>6.1f}  {fps_int8:>7.1f}  {t_pt/t_int8:>8.2f}x  {size_int8:>10.1f}  {ram_int8:>9.1f}")
print("-" * 90)
print(f"INT8 vs FP32 speedup : {t_fp32/t_int8:.2f}x  |  size reduction : {size_fp32/size_int8:.2f}x smaller")

print(f"\n{'='*90}")
print(f"  YOLOv11-POSE ONNX BENCHMARK")
print(f"{'='*90}")
print(f"{'Model':<{col}} {'Mean (ms)':>10} {'Std':>7} {'FPS':>8} {'Size (MB)':>11} {'RAM (MB)':>10}")
print("-" * 90)
print(f"{'YOLOv11-Pose ONNX':<{col}} {t_yolo:>9.1f}  {s_yolo:>6.1f}  {fps_yolo:>7.1f}  {size_yolo:>10.1f}  {ram_yolo:>9.1f}")
print("-" * 90)

print(f"\n{'='*90}")
print(f"  TÓM TẮT CHO BẢNG BÁO CÁO")
print(f"{'='*90}")
print(f"  YOLOv11-Pose ONNX latency : {t_yolo:.1f} ms  ({fps_yolo:.1f} FPS)")
print(f"  SkateFormer ONNX FP32     : {t_fp32:.1f} ms  ({fps_fp32:.1f} FPS)")
print(f"  SkateFormer ONNX INT8     : {t_int8:.1f} ms  ({fps_int8:.1f} FPS)")
print(f"{'='*90}\n")