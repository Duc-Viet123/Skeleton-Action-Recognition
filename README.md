<div align="center">

# Skeleton-Based Action Recognition

Phát hiện hành động bất thường từ dữ liệu skeleton người với SkateFormer và YOLO Pose.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray?logo=onnx)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deploy-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/tdv511/Data_Skeleton_Action_Recognition)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Giới Thiệu

Dự án xây dựng hệ thống nhận dạng hành động bất thường trong video, tập trung vào 3 lớp: `Normal`, `Fall`, `Fight`. Pipeline triển khai theo hướng tách nhận diện pose và nhận dạng hành động: YOLO Pose trích xuất keypoints, sau đó mô hình skeleton-based action recognition phân loại hành động theo cửa sổ thời gian.

```text
Video/Webcam -> YOLO Pose -> Skeleton COCO-18 -> SkateFormer -> Normal / Fall / Fight
```

Các phần chính của dự án:

- Tiền xử lý video/skeleton thô thành tensor `.npy` dạng `(N, C, T, V, M)`.
- Pre-training tự giám sát với NT-Xent contrastive loss.
- Fine-tuning supervised cho 3 lớp hành động.
- Baseline train-from-scratch và ST-GCN để so sánh.
- Export ONNX, quantize INT8 và benchmark end-to-end.
- FastAPI server kèm Web UI để chạy webcam hoặc upload video.

## Kiến Trúc Mô Hình

Mô hình chính là `SkateFormer`, một kiến trúc Transformer cho skeleton action recognition. Cấu hình trong repo đang dùng:

| Tham số | Giá trị |
| --- | --- |
| Input channels | 3 (`x`, `y`, `confidence`) |
| Số frame mỗi window | 64 |
| Số khớp | 18 |
| Số người tối đa | 2 |
| Embed dim | 96 |
| Depths | `[2, 2, 2, 2]` |
| Channels | `[96, 192, 192, 192]` |
| Attention heads | 32 |

Repo cũng có baseline `ST-GCN` trong `model/st_gcn.py` với config `config/gcn.yaml`.

## Cấu Trúc Dự Án

```text
Skeleton-Action-Recognition/
├── config/
│   ├── pretrain.yaml             # Config self-supervised pre-training
│   ├── finetune.yaml             # Config fine-tuning SkateFormer
│   ├── baseline.yaml             # Config train SkateFormer from scratch
│   └── gcn.yaml                  # Config ST-GCN baseline
├── Feeders/
│   ├── feeder_pretrain.py        # Dataset/augmentation cho pre-training
│   └── feeder_finetune.py        # Dataset/augmentation cho fine-tuning
├── model/
│   ├── SkateFormer.py            # SkateFormer classification model
│   ├── SkateFormerPre.py         # SkateFormer projection head cho pre-training
│   ├── st_gcn.py                 # ST-GCN baseline
│   └── utils/                    # Graph/T-GCN utilities
├── scripts/
│   ├── download_data.py          # Tải dataset từ Hugging Face
│   ├── preprocess_pretrain.py    # Raw JSON skeleton -> pretrain .npy
│   ├── preprocess_finetune.py    # Raw video -> finetune .npy bằng YOLO Pose
│   ├── train_pretrain.py         # Self-supervised pre-training
│   ├── train_finetune.py         # Fine-tuning và baseline SkateFormer
│   ├── train_gcn.py              # ST-GCN baseline
│   ├── Analyze_pretrain.py       # Phân tích tập pretrain
│   ├── Analyze_finetune.py       # Phân tích tập finetune
│   ├── convert_onnxruntime.py    # Export PyTorch checkpoint -> ONNX
│   ├── quantize_benmark.py       # Quantize INT8 và benchmark model
│   └── benchmark_e2e.py          # Benchmark pipeline YOLO + action model
├── deploy/
│   ├── server.py                 # FastAPI server
│   ├── static/index.html         # Web UI
│   ├── alerts/.gitkeep           # Snapshot cảnh báo runtime
│   ├── uploads/.gitkeep          # Video upload runtime
│   └── requirements.txt          # Dependencies cho deploy
├── results/                      # Checkpoint, log, figure sinh ra khi train
├── requirements.txt              # Dependencies cho train/export/benchmark
├── LICENSE
└── README.md
```

## Dataset

Dataset được lưu trên Hugging Face Hub: [tdv511/Data_Skeleton_Action_Recognition](https://huggingface.co/datasets/tdv511/Data_Skeleton_Action_Recognition).

Các archive chính:

| File | Kích thước tham khảo | Nội dung |
| --- | ---: | --- |
| `pretrain.zip` | ~3.77 GB | Skeleton `.npy` cho pre-training |
| `finetune.zip` | ~53.6 MB | Skeleton `.npy` và label cho fine-tuning |
| `raw.zip` | ~14.7 GB | Dữ liệu thô để tái tạo dataset |
| `test_raw_videos.zip` | ~613 MB | Video kiểm thử/demo |

Tải toàn bộ dữ liệu:

```bash
python scripts/download_data.py
```

Chỉ tải dữ liệu đã xử lý để train nhanh:

```bash
python scripts/download_data.py --files pretrain.zip finetune.zip
```

Sau khi giải nén, cấu trúc dữ liệu cần khớp với các config:

```text
data/
├── pretrain/
│   ├── train/train_data.npy
│   └── val/val_data.npy
├── finetune/
│   ├── train/  # train_data.npy, train_label.npy
│   ├── val/    # val_data.npy, val_label.npy
│   └── test/   # test_data.npy, test_label.npy
├── raw/
│   ├── pretrain/
│   └── finetune/
└── test_raw_videos/
```

## Cài Đặt

Tạo môi trường Python và cài dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu chỉ chạy server deploy:

```bash
pip install -r deploy/requirements.txt
```

Với GPU, cài bản PyTorch/CUDA phù hợp theo hướng dẫn chính thức của PyTorch trước khi train.

## Tiền Xử Lý Dữ Liệu

Nếu đã tải `pretrain.zip` và `finetune.zip`, có thể bỏ qua bước này. Nếu muốn tái tạo `.npy` từ dữ liệu thô:

```bash
python scripts/preprocess_pretrain.py --raw_dir data/raw/pretrain --out_dir data/pretrain
```

```bash
python scripts/preprocess_finetune.py --raw_dir data/raw/finetune --out_dir data/finetune --model yolov8n-pose.pt
```

`preprocess_finetune.py` kỳ vọng thư mục raw có các class folder `Normal/`, `Fall/`, `Fight/`.

## Huấn Luyện

Pre-training tự giám sát:

```bash
python scripts/train_pretrain.py --config config/pretrain.yaml
```

Fine-tuning từ checkpoint pre-trained:

```bash
python scripts/train_finetune.py --config config/finetune.yaml
```

Baseline SkateFormer train from scratch:

```bash
python scripts/train_finetune.py --config config/baseline.yaml
```

Baseline ST-GCN:

```bash
python scripts/train_gcn.py --config config/gcn.yaml
```

Checkpoint và log được lưu theo `work_dir` trong từng file config.

## Phân Tích Và Benchmark

Phân tích dataset:

```bash
python scripts/Analyze_pretrain.py
python scripts/Analyze_finetune.py
```

Export SkateFormer sang ONNX:

```bash
python scripts/convert_onnxruntime.py --weights results/finetune/best.pt --output deploy/skateformer.onnx
```

Quantize INT8 và benchmark model:

```bash
python scripts/quantize_benmark.py --weights results/finetune/best.pt --input deploy/skateformer.onnx --output deploy/skateformer_int8.onnx --yolo deploy/yolo11s-pose.onnx
```

Benchmark end-to-end pipeline YOLO + action recognition:

```bash
python scripts/benchmark_e2e.py --source data/test_raw_videos/data_fall_detection/N3_buocgiay.mp4 --models onnx int8 --max-frames 300
```

## Deploy

Server deploy cần các file model sau trong thư mục `deploy/`:

```text
deploy/yolo11s-pose.onnx
deploy/skateformer_int8.onnx
```

Chạy server:

```bash
cd deploy
python server.py
```

Hoặc:

```bash
cd deploy
uvicorn server:app --host 0.0.0.0 --port 8000
```

Mở Web UI tại `http://localhost:8000`. Web UI hiện hỗ trợ upload video, chạy webcam, xem kết quả realtime và xem snapshot cảnh báo được lưu trong `deploy/alerts/`.

## Nhãn Phân Loại

| ID | Nhãn | Ý nghĩa |
| ---: | --- | --- |
| 0 | `Normal` | Hành động bình thường |
| 1 | `Fall` | Ngã/té ngã |
| 2 | `Fight` | Ẩu đả/xô xát |

## Lưu Ý Khi Public GitHub

Các file dữ liệu, checkpoint, model export, video upload và snapshot runtime có dung lượng lớn hoặc được sinh tự động. Repo đã cấu hình `.gitignore` cho các nhóm file này. Nếu cần publish model weights, nên dùng Git LFS hoặc GitHub Releases thay vì commit trực tiếp vào Git thường.

Những file deploy bắt buộc như `yolo11s-pose.onnx` và `skateformer_int8.onnx` có thể được tạo bằng script export/quantize hoặc tải từ release tương ứng.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
