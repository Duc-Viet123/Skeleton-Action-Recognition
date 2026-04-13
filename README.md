# Skeleton-Based Action Recognition – SkateFormer

Dự án: **Nhận dạng hành động đáng ngờ** (ngã, ẩu đả) từ dữ liệu skeleton sử dụng mô hình SkateFormer với chiến lược Pre-training tự giám sát.

---

## Cấu trúc dự án

```
datn_dataset/
├── config/              
│   ├── pretrain.yaml     
│   ├── finetune.yaml     
│   └── baseline.yaml    
├── model/               
│   ├── SkateFormer.py    
│   └── SkateFormerPre.py 
├── Feeders/             
│   ├── feeder_pretrain.py   
│   └── feeder_finetune.py   
├── scripts/             
│   ├── train_pretrain.py    
│   ├── train_finetune.py    
│   ├── Analyze_pretrain.py  
│   ├── Analyze_finetune.py  
│   ├── convert_onnxruntime.py   
│   └── quantize_benmark.py      
├── deploy/              
│   ├── server.py                    
│   ├── static/           
│   └── requirements.txt  
├── data/               
├── results/             
├── requirements.txt     
└── .gitignore
```

---

## Tải Dataset

Dữ liệu của dự án có dung lượng khoảng **60 GB**, gồm các file skeleton embeddings (`.npy`) và video gốc. Do dung lượng quá lớn, dữ liệu được lưu trữ trên **Hugging Face Hub** dưới dạng các file `.zip`.

**[📦 Dataset trên Hugging Face](https://huggingface.co/datasets/tdv511/Data_Skeleton_Action_Recognition)**

### Các file zip trong repo HuggingFace

| File | Dung lượng | Nội dung |
|------|-----------|----------|
| `pretrain.zip` | ~3.77 GB | Skeleton data cho pre-training (train/val) |
| `finetune.zip` | ~53.6 MB | Skeleton data cho fine-tuning (train/val) |
| `raw.zip` | ~14.7 GB | Video/skeleton thô (pretrain + finetune) |
| `test_raw_videos.zip` | ~613 MB | Video test |

### Cách tải và giải nén

**Bước 1:** Cài thư viện
```bash
pip install huggingface_hub
```

**Bước 2:** Download và extract (chạy tại thư mục gốc dự án)
```python
from huggingface_hub import hf_hub_download
import zipfile, os

REPO = "tdv511/Data_Skeleton_Action_Recognition"
FILES = ["pretrain.zip", "finetune.zip", "raw.zip", "test_raw_videos.zip"]

os.makedirs("data", exist_ok=True)
for fname in FILES:
    local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", local_dir="data")
    with zipfile.ZipFile(local, "r") as z:
        z.extractall("data")
    os.remove(local)
    print(f"✅ {fname} extracted")
```

Hoặc dùng script có sẵn:
```bash
python download_data.py
```

### Cấu trúc `data/` sau khi extract

```
data/
├── pretrain/
│   ├── train/
│   │   └── train_data.npy
│   └── val/
│       └── val_data.npy
├── finetune/
│   ├── train/
│   └── val/
├── raw/
│   ├── pretrain/
│   └── finetune/
└── test_raw_videos/
```

---

## Cài đặt

### Training
```bash
pip install -r requirements.txt
```

### Deploy
```bash
pip install -r deploy/requirements.txt
```

---

## Cách chạy

### 1. Pre-training (Self-supervised Contrastive Learning)
```bash
python scripts/train_pretrain.py --config config/pretrain.yaml
```

### 2. Fine-tuning
```bash
python scripts/train_finetune.py --config config/finetune.yaml
```

### 3. Baseline (train từ scratch, không pre-train)
```bash
python scripts/train_finetune.py --config config/baseline.yaml
```

### 4. Phân tích dataset
```bash
python scripts/Analyze_pretrain.py
python scripts/Analyze_finetune.py
```

### 5. Export ONNX & Quantize
```bash
python scripts/convert_onnxruntime.py
python scripts/quantize_benmark.py
```

### 6. Chạy server deploy
```bash
cd deploy
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Classes nhận dạng

| ID | Nhãn  | Mô tả                  |
|----|-------|------------------------|
| 0  | Normal| Hành động bình thường  |
| 1  | Fall  | Ngã                    |
| 2  | Fight | Ẩu đả / xô xát        |

---

## Yêu cầu hệ thống

- Python 3.9+
- CUDA 11.8+ (training)
- RAM ≥ 16GB (training), ≥ 8GB (deploy CPU)
