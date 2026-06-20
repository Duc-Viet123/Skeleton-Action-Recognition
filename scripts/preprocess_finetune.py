import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not found. Install: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Install: pip install ultralytics")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not found. Progress bar disabled.")


NUM_JOINTS = 18
MAX_BODY = 2
WINDOW_SIZE = 64
CHANNELS = 3

CLASS_MAP = {
    "Normal": 0,
    "Fall": 1,
    "Fight": 2,
}

YOLO_TO_COCO18_NECK_IDX = 17
YOLO_LSHOULDER_IDX = 5
YOLO_RSHOULDER_IDX = 6


def extract_keypoints_from_video(video_path, model, max_body=MAX_BODY):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = model(frame, verbose=False)
        frame_arr = np.zeros((CHANNELS, NUM_JOINTS, max_body), dtype=np.float32)

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None and len(result.keypoints) > 0:
                kps_data = result.keypoints.data.cpu().numpy()

                if len(kps_data) > 1:
                    person_scores = kps_data[:, :, 2].sum(axis=1)
                    sorted_idx = np.argsort(person_scores)[::-1]
                    kps_data = kps_data[sorted_idx]

                num_persons = min(len(kps_data), max_body)

                for m in range(num_persons):
                    person_kps = kps_data[m]

                    x_norm = person_kps[:, 0] / w
                    y_norm = person_kps[:, 1] / h
                    scores = person_kps[:, 2]

                    x_norm = np.clip(x_norm, 0.0, 1.0)
                    y_norm = np.clip(y_norm, 0.0, 1.0)
                    scores = np.clip(scores, 0.0, 1.0)

                    frame_arr[0, :17, m] = x_norm
                    frame_arr[1, :17, m] = y_norm
                    frame_arr[2, :17, m] = scores

                    ls_x = x_norm[YOLO_LSHOULDER_IDX]
                    rs_x = x_norm[YOLO_RSHOULDER_IDX]
                    ls_y = y_norm[YOLO_LSHOULDER_IDX]
                    rs_y = y_norm[YOLO_RSHOULDER_IDX]
                    ls_s = scores[YOLO_LSHOULDER_IDX]
                    rs_s = scores[YOLO_RSHOULDER_IDX]

                    frame_arr[0, YOLO_TO_COCO18_NECK_IDX, m] = (ls_x + rs_x) / 2
                    frame_arr[1, YOLO_TO_COCO18_NECK_IDX, m] = (ls_y + rs_y) / 2
                    frame_arr[2, YOLO_TO_COCO18_NECK_IDX, m] = (ls_s + rs_s) / 2

        frames_data.append(frame_arr)

    cap.release()

    if not frames_data:
        return None

    return frames_data


def temporal_resize(frames_data, window_size=WINDOW_SIZE):
    total_frames = len(frames_data)
    if total_frames == 0:
        return None

    arr = np.stack(frames_data, axis=0)

    if total_frames == window_size:
        return arr

    indices = np.linspace(0, total_frames - 1, window_size).astype(int)
    return arr[indices]


def process_video(video_path, model, label_idx, window_size=WINDOW_SIZE, max_body=MAX_BODY):
    frames_data = extract_keypoints_from_video(video_path, model, max_body=max_body)
    if frames_data is None or len(frames_data) == 0:
        print(f"  [SKIP] No frames: {video_path.name}")
        return None

    if len(frames_data) < 4:
        print(f"  [SKIP] Too few frames ({len(frames_data)}): {video_path.name}")
        return None

    arr = temporal_resize(frames_data, window_size=window_size)
    if arr is None:
        return None

    arr = arr.transpose(1, 0, 2, 3)

    return arr.astype(np.float32), label_idx


def collect_video_files(raw_dir):
    raw_dir = Path(raw_dir)
    video_list = []
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}

    for cls_name, label_idx in CLASS_MAP.items():
        cls_dir = raw_dir / cls_name
        if not cls_dir.exists():
            print(f"[WARNING] Class directory not found: {cls_dir}")
            continue

        cls_files = [
            f for f in sorted(cls_dir.iterdir())
            if f.suffix in valid_exts
        ]
        print(f"  {cls_name} (label={label_idx}): {len(cls_files)} videos")
        for fp in cls_files:
            video_list.append((fp, label_idx))

    return video_list


def split_dataset(video_list, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in video_list:
        label_idx = item[1]
        by_class[label_idx].append(item)

    train_list, val_list, test_list = [], [], []

    for label_idx in sorted(by_class.keys()):
        items = by_class[label_idx][:]
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_list.extend(items[:n_train])
        val_list.extend(items[n_train:n_train + n_val])
        test_list.extend(items[n_train + n_val:])

        cls_name = [k for k, v in CLASS_MAP.items() if v == label_idx][0]
        print(f"  {cls_name}: train={n_train}, val={n_val}, test={n_test}")

    return train_list, val_list, test_list


def process_and_save(
    video_list,
    split_name,
    output_dir,
    model,
    window_size=WINDOW_SIZE,
    max_body=MAX_BODY,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    all_labels = []
    skipped = 0

    iterator = tqdm(video_list, desc=f"[{split_name}]") if HAS_TQDM else video_list

    for video_path, label_idx in iterator:
        result = process_video(
            video_path=video_path,
            model=model,
            label_idx=label_idx,
            window_size=window_size,
            max_body=max_body,
        )
        if result is None:
            skipped += 1
            continue
        arr, lbl = result
        all_data.append(arr)
        all_labels.append(lbl)

    print(f"\n[{split_name}] Processed: {len(all_data)}, Skipped: {skipped}")

    if not all_data:
        print(f"[WARNING] No data for split '{split_name}'. Skipping save.")
        return

    data_arr = np.stack(all_data, axis=0).astype(np.float32)
    label_arr = np.array(all_labels, dtype=np.int64)

    print(f"  Final data shape:  {data_arr.shape}")
    print(f"  Final label shape: {label_arr.shape}")
    print(f"  Data range: [{data_arr.min():.4f}, {data_arr.max():.4f}]")

    cls_counts = dict(zip(*np.unique(label_arr, return_counts=True)))
    cls_names = {v: k for k, v in CLASS_MAP.items()}
    for lbl, cnt in sorted(cls_counts.items()):
        print(f"    {cls_names.get(lbl, lbl)}: {cnt} samples")

    if split_name == "train":
        data_path = output_dir / "train_data.npy"
        label_path = output_dir / "train_label.npy"
    elif split_name == "val":
        data_path = output_dir / "val_data.npy"
        label_path = output_dir / "val_label.npy"
    elif split_name == "test":
        data_path = output_dir / "test_data.npy"
        label_path = output_dir / "test_label.npy"
    else:
        data_path = output_dir / f"{split_name}_data.npy"
        label_path = output_dir / f"{split_name}_label.npy"

    np.save(str(data_path), data_arr)
    np.save(str(label_path), label_arr)
    print(f"  Saved data  -> {data_path}")
    print(f"  Saved label -> {label_path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Preprocess finetune video data -> .npy using YOLOv8-pose"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(root, "data", "raw", "finetune"),
        help="Thu muc chua raw video (co Fall/, Fight/, Normal/ ben trong)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root, "data", "finetune"),
        help="Thu muc luu output (co train/, val/, test/ ben trong)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLOv8 pose model (default: yolov8n-pose.pt). Lon hon: yolov8s-pose.pt, yolov8m-pose.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 0 (GPU), cpu (default: 0)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=WINDOW_SIZE,
        help="So frame output (default: 64)",
    )
    parser.add_argument(
        "--max_body",
        type=int,
        default=MAX_BODY,
        help="So nguoi toi da moi frame (default: 2)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Ti le train (default: 0.70)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ti le val (default: 0.15, phan con lai la test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho split (default: 42)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Chi xu ly split nao (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  FINETUNE PREPROCESSING")
    print(f"{'=' * 60}")
    print(f"  raw_dir    : {args.raw_dir}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  model      : {args.model}")
    print(f"  device     : {args.device}")
    print(f"  window_size: {args.window_size}")
    print(f"  max_body   : {args.max_body}")
    print(f"  train/val/test ratio: {args.train_ratio}/{args.val_ratio}/{1 - args.train_ratio - args.val_ratio:.2f}")
    print(f"{'=' * 60}\n")

    print(f"[1/4] Loading YOLOv8-pose model: {args.model}")
    model = YOLO(args.model)

    print(f"\n[2/4] Collecting video files from: {args.raw_dir}")
    video_list = collect_video_files(args.raw_dir)
    print(f"  Total videos: {len(video_list)}")

    if not video_list:
        print("[ERROR] No video files found. Check --raw_dir")
        return

    print(f"\n[3/4] Splitting dataset (seed={args.seed}):")
    train_list, val_list, test_list = split_dataset(
        video_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Total - train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    print("\n[4/4] Processing videos...")

    split_map = {
        "train": (train_list, os.path.join(args.out_dir, "train")),
        "val": (val_list, os.path.join(args.out_dir, "val")),
        "test": (test_list, os.path.join(args.out_dir, "test")),
    }

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split_name in splits_to_run:
        vlist, out_subdir = split_map[split_name]
        print(f"\n--- Processing {split_name} ({len(vlist)} videos) ---")
        process_and_save(
            video_list=vlist,
            split_name=split_name,
            output_dir=out_subdir,
            model=model,
            window_size=args.window_size,
            max_body=args.max_body,
        )

    print(f"\n{'=' * 60}")
    print("  [DONE] Finetune preprocessing complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
