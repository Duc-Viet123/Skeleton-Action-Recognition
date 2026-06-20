import os
import json
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


NUM_JOINTS = 18
MAX_BODY = 2
WINDOW_SIZE = 64
CHANNELS = 3


def read_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, None, None

    frames = obj.get("data", [])
    label = obj.get("label", "")
    lab_idx = obj.get("label_index", -1)
    return frames, label, lab_idx


def frames_to_array(frames, num_joints=NUM_JOINTS, max_body=MAX_BODY):
    if not frames:
        return None

    total_frames = len(frames)
    arr = np.zeros((total_frames, CHANNELS, num_joints, max_body), dtype=np.float32)

    for t, frame in enumerate(frames):
        skeletons = frame.get("skeleton", [])
        for m, sk in enumerate(skeletons[:max_body]):
            pose = sk.get("pose", [])
            score = sk.get("score", [])

            for v in range(num_joints):
                if 2 * v + 1 < len(pose):
                    arr[t, 0, v, m] = float(pose[2 * v])
                    arr[t, 1, v, m] = float(pose[2 * v + 1])
                if v < len(score):
                    arr[t, 2, v, m] = float(score[v])

    return arr


def temporal_resize(arr, window_size=WINDOW_SIZE):
    total_frames = arr.shape[0]
    if total_frames == window_size:
        return arr

    indices = np.linspace(0, total_frames - 1, window_size).astype(int)
    return arr[indices]


def normalize_coordinates(arr, eps=1e-6):
    x_p0 = arr[:, 0, :, 0]
    y_p0 = arr[:, 1, :, 0]
    score_p0 = arr[:, 2, :, 0]
    valid_mask = score_p0 > 1e-5

    if not valid_mask.any():
        return arr

    mean_x = x_p0[valid_mask].mean()
    mean_y = y_p0[valid_mask].mean()
    mean_s = score_p0[valid_mask].mean()

    out = arr.copy()
    out[:, 0, :, :] -= mean_x
    out[:, 1, :, :] -= mean_y

    any_valid_mask = arr[:, 2, :, :] > 1e-5
    out[:, 2, :, :][any_valid_mask] -= mean_s

    max_val = np.abs(out[:, :2, :, :]).max()
    if max_val > eps:
        out /= max_val

    return out


def process_one_file(args):
    filepath, window_size, num_joints, max_body = args

    frames, label, lab_idx = read_json(filepath)
    if frames is None or not frames:
        return None

    arr = frames_to_array(frames, num_joints=num_joints, max_body=max_body)
    if arr is None:
        return None

    arr = temporal_resize(arr, window_size=window_size)
    arr = normalize_coordinates(arr)
    arr = arr.transpose(1, 0, 2, 3)

    return arr, lab_idx


def process_split(
    json_dir,
    output_dir,
    split_name,
    window_size=WINDOW_SIZE,
    num_joints=NUM_JOINTS,
    max_body=MAX_BODY,
    num_workers=4,
    max_files=None,
):
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    total = len(json_files)
    print(f"\n[{split_name.upper()}] Found {total} JSON files in {json_dir}")
    print(f"  window_size={window_size}, num_joints={num_joints}, max_body={max_body}")
    print(f"  num_workers={num_workers}")

    args_list = [
        (str(fp), window_size, num_joints, max_body)
        for fp in json_files
    ]

    all_data = []
    all_labels = []
    skipped = 0
    done = 0

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one_file, a): i for i, a in enumerate(args_list)}
            for future in as_completed(futures):
                done += 1
                if done % 5000 == 0:
                    print(f"  Progress: {done}/{total} ({done / total * 100:.1f}%)  skipped={skipped}")

                result = future.result()
                if result is None:
                    skipped += 1
                    continue
                arr, lab_idx = result
                all_data.append(arr)
                all_labels.append(lab_idx)
    else:
        for i, a in enumerate(args_list):
            if i % 5000 == 0:
                print(f"  Progress: {i}/{total} ({i / total * 100:.1f}%)  skipped={skipped}")
            result = process_one_file(a)
            if result is None:
                skipped += 1
                continue
            arr, lab_idx = result
            all_data.append(arr)
            all_labels.append(lab_idx)

    print(f"  Total processed: {len(all_data)}, Skipped (empty): {skipped}")

    if not all_data:
        print(f"  [WARNING] No data for split '{split_name}'. Skipping save.")
        return

    data_arr = np.stack(all_data, axis=0).astype(np.float32)
    label_arr = np.array(all_labels, dtype=np.int64)

    print(f"  Final data shape:  {data_arr.shape}")
    print(f"  Final label shape: {label_arr.shape}")
    print(f"  Data range: [{data_arr.min():.4f}, {data_arr.max():.4f}]")

    if split_name == "train":
        data_path = output_dir / "train_data.npy"
        label_path = output_dir / "train_label.npy"
    elif split_name == "val":
        data_path = output_dir / "val_data.npy"
        label_path = output_dir / "val_label.npy"
    else:
        data_path = output_dir / f"{split_name}_data.npy"
        label_path = output_dir / f"{split_name}_label.npy"

    np.save(str(data_path), data_arr)
    np.save(str(label_path), label_arr)
    print(f"  Saved data  -> {data_path}")
    print(f"  Saved label -> {label_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pretrain JSON skeleton data -> .npy"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "raw",
            "pretrain",
        ),
        help="Thu muc chua raw JSON (co train/ va val/ ben trong)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "pretrain",
        ),
        help="Thu muc luu output (co train/ va val/ ben trong)",
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
        "--num_joints",
        type=int,
        default=NUM_JOINTS,
        help="So keypoints (default: 18)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="So workers de xu ly song song",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Xu ly split nao (default: all)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Gioi han so file (de test nhanh)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        json_dir = raw_dir / split
        output_dir = out_dir / split

        if not json_dir.exists():
            print(f"[WARNING] {json_dir} not found, skipping.")
            continue

        process_split(
            json_dir=json_dir,
            output_dir=output_dir,
            split_name=split,
            window_size=args.window_size,
            num_joints=args.num_joints,
            max_body=args.max_body,
            num_workers=args.num_workers,
            max_files=args.max_files,
        )

    print("\n[DONE] Pretrain preprocessing complete!")


if __name__ == "__main__":
    main()
