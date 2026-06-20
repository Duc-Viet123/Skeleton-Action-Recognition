import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "tdv511/Data_Skeleton_Action_Recognition"
DEFAULT_FILES = [
    "pretrain.zip",
    "finetune.zip",
    "raw.zip",
    "test_raw_videos.zip",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and optionally extract project dataset archives from Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--local-dir", default="data", help="Directory to store downloaded data")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Zip files to download. Defaults to all project archives.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only download zip files.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip files after extraction.")
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname in args.files:
        print(f"[Download] {fname}")
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=fname,
            repo_type="dataset",
            local_dir=str(local_dir),
        )

        if args.no_extract:
            print(f"  saved: {local_path}")
            continue

        print(f"  extracting to: {local_dir}")
        with zipfile.ZipFile(local_path, "r") as zip_file:
            zip_file.extractall(local_dir)

        if not args.keep_zip:
            os.remove(local_path)

        print(f"  done: {fname}")


if __name__ == "__main__":
    main()
