import argparse
import sys
from pathlib import Path

import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.SkateFormer import Model as SkateFormer


class SkateFormerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, index_t):
        return self.model(x, index_t)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    clean = {}
    for key, value in state_dict.items():
        name = key.replace("module.", "")
        if name.startswith("proj_head."):
            name = name.replace("proj_head.", "head.")
        clean[name] = value
    return clean


def parse_args():
    parser = argparse.ArgumentParser(description="Export SkateFormer checkpoint to ONNX.")
    parser.add_argument(
        "--weights",
        default="results/finetune/best.pt",
        help="Path to fine-tuned SkateFormer checkpoint.",
    )
    parser.add_argument("--output", default="deploy/skateformer.onnx", help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = resolve_path(args.weights)
    output = resolve_path(args.output)

    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = SkateFormer(
        num_classes=3,
        num_people=2,
        num_points=18,
        num_frames=64,
        type_1_size=(8, 6),
        type_2_size=(8, 3),
        type_3_size=(8, 6),
        type_4_size=(8, 3),
    )
    model.load_state_dict(load_checkpoint(weights), strict=False)
    model.eval()

    wrapper = SkateFormerWrapper(model).eval()
    dummy_x = torch.randn(1, 3, 64, 18, 2)
    dummy_index_t = torch.zeros(1, 64, dtype=torch.long)

    print(f"Exporting SkateFormer to ONNX: {output}")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_index_t),
        str(output),
        input_names=["input", "index_t"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "index_t": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    model_onnx = onnx.load(str(output))
    onnx.checker.check_model(model_onnx)
    print("ONNX model verified OK")


if __name__ == "__main__":
    main()
