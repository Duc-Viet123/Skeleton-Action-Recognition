
import torch
from model.SkateFormer import SkateFormer
MODEL_WEIGHTS = r"C:\Users\VietTruongDuc\Documents\datn_dataset\results5\best.pt"
OUTPUT_ONNX   = "skateformer1.onnx"

model = SkateFormer(
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
model.load_state_dict(clean, strict=False)
model.eval()

class SkateFormerWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x, index_t):
        return self.model(x, index_t)

wrapper = SkateFormerWrapper(model)
wrapper.eval()

# Shape: (batch, C=3, T=64, V=18, M=2)
dummy_x       = torch.randn(1, 3, 64, 18, 2)
dummy_index_t = torch.zeros(1, 64, dtype=torch.long)

print("Exporting SkateFormer to ONNX...")
torch.onnx.export(
    wrapper,
    (dummy_x, dummy_index_t),
    OUTPUT_ONNX,
    input_names=["input", "index_t"],
    output_names=["logits"],
    dynamic_axes={
        "input":   {0: "batch"},
        "index_t": {0: "batch"},
    },
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,
)
print(f"Done! Saved to {OUTPUT_ONNX}")

import onnx
m = onnx.load(OUTPUT_ONNX)
onnx.checker.check_model(m)
print("ONNX model verified OK")