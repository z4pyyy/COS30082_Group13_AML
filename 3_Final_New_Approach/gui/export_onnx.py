"""Export the two DINO-CPS models to int8 ONNX for the static (in-browser) Space.

    python3.14 3_Final_New_Approach/gui/export_onnx.py --weights DIR      # DIR has gui_crop.pt, gui_dis_attn.pt
    python3.14 3_Final_New_Approach/gui/export_onnx.py --dummy            # random weights, for size / speed tests only

Writes 3_Final_New_Approach/gui/static/models/dino_cps_crop.int8.onnx   input "input" [1,3,224,224] -> "crop_logp" [1,55]
       3_Final_New_Approach/gui/static/models/dino_cps_disease.int8.onnx  -> "disease_logp" [1,175], "att" [1,256]
int8 = dynamic quantisation of the Linear/MatMul weights (onnxruntime.quantization), activations stay float.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm
from onnxruntime.quantization import QuantType, quantize_dynamic

HERE = Path(__file__).resolve().parent
OUT = HERE / "static" / "models"
BACKBONE = "vit_large_patch14_dinov2.lvd142m"


class AttnPool(nn.Module):   # same as 3_Final_New_Approach/training/train.py
    def __init__(self, d, h=256):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, t):
        a = self.w(torch.tanh(self.v(t)) * torch.sigmoid(self.u(t))).squeeze(-1).softmax(1)
        return (a.unsqueeze(-1) * t).sum(1), a


class Net(nn.Module):        # inference part of 3_Final_New_Approach/training/train.py Net
    def __init__(self, n_out, attn):
        super().__init__()
        self.bb = timm.create_model(BACKBONE, pretrained=False, num_classes=0, img_size=224)
        self.pool = AttnPool(self.bb.num_features) if attn else None
        self.head = nn.Linear(self.bb.num_features, n_out)

    def forward(self, x):
        t = self.bb.forward_features(x)
        if self.pool is None:
            return self.head(t[:, 0]).log_softmax(1)
        z, a = self.pool(t[:, self.bb.num_prefix_tokens:])
        return self.head(z).log_softmax(1), a


def build(name, n_out, attn, wdir):
    m = Net(n_out, attn)
    if wdir:
        sd = torch.load(Path(wdir) / f"{name}.pt", map_location="cpu")
        m.load_state_dict({k: v.float() for k, v in sd.items() if not k.startswith(("adv.", "dom."))})
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    if not (a.weights or a.dummy):
        sys.exit("give --weights DIR or --dummy")
    OUT.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, 224, 224)
    for name, n, attn, stem, outs in (("gui_crop", 55, False, "dino_cps_crop", ["crop_logp"]),
                                      ("gui_dis_attn", 175, True, "dino_cps_disease", ["disease_logp", "att"])):
        m = build(name, n, attn, a.weights)
        fp32 = OUT / f"{stem}.fp32.onnx"
        torch.onnx.export(m, x, fp32, input_names=["input"], output_names=outs, opset_version=17, dynamo=False)
        quantize_dynamic(fp32, OUT / f"{stem}.int8.onnx", weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"])   # ORT-web wasm has no ConvInteger
        fp32.unlink()
        for f in fp32.parent.glob(f"{stem}.fp32.onnx*"):   # external-data sidecar, if any
            f.unlink()
        print(stem, f"{(OUT / f'{stem}.int8.onnx').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
