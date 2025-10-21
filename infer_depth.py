import argparse, os, cv2, torch, numpy as np
from pathlib import Path

# ---------------------------
# Model loaders (MiDaS, Zoe)
# ---------------------------
def load_midas(model_name="DPT_Large", device="cuda"):
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_name else transforms.small_transform
    midas.to(device).eval()
    return midas, transform

def load_zoedepth(variant="zoedepth_n", device="cuda"):
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True) if variant=="zoedepth_n" \
        else torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)
    zoe.to(device).eval()
    return zoe

# ---------------------------
# Utilities
# ---------------------------
def normalize_to_16bit(depth):
    valid = np.isfinite(depth)
    if not np.any(valid):
        raise RuntimeError("Depth has no finite values.")
    vmin, vmax = np.percentile(depth[valid], [2, 98])
    depth = np.clip(depth, vmin, vmax)
    depth = (depth - vmin) / max(1e-9, (vmax - vmin))
    return (depth * 65535.0).astype(np.uint16)

def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def run_midas(midas, transform, rgb, device):
    inp = transform(rgb).to(device)
    pred = midas(inp)
    depth = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return 1.0 / np.maximum(depth, 1e-9)

@torch.no_grad()
def run_zoe(zoe, rgb, device):
    img = torch.from_numpy(rgb).permute(2,0,1).float()[None] / 255.0
    out = zoe.infer(img.to(device))
    return out.squeeze().cpu().numpy()

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", type=str, help="Directory containing 512x512 RGB images")
    ap.add_argument("--model", choices=["midas_large","midas_hybrid","midas_small","zoedepth_n","zoedepth_k"],
                    default="midas_large")
    ap.add_argument("--out_dir", type=str, default="depth_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model.startswith("midas"):
        name = {"midas_large":"DPT_Large","midas_hybrid":"DPT_Hybrid","midas_small":"MiDaS_small"}[args.model]
        model, transform = load_midas(name, device)
        infer_fn = lambda rgb: run_midas(model, transform, rgb, device)
    else:
        variant = "zoedepth_n" if args.model=="zoedepth_n" else "zoedepth_k"
        model = load_zoedepth(variant, device)
        infer_fn = lambda rgb: run_zoe(model, rgb, device)

    imgs = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
    if not imgs:
        print("No images found in", src_dir)
        return

    for img_path in imgs:
        print(f"Processing {img_path.name}...")
        rgb = read_rgb(img_path)
        depth = infer_fn(rgb)

        # Save depth as .npy and 16-bit PNG using just basename
        base = img_path.stem
        np.save(out_dir / f"{base}.npy", depth)
        depth16 = normalize_to_16bit(depth)
        cv2.imwrite(str(out_dir / f"{base}.png"), depth16)

    print(f"✅ Done. Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()

