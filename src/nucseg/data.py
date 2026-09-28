"""Loading, splits and training crops. Images stay uint8 in memory; the model does the normalization."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.metrics import gt_mask  # noqa: E402


def find_data_root(explicit=None):
    """The `stage1_train` folder: explicit path, else searched under /kaggle/input, else under data/."""
    if explicit:
        return Path(explicit)
    for base in (Path("/kaggle/input"), ROOT / "data"):
        if base.exists():
            for p in base.rglob("stage1_train"):
                if p.is_dir():
                    return p
    raise FileNotFoundError("no stage1_train folder under /kaggle/input or data/")


def load_splits():
    return json.loads((ROOT / "eval" / "splits.json").read_text())


def fold_ids(splits, fold):
    """Train, stop and validation ids for one fold. The test set is never returned."""
    val = splits["folds"][str(fold)]
    stop = splits["stop"][str(fold)]
    pool = [i for k, ids in splits["folds"].items() for i in ids]
    train = sorted(set(pool) - set(val) - set(stop))
    return train, stop, val


_CACHE = {}


def load_samples(root, ids):
    """id -> (RGB uint8 image, boolean ground-truth mask). Cached, so later folds reuse earlier reads."""
    for i in ids:
        if i not in _CACHE:
            d = Path(root) / i
            _CACHE[i] = (np.asarray(Image.open(next((d / "images").glob("*.png"))).convert("RGB")), gt_mask(d))
    return {i: _CACHE[i] for i in ids}


def elastic(img, mask, rng, px, grid):
    """Smooth random warp: displacements with std `px` pixels on a `grid` x `grid` lattice, upsampled bicubically."""
    h, w = mask.shape
    d = torch.from_numpy(rng.normal(0, px, (1, 2, grid, grid)).astype(np.float32))
    d = F.interpolate(d, size=(h, w), mode="bicubic", align_corners=True)
    d = d * torch.tensor([2 / w, 2 / h]).view(1, 2, 1, 1)  # pixels -> grid_sample's [-1, 1] units
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    g = (torch.stack([xs, ys])[None] + d).permute(0, 2, 3, 1)
    im = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float()
    m = torch.from_numpy(np.array(mask)).float()[None, None]
    im = F.grid_sample(im, g, mode="bilinear", padding_mode="reflection", align_corners=True)
    m = F.grid_sample(m, g, mode="nearest", padding_mode="reflection", align_corners=True)
    return im[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy(), m[0, 0].numpy() > 0.5


def hue_saturation(img, rng, hue, saturation):
    """Shift hue by up to `hue` of the colour wheel and scale saturation by 1 +/- `saturation`.
    Grayscale images have zero saturation, so only colour (H&E, brightfield) images change."""
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(round(rng.uniform(-hue, hue) * 255))) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-saturation, saturation)), 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


class CropDataset(torch.utils.data.Dataset):
    """Random crops with scale, flip, 90-degree rotation and brightness/contrast jitter.

    `aug` (optional) adds elastic warps, hue/saturation, gamma, blur and noise. Without it no extra
    random numbers are drawn, so older configs get the same crops as before.
    One epoch visits every image `crops_per_image` times. Returns float images in [0, 1] (3, crop, crop)
    and float masks (1, crop, crop).
    """

    def __init__(self, samples, crop, crops_per_image, scale, brightness, contrast, aug=None):
        self.items = list(samples.values())
        self.crop, self.k = crop, crops_per_image
        self.scale, self.brightness, self.contrast = scale, brightness, contrast
        self.aug = aug

    def __len__(self):
        return len(self.items) * self.k

    def __getitem__(self, i):
        # Seeded from torch so DataLoader workers get distinct, reproducible streams.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        img, mask = self.items[i // self.k]
        src = int(round(self.crop / rng.uniform(*self.scale)))
        h, w = mask.shape
        ph, pw = max(0, src - h), max(0, src - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        y0 = rng.integers(0, mask.shape[0] - src + 1)
        x0 = rng.integers(0, mask.shape[1] - src + 1)
        img, mask = img[y0:y0 + src, x0:x0 + src], mask[y0:y0 + src, x0:x0 + src]
        if src != self.crop:
            img = np.asarray(Image.fromarray(img).resize((self.crop, self.crop), Image.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize((self.crop, self.crop), Image.NEAREST))
        if rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        r = rng.integers(4)
        img, mask = np.rot90(img, r), np.rot90(mask, r)
        a = self.aug
        if a and rng.random() < a["elastic_p"]:
            img, mask = elastic(img, mask, rng, a["elastic_px"], a["elastic_grid"])
        if a:
            img = hue_saturation(img, rng, a["hue"], a["saturation"])
        if a and rng.random() < a["blur_p"]:
            img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).filter(
                ImageFilter.GaussianBlur(rng.uniform(*a["blur_sigma"]))))
        x = img.astype(np.float32) / 255
        if a:
            x = x ** float(np.exp(rng.uniform(*np.log(a["gamma"]))))  # log-uniform, so 1/g and g are equally likely
        x = x * (1 + rng.uniform(-self.contrast, self.contrast)) + rng.uniform(-self.brightness, self.brightness)
        if a and rng.random() < a["noise_p"]:
            x = x + rng.normal(0, rng.uniform(0, a["noise_std"]), x.shape).astype(np.float32)
        x = np.clip(x, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None])
