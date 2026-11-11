"""Full-image prediction by overlapping tiles or one whole-image pass, and the FGSM attack used for the robustness score."""
import numpy as np
import torch
import torch.nn.functional as F

from nucseg.losses import bce_dice


def _positions(n, tile, stride):
    pos = list(range(0, n - tile + 1, stride))
    return pos if pos[-1] == n - tile else pos + [n - tile]


def _pad(t, tile):
    """Pad (N, C, H, W) at the bottom and right up to `tile`; reflect where possible."""
    ph, pw = max(0, tile - t.shape[-2]), max(0, tile - t.shape[-1])
    if not (ph or pw):
        return t
    mode = "reflect" if ph < t.shape[-2] and pw < t.shape[-1] else "replicate"
    return F.pad(t, (0, pw, 0, ph), mode=mode)


def _tiles(h, w, tile, overlap):
    stride = max(1, int(tile * (1 - overlap)))
    return [(y, x) for y in _positions(h, tile, stride) for x in _positions(w, tile, stride)]


def _to_tensor(img):
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float() / 255


def _tta_logits(model, crops):
    """Mean logits over the 8 orientations of a batch of images (4 rotations, each with and without a flip)."""
    out = 0
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(crops, k, (2, 3))
            o = model(t.flip(3) if flip else t)
            out = out + torch.rot90(o.flip(3) if flip else o, -k, (2, 3))
    return out / 8


def _tile_logits(model, x, tile, overlap, device, amp, batch, tta):
    """Logits (H, W) of a padded (1, 3, H, W) image from overlapping tiles, averaged where tiles overlap."""
    H, W = x.shape[-2:]
    acc = torch.zeros(H, W, device=device)
    cnt = torch.zeros(H, W, device=device)
    pos = _tiles(H, W, tile, overlap)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        crops = torch.cat([x[..., y:y + tile, x0:x0 + tile] for y, x0 in chunk])
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = (_tta_logits(model, crops) if tta else model(crops)).float()
        for (y, x0), lg in zip(chunk, logits):
            acc[y:y + tile, x0:x0 + tile] += lg[0]
            cnt[y:y + tile, x0:x0 + tile] += 1
    return acc / cnt


def _whole_logits(model, x, tile, amp, tta):
    """Logits (h, w) from one pass over the whole image. The image is reflect-padded at the bottom and right to a
    multiple of `tile`, so a Swin encoder built with `strict_img_size: false` never pads its windows with zeros."""
    h, w = x.shape[-2:]
    ph, pw = -h % tile, -w % tile
    mode = "reflect" if ph < h and pw < w else "replicate"
    xp = F.pad(x, (0, pw, 0, ph), mode=mode) if (ph or pw) else x
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        logits = (_tta_logits(model, xp) if tta else model(xp)).float()
    return logits[0, 0, :h, :w]


@torch.no_grad()
def predict(model, x, tile, overlap, device, amp, batch=16, tta=False, view="tiles"):
    """Probabilities (H, W) for one image. `x` is uint8 (H, W, 3) or a float tensor (1, 3, H, W) in [0, 1].
    `view` is "tiles" (overlapping tiles, logits averaged where they overlap), "whole" (one pass over the whole
    image) or "both" (the mean of the two logit maps). With `tta`, every pass is the 8-orientation mean."""
    assert view in ("tiles", "whole", "both"), view
    if isinstance(x, np.ndarray):
        x = _to_tensor(x)
    h, w = x.shape[-2:]
    x = x.to(device)
    parts = []
    if view in ("tiles", "both"):
        parts.append(_tile_logits(model, _pad(x, tile), tile, overlap, device, amp, batch, tta)[:h, :w])
    if view in ("whole", "both"):
        parts.append(_whole_logits(model, x, tile, amp, tta))
    return torch.sigmoid(sum(parts) / len(parts)).cpu().numpy()


def fgsm(model, img, gt, eps, tile, overlap, device, batch=16):
    """The image after one FGSM step of size `eps` against `model`, in fp32. The gradient of the summed tile
    losses is taken with respect to the whole image, so overlapping tiles share one perturbation."""
    h, w = gt.shape
    x = _pad(_to_tensor(img).to(device), tile).requires_grad_(True)
    y = _pad(torch.from_numpy(gt.astype(np.float32))[None, None].to(device), tile)
    grad = torch.zeros_like(x)
    pos = _tiles(*x.shape[-2:], tile, overlap)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        crops = torch.cat([x[..., r:r + tile, c:c + tile] for r, c in chunk])
        target = torch.cat([y[..., r:r + tile, c:c + tile] for r, c in chunk])
        (g,) = torch.autograd.grad(bce_dice(model(crops), target), x)
        grad += g
    return (x.detach() + eps * grad.sign()).clamp(0, 1)[..., :h, :w]
