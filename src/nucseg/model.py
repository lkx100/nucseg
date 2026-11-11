"""Transformer encoder (Swin by default) with a U-Net decoder. Takes images in [0, 1] and returns 1-channel logits at
input size."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018): the decoder feature g decides, pixel by pixel, how much of the
    skip feature x passes. g and x have the same height and width."""

    def __init__(self, cg, cx):
        super().__init__()
        ci = max(8, cx // 2)
        self.wg = nn.Sequential(nn.Conv2d(cg, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.wx = nn.Sequential(nn.Conv2d(cx, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.psi = nn.Sequential(nn.Conv2d(ci, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class SCSE(nn.Module):
    """Concurrent channel and spatial squeeze-excitation (Roy et al. 2018): reweights the channels of a feature map
    from its global average, and its pixels from a 1x1 conv, and adds the two."""

    def __init__(self, c, reduction=16):
        super().__init__()
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // reduction), 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(max(4, c // reduction), c, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class SwinUNet(nn.Module):
    """Swin features at 1/4..1/32 scale, three skip-connected decoder stages back to 1/4, then two
    upsample + conv stages to full resolution.

    hires_skip: a conv stem on the image gives the last two stages skip features at 1/2 and full resolution.
    input_scale: the input is upsampled by this factor inside the model and the logits are averaged back
    down, so callers always work at the image's own resolution.
    encoder: any timm model with features at 1/4, 1/8, 1/16 and 1/32 scale; out_indices picks them.
    window_size: passed to the encoder only when not None.
    decoder_attention: None, "gate" (an attention gate on every skip) or "scse" (scSE after every decoder block).
    """

    def __init__(self, encoder, pretrained, img_size, window_size, decoder_channels, encoder_kwargs=None,
                 hires_skip=False, input_scale=1, out_indices=(0, 1, 2, 3), decoder_attention=None):
        super().__init__()
        assert decoder_attention in (None, "gate", "scse"), decoder_attention
        self.input_scale = input_scale
        kwargs = dict(encoder_kwargs or {})
        if window_size is not None:
            kwargs["window_size"] = window_size
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True,
                                         out_indices=tuple(out_indices), img_size=img_size * input_scale, **kwargs)
        assert self.encoder.feature_info.reduction() == [4, 8, 16, 32], self.encoder.feature_info.reduction()
        self.nhwc = str(getattr(self.encoder, "output_fmt", "NCHW")).endswith("NHWC")
        enc = self.encoder.feature_info.channels()
        c = decoder_channels

        def block(cin, cout):
            return nn.Sequential(ConvBlock(cin, cout), SCSE(cout)) if decoder_attention == "scse" else ConvBlock(cin, cout)

        self.skip_blocks = nn.ModuleList()
        self.gates = nn.ModuleList() if decoder_attention == "gate" else None
        cin = enc[3]
        for skip, cout in zip(enc[2::-1], c[:3]):
            self.skip_blocks.append(block(cin + skip, cout))
            if self.gates is not None:
                self.gates.append(AttentionGate(cin, skip))
            cin = cout
        self.stem = None
        if hires_skip:
            self.stem = nn.ModuleList([ConvBlock(3, c[4]), nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[4], c[3]))])
            if self.gates is not None:
                self.gates.extend([AttentionGate(c[2], c[3]), AttentionGate(c[3], c[4])])
        extra = (c[3], c[4]) if hires_skip else (0, 0)
        self.up_blocks = nn.ModuleList([block(c[2] + extra[0], c[3]), block(c[3] + extra[1], c[4])])
        self.head = nn.Conv2d(c[4], 1, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if self.input_scale != 1:
            x = F.interpolate(x, scale_factor=self.input_scale, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.encoder(x)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        y = feats[3]
        for n, (block, skip) in enumerate(zip(self.skip_blocks, feats[2::-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.gates is not None:
                skip = self.gates[n](y, skip)
            y = block(torch.cat([y, skip], 1))
        hires = []
        if self.stem is not None:
            full = self.stem[0](x)
            hires = [self.stem[1](full), full]
        for n, block in enumerate(self.up_blocks):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            if hires and self.gates is not None:
                hires[n] = self.gates[3 + n](y, hires[n])
            y = block(torch.cat([y, hires[n]], 1) if hires else y)
        y = self.head(y)
        return F.avg_pool2d(y, self.input_scale) if self.input_scale != 1 else y


def build_model(cfg, smoke=False):
    m = dict(cfg["model"])
    kwargs = dict(m.get("encoder_kwargs") or {})  # e.g. strict_img_size: false lets Swin take any input size
    if smoke:  # tiny untrained encoder of the same family, CPU-sized
        m["pretrained"] = False
        if m["encoder"].startswith("maxvit"):
            kwargs |= dict(embed_dim=(32, 64, 128, 256), depths=(1, 1, 1, 1), stem_width=16)
        else:
            kwargs |= dict(embed_dim=24, depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8))
    return SwinUNet(m["encoder"], m["pretrained"], cfg["data"]["crop"], m.get("window_size"), m["decoder_channels"],
                    kwargs, hires_skip=m.get("hires_skip", False), input_scale=m.get("input_scale", 1),
                    out_indices=m.get("out_indices", (0, 1, 2, 3)), decoder_attention=m.get("decoder_attention"))
