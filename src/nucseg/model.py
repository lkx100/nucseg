"""Swin encoder with a U-Net decoder. Takes images in [0, 1] and returns 1-channel logits at input size."""
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


class SwinUNet(nn.Module):
    """Swin features at 1/4..1/32 scale, three skip-connected decoder stages back to 1/4, then two
    upsample + conv stages to full resolution."""

    def __init__(self, encoder, pretrained, img_size, window_size, decoder_channels, encoder_kwargs=None):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
                                         img_size=img_size, window_size=window_size, **(encoder_kwargs or {}))
        self.nhwc = str(getattr(self.encoder, "output_fmt", "NCHW")).endswith("NHWC")
        enc = self.encoder.feature_info.channels()
        c = decoder_channels
        self.skip_blocks = nn.ModuleList()
        cin = enc[3]
        for skip, cout in zip(enc[2::-1], c[:3]):
            self.skip_blocks.append(ConvBlock(cin + skip, cout))
            cin = cout
        self.up_blocks = nn.ModuleList([ConvBlock(c[2], c[3]), ConvBlock(c[3], c[4])])
        self.head = nn.Conv2d(c[4], 1, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        feats = self.encoder((x - self.mean) / self.std)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        y = feats[3]
        for block, skip in zip(self.skip_blocks, feats[2::-1]):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], 1))
        for block in self.up_blocks:
            y = block(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(y)


def build_model(cfg, smoke=False):
    m = dict(cfg["model"])
    kwargs = {}
    if smoke:  # tiny untrained Swin, CPU-sized
        m["pretrained"] = False
        kwargs = dict(embed_dim=24, depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8))
    return SwinUNet(m["encoder"], m["pretrained"], cfg["data"]["crop"], m["window_size"], m["decoder_channels"], kwargs)
