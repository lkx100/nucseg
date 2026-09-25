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
    upsample + conv stages to full resolution.

    hires_skip: a conv stem on the image gives the last two stages skip features at 1/2 and full resolution.
    input_scale: the input is upsampled by this factor inside the model and the logits are averaged back
    down, so callers always work at the image's own resolution.
    """

    def __init__(self, encoder, pretrained, img_size, window_size, decoder_channels, encoder_kwargs=None,
                 hires_skip=False, input_scale=1):
        super().__init__()
        self.input_scale = input_scale
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
                                         img_size=img_size * input_scale, window_size=window_size,
                                         **(encoder_kwargs or {}))
        self.nhwc = str(getattr(self.encoder, "output_fmt", "NCHW")).endswith("NHWC")
        enc = self.encoder.feature_info.channels()
        c = decoder_channels
        self.skip_blocks = nn.ModuleList()
        cin = enc[3]
        for skip, cout in zip(enc[2::-1], c[:3]):
            self.skip_blocks.append(ConvBlock(cin + skip, cout))
            cin = cout
        self.stem = None
        if hires_skip:
            self.stem = nn.ModuleList([ConvBlock(3, c[4]), nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[4], c[3]))])
        extra = (c[3], c[4]) if hires_skip else (0, 0)
        self.up_blocks = nn.ModuleList([ConvBlock(c[2] + extra[0], c[3]), ConvBlock(c[3] + extra[1], c[4])])
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
        for block, skip in zip(self.skip_blocks, feats[2::-1]):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], 1))
        hires = []
        if self.stem is not None:
            full = self.stem[0](x)
            hires = [self.stem[1](full), full]
        for n, block in enumerate(self.up_blocks):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = block(torch.cat([y, hires[n]], 1) if hires else y)
        y = self.head(y)
        return F.avg_pool2d(y, self.input_scale) if self.input_scale != 1 else y


def build_model(cfg, smoke=False):
    m = dict(cfg["model"])
    kwargs = {}
    if smoke:  # tiny untrained Swin, CPU-sized
        m["pretrained"] = False
        kwargs = dict(embed_dim=24, depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8))
    return SwinUNet(m["encoder"], m["pretrained"], cfg["data"]["crop"], m["window_size"], m["decoder_channels"], kwargs,
                    hires_skip=m.get("hires_skip", False), input_scale=m.get("input_scale", 1))
