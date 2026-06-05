import os
import re

import torch
import torch.nn as nn

from .Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from .Hallucination_arch import (
    Hallucination_Generator,
    Hallucination_Generator_Direct,
    Hallucination_Generator_FusedBN,
    Hallucination_Generator_ResizeConv,
)


class HG_Composite(nn.Module):
    """Composite HDRTVNet++ pipeline: AGCM+LE followed by HG refinement."""

    def __init__(self, classifier="color_condition", cond_c=6,
                 in_nc=3, out_nc=3, nf=32, act_type="relu",
                 weighting_network=False, hg_nf=64, mask_r=0.75,
                 hg_arch=None, le_arch=None):
        super().__init__()
        hg_arch = str(
            hg_arch or os.environ.get("HDRTVNET_HG_ARCH", "pixelshuffle")
        ).strip().lower()
        canonical_hg_arch = hg_arch.replace("-", "").replace("_", "")
        self.base = Ensemble_AGCM_LE(
            classifier=classifier,
            cond_c=cond_c,
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
            le_arch=le_arch,
        )
        direct_match = re.fullmatch(
            r"directh(2|4|8|16|32)wide([0-9]+)x([0-9]+)",
            canonical_hg_arch,
        )
        if direct_match:
            bottleneck_scale = int(direct_match.group(1))
            wide_nf = int(direct_match.group(2))
            trunk_depth = int(direct_match.group(3))
            self.hg = Hallucination_Generator_Direct(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=hg_nf,
                sf=2,
                trunk_depth=trunk_depth,
                wide_nf=wide_nf,
                bottleneck_scale=bottleneck_scale,
            )
            self.hg_arch = f"directh{bottleneck_scale}wide{wide_nf}x{trunk_depth}"
        elif hg_arch in {"resizeconv", "resize-conv", "nearestconv", "nearest-conv"}:
            hg_cls = Hallucination_Generator_ResizeConv
            self.hg = hg_cls(
                in_nc=in_nc, out_nc=out_nc, nf=hg_nf, sf=2
            )
            self.hg_arch = "resizeconv"
        elif hg_arch in {"fusedbn", "fused-bn", "qfriendly", "quant-friendly"}:
            hg_cls = Hallucination_Generator_FusedBN
            self.hg = hg_cls(
                in_nc=in_nc, out_nc=out_nc, nf=hg_nf, sf=2
            )
            self.hg_arch = "fusedbn"
        else:
            hg_cls = Hallucination_Generator
            self.hg = hg_cls(
                in_nc=in_nc, out_nc=out_nc, nf=hg_nf, sf=2
            )
            self.hg_arch = "pixelshuffle"
        self._mask_r = mask_r

    @staticmethod
    def _make_mask(img, r=0.75, thresh=0.1):
        # img: (N,3,H,W) in [0,1]
        m = img.max(dim=1, keepdim=True).values
        m = (m - r) / (1.0 - r)
        m = m.clamp(0.0, 1.0)
        m = (m > thresh).float()
        return m

    def forward(self, x):
        base_out, cond_out = self.base(x)
        mask = self._make_mask(base_out, r=self._mask_r)

        # HG path downsamples 5x by 2 (requires H/W divisible by 32).
        # Pad reflect to the next multiple of 32 to avoid shape mismatch in cat.
        _, _, h, w = base_out.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h or pad_w:
            base_pad = torch.nn.functional.pad(
                base_out, (0, pad_w, 0, pad_h), mode="reflect"
            )
            mask_pad = torch.nn.functional.pad(
                mask, (0, pad_w, 0, pad_h), mode="reflect"
            )
            hg_out = self.hg((base_pad, mask_pad))
            hg_out = hg_out[:, :, :h, :w]
        else:
            hg_out = self.hg((base_out, mask))

        return hg_out, cond_out
