import torch
import torch.nn as nn

from .Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from .Hallucination_arch import Hallucination_Generator


class HG_Composite(nn.Module):
    """Composite HDRTVNet++ pipeline: AGCM+LE followed by HG refinement."""

    def __init__(self, classifier="color_condition", cond_c=6,
                 in_nc=3, out_nc=3, nf=32, act_type="relu",
                 weighting_network=False, hg_nf=64, mask_r=0.75):
        super().__init__()
        self.base = Ensemble_AGCM_LE(
            classifier=classifier,
            cond_c=cond_c,
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        self.hg = Hallucination_Generator(
            in_nc=in_nc, out_nc=out_nc, nf=hg_nf, sf=2
        )
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
