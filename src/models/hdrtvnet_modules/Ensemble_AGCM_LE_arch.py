import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Condition_arch import ConditionNet, is_plain_agcm_classifier, remap_condition_state_dict
from .HDRUNet3T1_arch import (
    HDRUNet3T1,
    HDRUNet3T1BottleneckHeavy,
    HDRUNet3T1CleanTrunk,
    HDRUNet3T1CleanTrunkDeep,
    HDRUNet3T1CleanTrunkWideExtra,
    HDRUNet3T1CondDirect,
    HDRUNet3T1CondGatedDirect,
    HDRUNet3T1FlatTrunk,
    HDRUNet3T1PlainBottleneck,
    HDRUNet3T1PlainDirect,
    HDRUNet3T1PlainFlat,
    HDRUNet3T1SelectiveSFT,
)


def _parse_post_correction(spec):
    spec = str(spec or "").strip().lower()
    if not spec or spec in {"none", "off", "0", "false"}:
        return None
    canonical = spec.replace("-", "").replace("_", "")
    stack_match = re.fullmatch(
        r"(?:post)?global(?:color)?(?:correct|correction|corr)?wide([0-9]+)x([0-9]+)"
        r"(?:post)?(?:color)?(?:correct|correction|corr)h(4|8|16)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if stack_match:
        global_width = int(stack_match.group(1))
        global_depth = int(stack_match.group(2))
        scale = int(stack_match.group(3))
        width = int(stack_match.group(4))
        depth = int(stack_match.group(5))
        return "global_spatial", global_width, global_depth, scale, width, depth
    stack_residual_match = re.fullmatch(
        r"(?:post)?global(?:color)?(?:correct|correction|corr)?wide([0-9]+)x([0-9]+)"
        r"(?:post)?res(?:idual)?h(2|4|8|16)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if stack_residual_match:
        global_width = int(stack_residual_match.group(1))
        global_depth = int(stack_residual_match.group(2))
        scale = int(stack_residual_match.group(3))
        width = int(stack_residual_match.group(4))
        depth = int(stack_residual_match.group(5))
        return "global_residual", global_width, global_depth, scale, width, depth
    residual_match = re.fullmatch(
        r"(?:post)?res(?:idual)?h(2|4|8|16)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if residual_match:
        scale = int(residual_match.group(1))
        width = int(residual_match.group(2))
        depth = int(residual_match.group(3))
        return "residual", scale, width, depth
    global_match = re.fullmatch(
        r"(?:post)?global(?:color)?(?:correct|correction|corr)?wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if global_match:
        width = int(global_match.group(1))
        depth = int(global_match.group(2))
        return "global", 1, width, depth
    affine_match = re.fullmatch(
        r"(?:post)?affineh(4|8|16)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if affine_match:
        scale = int(affine_match.group(1))
        width = int(affine_match.group(2))
        depth = int(affine_match.group(3))
        return "affine", scale, width, depth
    match = re.fullmatch(
        r"(?:post)?(?:color)?(?:correct|correction|corr)h(4|8|16)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if not match:
        raise ValueError(f"Unsupported post_correction '{spec}'")
    scale = int(match.group(1))
    width = int(match.group(2))
    depth = int(match.group(3))
    return "spatial", scale, width, depth


class SpatialColorCorrection(nn.Module):
    """Small identity-initialized output color corrector.

    It predicts a low-resolution 3x3 color matrix plus RGB shift from the
    original SDR input and the HR output, then applies it at full resolution.
    The zero-initialized final layer makes the module an exact identity until
    trained.
    """

    def __init__(self, scale=8, width=32, depth=3, limit=0.25):
        super().__init__()
        self.scale = int(scale)
        self.width = int(width)
        self.depth = int(depth)
        self.limit = float(limit)
        layers = [nn.Conv2d(6, self.width, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, self.depth)):
            layers.extend([
                nn.Conv2d(self.width, self.width, 3, 1, 1),
                nn.ReLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(self.width, 12, 1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, img, hdr):
        low = F.avg_pool2d(torch.cat((img, hdr), dim=1), self.scale, self.scale)
        params_low = self.out(self.trunk(low))

        def _full_param(value):
            if value.shape[-2:] != hdr.shape[-2:]:
                value = F.interpolate(
                    value,
                    size=hdr.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return value.tanh() * self.limit

        row0 = _full_param(params_low[:, 0:3])
        row1 = _full_param(params_low[:, 3:6])
        row2 = _full_param(params_low[:, 6:9])
        shift_r = _full_param(params_low[:, 9:10])
        shift_g = _full_param(params_low[:, 10:11])
        shift_b = _full_param(params_low[:, 11:12])
        r = hdr[:, 0:1]
        g = hdr[:, 1:2]
        b = hdr[:, 2:3]
        out_r = r * (row0[:, 0:1] + 1.0) + g * row0[:, 1:2] + b * row0[:, 2:3] + shift_r
        out_g = r * row1[:, 0:1] + g * (row1[:, 1:2] + 1.0) + b * row1[:, 2:3] + shift_g
        out_b = r * row2[:, 0:1] + g * row2[:, 1:2] + b * (row2[:, 2:3] + 1.0) + shift_b
        return torch.cat((out_r, out_g, out_b), dim=1)


class SpatialAffineCorrection(nn.Module):
    """Spatial per-channel scale/shift correction."""

    def __init__(self, scale=8, width=32, depth=3, limit=0.25):
        super().__init__()
        self.scale = int(scale)
        self.width = int(width)
        self.depth = int(depth)
        self.limit = float(limit)
        layers = [nn.Conv2d(6, self.width, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, self.depth)):
            layers.extend([
                nn.Conv2d(self.width, self.width, 3, 1, 1),
                nn.ReLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(self.width, 6, 1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, img, hdr):
        low = F.avg_pool2d(torch.cat((img, hdr), dim=1), self.scale, self.scale)
        params = self.out(self.trunk(low))
        if params.shape[-2:] != hdr.shape[-2:]:
            params = F.interpolate(
                params,
                size=hdr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        params = params.tanh() * self.limit
        scale = params[:, :3]
        shift = params[:, 3:]
        return hdr * (1.0 + scale) + shift


class SpatialResidualCorrection(nn.Module):
    """Low-resolution residual tail for local detail/color correction."""

    def __init__(self, scale=4, width=64, depth=6, limit=0.20):
        super().__init__()
        self.scale = int(scale)
        self.width = int(width)
        self.depth = int(depth)
        self.limit = float(limit)
        layers = [nn.Conv2d(6, self.width, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, self.depth)):
            layers.extend([
                nn.Conv2d(self.width, self.width, 3, 1, 1),
                nn.ReLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(self.width, 3 * self.scale * self.scale, 1)
        self.up = nn.PixelShuffle(self.scale)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, img, hdr):
        low = F.avg_pool2d(torch.cat((img, hdr), dim=1), self.scale, self.scale)
        delta = self.up(self.out(self.trunk(low))).tanh() * self.limit
        if delta.shape[-2:] != hdr.shape[-2:]:
            delta = F.interpolate(
                delta,
                size=hdr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return hdr + delta


class GlobalColorCorrection(nn.Module):
    """Per-frame color matrix correction generated from global image context."""

    def __init__(self, width=48, depth=2, limit=0.25):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.limit = float(limit)
        layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(6, self.width, 1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, self.depth - 1)):
            layers.extend([
                nn.Conv2d(self.width, self.width, 1),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(self.width, 12, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, img, hdr):
        params = self.net(torch.cat((img, hdr), dim=1)).tanh() * self.limit
        delta = params[:, :9]
        shift = params[:, 9:]
        r = hdr[:, 0:1]
        g = hdr[:, 1:2]
        b = hdr[:, 2:3]
        out_r = r * (delta[:, 0:1] + 1.0) + g * delta[:, 1:2] + b * delta[:, 2:3] + shift[:, 0:1]
        out_g = r * delta[:, 3:4] + g * (delta[:, 4:5] + 1.0) + b * delta[:, 5:6] + shift[:, 1:2]
        out_b = r * delta[:, 6:7] + g * delta[:, 7:8] + b * (delta[:, 8:9] + 1.0) + shift[:, 2:3]
        return torch.cat((out_r, out_g, out_b), dim=1)


class GlobalThenSpatialCorrection(nn.Module):
    """Existing global correction followed by an identity spatial corrector."""

    def __init__(
        self,
        global_width=48,
        global_depth=2,
        scale=8,
        width=48,
        depth=3,
        limit=0.25,
    ):
        super().__init__()
        self.net = GlobalColorCorrection(
            width=global_width,
            depth=global_depth,
            limit=limit,
        ).net
        self.spatial = SpatialColorCorrection(
            scale=scale,
            width=width,
            depth=depth,
            limit=limit,
        )

    def _global(self, img, hdr):
        params = self.net(torch.cat((img, hdr), dim=1)).tanh() * 0.25
        delta = params[:, :9]
        shift = params[:, 9:]
        r = hdr[:, 0:1]
        g = hdr[:, 1:2]
        b = hdr[:, 2:3]
        out_r = r * (delta[:, 0:1] + 1.0) + g * delta[:, 1:2] + b * delta[:, 2:3] + shift[:, 0:1]
        out_g = r * delta[:, 3:4] + g * (delta[:, 4:5] + 1.0) + b * delta[:, 5:6] + shift[:, 1:2]
        out_b = r * delta[:, 6:7] + g * delta[:, 7:8] + b * (delta[:, 8:9] + 1.0) + shift[:, 2:3]
        return torch.cat((out_r, out_g, out_b), dim=1)

    def forward(self, img, hdr):
        return self.spatial(img, self._global(img, hdr))


class GlobalThenResidualCorrection(GlobalThenSpatialCorrection):
    """Existing global correction followed by an identity residual tail."""

    def __init__(
        self,
        global_width=48,
        global_depth=2,
        scale=4,
        width=64,
        depth=6,
        limit=0.20,
    ):
        nn.Module.__init__(self)
        self.net = GlobalColorCorrection(
            width=global_width,
            depth=global_depth,
        ).net
        self.residual = SpatialResidualCorrection(
            scale=scale,
            width=width,
            depth=depth,
            limit=limit,
        )

    def forward(self, img, hdr):
        return self.residual(img, self._global(img, hdr))


class Ensemble_AGCM_LE(nn.Module):
    def __init__(self, classifier='color_condition', cond_c=6, in_nc=3, out_nc=3, nf=32, act_type='relu', weighting_network=False, le_arch=None, post_correction=None):
        super(Ensemble_AGCM_LE, self).__init__()
        self.AGCM = ConditionNet(classifier=classifier, cond_c=cond_c)
        post_correction = (
            post_correction
            if post_correction is not None
            else os.environ.get("HDRTVNET_POST_CORRECTION", "")
        )
        post_cfg = _parse_post_correction(post_correction)
        if post_cfg is None:
            self.post_correction_name = ""
            self.post_correction = nn.Identity()
        else:
            mode = post_cfg[0]
            if mode == "global":
                _, scale, width, depth = post_cfg
                self.post_correction_name = f"postglobalwide{width}x{depth}"
                self.post_correction = GlobalColorCorrection(
                    width=width,
                    depth=depth,
                )
            elif mode == "global_spatial":
                _, global_width, global_depth, scale, width, depth = post_cfg
                self.post_correction_name = (
                    f"postglobalwide{global_width}x{global_depth}"
                    f"corrh{scale}wide{width}x{depth}"
                )
                self.post_correction = GlobalThenSpatialCorrection(
                    global_width=global_width,
                    global_depth=global_depth,
                    scale=scale,
                    width=width,
                    depth=depth,
                )
            elif mode == "global_residual":
                _, global_width, global_depth, scale, width, depth = post_cfg
                self.post_correction_name = (
                    f"postglobalwide{global_width}x{global_depth}"
                    f"resh{scale}wide{width}x{depth}"
                )
                self.post_correction = GlobalThenResidualCorrection(
                    global_width=global_width,
                    global_depth=global_depth,
                    scale=scale,
                    width=width,
                    depth=depth,
                )
            elif mode == "affine":
                _, scale, width, depth = post_cfg
                self.post_correction_name = f"postaffineh{scale}wide{width}x{depth}"
                self.post_correction = SpatialAffineCorrection(
                    scale=scale,
                    width=width,
                    depth=depth,
                )
            elif mode == "residual":
                _, scale, width, depth = post_cfg
                self.post_correction_name = f"postresh{scale}wide{width}x{depth}"
                self.post_correction = SpatialResidualCorrection(
                    scale=scale,
                    width=width,
                    depth=depth,
                )
            else:
                _, scale, width, depth = post_cfg
                self.post_correction_name = f"postcorrh{scale}wide{width}x{depth}"
                self.post_correction = SpatialColorCorrection(
                    scale=scale,
                    width=width,
                    depth=depth,
                )
        # fix AGCM
        # for p in self.parameters():
        #     p.requires_grad = False

        le_arch = str(le_arch or os.environ.get("HDRTVNET_LE_ARCH", "sft")).strip().lower()
        canonical_le_arch = le_arch.replace("-", "").replace("_", "")
        plain_bottleneck_match = re.fullmatch(
            r"plainbottleneckh(8|16)wide([0-9]+)x([0-9]+)",
            canonical_le_arch,
        )
        if plain_bottleneck_match:
            bottleneck_scale = int(plain_bottleneck_match.group(1))
            wide_nf = int(plain_bottleneck_match.group(2))
            trunk3_depth = int(plain_bottleneck_match.group(3))
            self.le_arch = f"plainbottleneckh{bottleneck_scale}wide{wide_nf}x{trunk3_depth}"
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=trunk3_depth,
                wide_nf=wide_nf,
                bottleneck_scale=bottleneck_scale,
                le_arch=self.le_arch,
            )
        else:
            gated_direct_match = re.fullmatch(
                r"condgatedirecth(2|4|8|16|32)wide([0-9]+)x([0-9]+)",
                canonical_le_arch,
            )
            if gated_direct_match:
                bottleneck_scale = int(gated_direct_match.group(1))
                wide_nf = int(gated_direct_match.group(2))
                trunk3_depth = int(gated_direct_match.group(3))
                self.le_arch = f"condgatedirecth{bottleneck_scale}wide{wide_nf}x{trunk3_depth}"
                self.LE = HDRUNet3T1CondGatedDirect(
                    in_nc=in_nc,
                    out_nc=out_nc,
                    nf=nf,
                    act_type=act_type,
                    weighting_network=weighting_network,
                    trunk3_depth=trunk3_depth,
                    wide_nf=wide_nf,
                    bottleneck_scale=bottleneck_scale,
                    le_arch=self.le_arch,
                )
                return
            cond_direct_match = re.fullmatch(
                r"conddirecth(2|4|8|16|32)wide([0-9]+)x([0-9]+)",
                canonical_le_arch,
            )
            if cond_direct_match:
                bottleneck_scale = int(cond_direct_match.group(1))
                wide_nf = int(cond_direct_match.group(2))
                trunk3_depth = int(cond_direct_match.group(3))
                self.le_arch = f"conddirecth{bottleneck_scale}wide{wide_nf}x{trunk3_depth}"
                self.LE = HDRUNet3T1CondDirect(
                    in_nc=in_nc,
                    out_nc=out_nc,
                    nf=nf,
                    act_type=act_type,
                    weighting_network=weighting_network,
                    trunk3_depth=trunk3_depth,
                    wide_nf=wide_nf,
                    bottleneck_scale=bottleneck_scale,
                    le_arch=self.le_arch,
                )
                return
            select_sft_match = re.fullmatch(r"(?:select|selective)?sft([1-5]+)", canonical_le_arch)
            if select_sft_match:
                requested = tuple(dict.fromkeys(select_sft_match.group(1)))
                trunk_names = tuple(f"recon_trunk{idx}" for idx in requested)
                self.le_arch = "selectsft" + "".join(requested)
                self.LE = HDRUNet3T1SelectiveSFT(
                    in_nc=in_nc,
                    out_nc=out_nc,
                    nf=nf,
                    act_type=act_type,
                    weighting_network=weighting_network,
                    sft_trunks=trunk_names,
                    le_arch=self.le_arch,
                )
                return
            plain_direct_match = re.fullmatch(
                r"plaindirecth(2|4|8|16|32)wide([0-9]+)x([0-9]+)",
                canonical_le_arch,
            )
            if plain_direct_match:
                bottleneck_scale = int(plain_direct_match.group(1))
                wide_nf = int(plain_direct_match.group(2))
                trunk3_depth = int(plain_direct_match.group(3))
                self.le_arch = f"plaindirecth{bottleneck_scale}wide{wide_nf}x{trunk3_depth}"
                self.LE = HDRUNet3T1PlainDirect(
                    in_nc=in_nc,
                    out_nc=out_nc,
                    nf=nf,
                    act_type=act_type,
                    weighting_network=weighting_network,
                    trunk3_depth=trunk3_depth,
                    wide_nf=wide_nf,
                    bottleneck_scale=bottleneck_scale,
                    le_arch=self.le_arch,
                )
            else:
                self._init_legacy_le(
                    le_arch=le_arch,
                    in_nc=in_nc,
                    out_nc=out_nc,
                    nf=nf,
                    act_type=act_type,
                    weighting_network=weighting_network,
                )

    def _init_legacy_le(self, le_arch, in_nc, out_nc, nf, act_type, weighting_network):
        if False:
            pass
        elif le_arch in {"cleantrunk", "clean-trunk"}:
            self.LE = HDRUNet3T1CleanTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
            )
            self.le_arch = "cleantrunk"
        elif le_arch in {"cleantrunk_deep8", "cleantrunk-deep8", "cleantrunk8", "deep_cleantrunk"}:
            self.LE = HDRUNet3T1CleanTrunkDeep(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                le_arch="cleantrunk_deep8",
            )
            self.le_arch = "cleantrunk_deep8"
        elif le_arch in {"cleantrunk_deep12", "cleantrunk-deep12", "cleantrunk12"}:
            self.LE = HDRUNet3T1CleanTrunkDeep(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=12,
                le_arch="cleantrunk_deep12",
            )
            self.le_arch = "cleantrunk_deep12"
        elif le_arch in {"cleantrunk_wide64x4", "cleantrunk-wide64x4", "wide64x4"}:
            self.LE = HDRUNet3T1CleanTrunkWideExtra(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                extra_blocks=4,
                wide_nf=64,
                le_arch="cleantrunk_wide64x4",
            )
            self.le_arch = "cleantrunk_wide64x4"
        elif le_arch in {"cleantrunk_wide64x8", "cleantrunk-wide64x8", "wide64x8"}:
            self.LE = HDRUNet3T1CleanTrunkWideExtra(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                extra_blocks=8,
                wide_nf=64,
                le_arch="cleantrunk_wide64x8",
            )
            self.le_arch = "cleantrunk_wide64x8"
        elif le_arch in {"cleantrunk_flat8", "cleantrunk-flat8", "flat8"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                le_arch="cleantrunk_flat8",
            )
            self.le_arch = "cleantrunk_flat8"
        elif le_arch in {"cleantrunk_flat16", "cleantrunk-flat16", "flat16"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                le_arch="cleantrunk_flat16",
            )
            self.le_arch = "cleantrunk_flat16"
        elif le_arch in {"cleantrunk_flatwide64x8", "cleantrunk-flatwide64x8", "flatwide64x8"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=64,
                le_arch="cleantrunk_flatwide64x8",
            )
            self.le_arch = "cleantrunk_flatwide64x8"
        elif le_arch in {"cleantrunk_flatall8", "cleantrunk-flatall8", "flatall8"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                flatten_all=True,
                le_arch="cleantrunk_flatall8",
            )
            self.le_arch = "cleantrunk_flatall8"
        elif le_arch in {"cleantrunk_flatallwide64x8", "cleantrunk-flatallwide64x8", "flatallwide64x8"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=64,
                flatten_all=True,
                le_arch="cleantrunk_flatallwide64x8",
            )
            self.le_arch = "cleantrunk_flatallwide64x8"
        elif le_arch in {"cleantrunk_flatallwide128x8", "cleantrunk-flatallwide128x8", "flatallwide128x8"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=128,
                flatten_all=True,
                le_arch="cleantrunk_flatallwide128x8",
            )
            self.le_arch = "cleantrunk_flatallwide128x8"
        elif le_arch in {"cleantrunk_flatallwide128x16", "cleantrunk-flatallwide128x16", "flatallwide128x16"}:
            self.LE = HDRUNet3T1FlatTrunk(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=128,
                flatten_all=True,
                le_arch="cleantrunk_flatallwide128x16",
            )
            self.le_arch = "cleantrunk_flatallwide128x16"
        elif le_arch in {"plainflatall8", "plain-flatall8", "plain_flatall8"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                le_arch="plainflatall8",
            )
            self.le_arch = "plainflatall8"
        elif le_arch in {"plainflatallwide64x8", "plain-flatallwide64x8", "plain_flatallwide64x8"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=64,
                le_arch="plainflatallwide64x8",
            )
            self.le_arch = "plainflatallwide64x8"
        elif le_arch in {"plainflatlinear8", "plain-flatlinear8", "plain_flatlinear8"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                linear_skips=True,
                le_arch="plainflatlinear8",
            )
            self.le_arch = "plainflatlinear8"
        elif le_arch in {"plainflatlinear16", "plain-flatlinear16", "plain_flatlinear16"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                linear_skips=True,
                le_arch="plainflatlinear16",
            )
            self.le_arch = "plainflatlinear16"
        elif le_arch in {"plainflatlinear32", "plain-flatlinear32", "plain_flatlinear32"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=32,
                linear_skips=True,
                le_arch="plainflatlinear32",
            )
            self.le_arch = "plainflatlinear32"
        elif le_arch in {"plainflatlinearwide64x8", "plain-flatlinearwide64x8", "plain_flatlinearwide64x8"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=64,
                linear_skips=True,
                le_arch="plainflatlinearwide64x8",
            )
            self.le_arch = "plainflatlinearwide64x8"
        elif le_arch in {"plainflatlinearwide64x16", "plain-flatlinearwide64x16", "plain_flatlinearwide64x16"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=64,
                linear_skips=True,
                le_arch="plainflatlinearwide64x16",
            )
            self.le_arch = "plainflatlinearwide64x16"
        elif le_arch in {"plainflatlinearwide128x8", "plain-flatlinearwide128x8", "plain_flatlinearwide128x8"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=128,
                linear_skips=True,
                le_arch="plainflatlinearwide128x8",
            )
            self.le_arch = "plainflatlinearwide128x8"
        elif le_arch in {"plainflatlinearwide128x16", "plain-flatlinearwide128x16", "plain_flatlinearwide128x16"}:
            self.LE = HDRUNet3T1PlainFlat(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=128,
                linear_skips=True,
                le_arch="plainflatlinearwide128x16",
            )
            self.le_arch = "plainflatlinearwide128x16"
        elif le_arch in {"plainbottleneckh8wide128x8", "plain-bottleneckh8wide128x8", "plain_bottleneckh8wide128x8"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=128,
                bottleneck_scale=8,
                le_arch="plainbottleneckh8wide128x8",
            )
            self.le_arch = "plainbottleneckh8wide128x8"
        elif le_arch in {"plainbottleneckh8wide128x16", "plain-bottleneckh8wide128x16", "plain_bottleneckh8wide128x16"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=128,
                bottleneck_scale=8,
                le_arch="plainbottleneckh8wide128x16",
            )
            self.le_arch = "plainbottleneckh8wide128x16"
        elif le_arch in {"plainbottleneckh8wide256x8", "plain-bottleneckh8wide256x8", "plain_bottleneckh8wide256x8"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=8,
                wide_nf=256,
                bottleneck_scale=8,
                le_arch="plainbottleneckh8wide256x8",
            )
            self.le_arch = "plainbottleneckh8wide256x8"
        elif le_arch in {"plainbottleneckh8wide256x16", "plain-bottleneckh8wide256x16", "plain_bottleneckh8wide256x16"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=256,
                bottleneck_scale=8,
                le_arch="plainbottleneckh8wide256x16",
            )
            self.le_arch = "plainbottleneckh8wide256x16"
        elif le_arch in {"plainbottleneckh16wide128x16", "plain-bottleneckh16wide128x16", "plain_bottleneckh16wide128x16"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=128,
                bottleneck_scale=16,
                le_arch="plainbottleneckh16wide128x16",
            )
            self.le_arch = "plainbottleneckh16wide128x16"
        elif le_arch in {"plainbottleneckh16wide256x16", "plain-bottleneckh16wide256x16", "plain_bottleneckh16wide256x16"}:
            self.LE = HDRUNet3T1PlainBottleneck(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                trunk3_depth=16,
                wide_nf=256,
                bottleneck_scale=16,
                le_arch="plainbottleneckh16wide256x16",
            )
            self.le_arch = "plainbottleneckh16wide256x16"
        elif le_arch in {"bottleneck_heavy", "bottleneck-heavy", "heavy_bottleneck", "heavy-bottleneck"}:
            self.LE = HDRUNet3T1BottleneckHeavy(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
            )
            self.le_arch = "bottleneck_heavy"
        elif le_arch in {"bottleneck_sft", "bottleneck-sft", "trunk3_sft", "trunk3-sft"}:
            self.LE = HDRUNet3T1SelectiveSFT(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                sft_trunks=("recon_trunk3",),
                le_arch="bottleneck_sft",
            )
            self.le_arch = "bottleneck_sft"
        elif le_arch in {"lowres_sft", "lowres-sft"}:
            self.LE = HDRUNet3T1SelectiveSFT(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                sft_trunks=("recon_trunk2", "recon_trunk3", "recon_trunk4"),
                le_arch="lowres_sft",
            )
            self.le_arch = "lowres_sft"
        elif le_arch in {"downpath_sft", "downpath-sft"}:
            self.LE = HDRUNet3T1SelectiveSFT(
                in_nc=in_nc,
                out_nc=out_nc,
                nf=nf,
                act_type=act_type,
                weighting_network=weighting_network,
                sft_trunks=("recon_trunk1", "recon_trunk2", "recon_trunk3"),
                le_arch="downpath_sft",
            )
            self.le_arch = "downpath_sft"
        else:
            self.LE = HDRUNet3T1(in_nc=in_nc, out_nc=out_nc, nf=nf, act_type=act_type, weighting_network=weighting_network)
            self.le_arch = "sft"

    def forward(self, x):
        condition_output, input = self.AGCM(x)
        LE_input = [condition_output, condition_output]
        # condition = image
        LE_output = self.LE(LE_input)
        output = LE_output[0]
        if self.post_correction_name:
            output = self.post_correction(input[0], output)
        return output, condition_output

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = remap_condition_state_dict(
            state_dict,
            prefix="AGCM.",
            classifier=getattr(self.AGCM, "classifier_name", "color_condition"),
        )
        if hasattr(self.LE, "remap_loaded_state_dict"):
            mapped = self.LE.remap_loaded_state_dict(mapped, prefix="LE.")
        if getattr(self, "le_arch", "") != "sft":
            current = self.state_dict()
            for key in list(mapped.keys()):
                if key not in current and key.startswith("LE."):
                    mapped.pop(key, None)
        if is_plain_agcm_classifier(getattr(self.AGCM, "classifier_name", "color_condition")):
            current = self.state_dict()
            for key, value in current.items():
                if key.startswith("AGCM.") and key not in mapped:
                    mapped[key] = value.clone()
            for key in list(mapped.keys()):
                if key.startswith("AGCM.") and key not in current:
                    mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)

if __name__=='__main__':
    net = Ensemble_AGCM_LE()
    net = net.state_dict()
    crt_net_keys = set(net.keys())
    print(crt_net_keys)
