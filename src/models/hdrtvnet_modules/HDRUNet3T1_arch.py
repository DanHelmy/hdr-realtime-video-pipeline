import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from . import arch_util


class HDRUNet3T1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu', weighting_network=True):
        super(HDRUNet3T1, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.SFT_layer1 = arch_util.SFTLayer(in_nc=nf//2, out_nc=nf, nf=nf//2)
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(nf, nf, 3, 2, 1)

        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 4)
        self.recon_trunk4 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk5 = arch_util.make_layer(basic_block, 1)


        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv3 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer(in_nc=nf//2, out_nc=nf, nf=nf//2)
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc = 3
        cond_nf = 64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet4 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 3, 2, 1))

        self.weighting_network = weighting_network
        if weighting_network:
            self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, nf, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, nf, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, out_nc, 1),
                                          )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Runtime may flip this before the first torch.compile trace for
        # GUI-aligned presets. Keep safe alignment as the default.
        self.assume_aligned_shapes = False

    @staticmethod
    def _align_to(x, ref):
        """Center-crop/pad x so spatial size matches ref (H, W)."""
        xh, xw = x.shape[-2:]
        rh, rw = ref.shape[-2:]

        # Crop when x is larger.
        if xh > rh:
            dh = xh - rh
            top = dh // 2
            x = x[..., top:top + rh, :]
        if xw > rw:
            dw = xw - rw
            left = dw // 2
            x = x[..., :, left:left + rw]

        # Pad when x is smaller.
        xh, xw = x.shape[-2:]
        ph = rh - xh
        pw = rw - xw
        if ph > 0 or pw > 0:
            pt = ph // 2
            pb = ph - pt
            pl = pw // 2
            pr = pw - pl
            x = F.pad(x, (pl, pr, pt, pb), mode='replicate')
        return x

    def _forward_assume_aligned(self, x):
        # v0.6-style fast path for GUI presets whose skip shapes are known to
        # line up exactly (1080p/720p). Avoid shape checks inside compile graph.
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0] # long skip connection

        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
        cond4 = self.CondNet4(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, cond3))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, cond4))

        out = out + fea3

        out = self.act(self.up_conv1(out)) + fea2
        out, _ = self.recon_trunk4((out, cond3))

        out = self.act(self.up_conv2(out)) + fea1
        out, _ = self.recon_trunk5((out, cond2))

        out = self.act(self.up_conv3(out)) + fea0
        out = self.SFT_layer2((out, cond1))

        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask_out + out
        return out, x[0]

    def _forward_safe_aligned(self, x):
        # x[0]: img; x[1]: cond
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0] # long skip connection

        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
        cond4 = self.CondNet4(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, cond3))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, cond4))

        out = out + fea3

        up = self.act(self.up_conv1(out))
        if up.shape[-2:] != fea2.shape[-2:]:
            up = self._align_to(up, fea2)
        out = up + fea2
        out, _ = self.recon_trunk4((out, cond3))

        up = self.act(self.up_conv2(out))
        if up.shape[-2:] != fea1.shape[-2:]:
            up = self._align_to(up, fea1)
        out = up + fea1
        out, _ = self.recon_trunk5((out, cond2))

        up = self.act(self.up_conv3(out))
        if up.shape[-2:] != fea0.shape[-2:]:
            up = self._align_to(up, fea0)
        out = up + fea0
        out = self.SFT_layer2((out, cond1))

        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        if out.shape[-2:] != mask_out.shape[-2:]:
            out = self._align_to(out, mask_out)
        out = mask_out + out
        return out, x[0]

    def forward(self, x):
        if self.assume_aligned_shapes:
            return self._forward_assume_aligned(x)
        return self._forward_safe_aligned(x)


class HDRUNet3T1CleanTrunk(HDRUNet3T1):
    """Quantization-friendly LE variant with SFT removed from residual trunks.

    The entry/exit SFT layers stay in place for color conditioning, but the
    heavy middle conv path becomes long Conv/ReLU residual regions that TensorRT
    can quantize without bouncing through dynamic scale/shift every block.
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu', weighting_network=True):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        basic_block = functools.partial(arch_util.ResBlock_noSFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 4)
        self.recon_trunk4 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk5 = arch_util.make_layer(basic_block, 1)
        self.le_arch = "cleantrunk"

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current and ".sft" in key:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class HDRUNet3T1CleanTrunkDeep(HDRUNet3T1CleanTrunk):
    """Clean-trunk variant with extra identity low-resolution residual blocks.

    The added h/8 blocks make the quantized region more compute-dense while
    preserving the initialized FP32 output. Their first conv is seeded from an
    existing trunk block, and their second conv is zeroed so each added residual
    block starts as an identity.
    """

    _EXTRA_TRUNK_SOURCES = {
        4: "recon_trunk1.0",
        5: "recon_trunk2.0",
        6: "recon_trunk4.0",
        7: "recon_trunk5.0",
    }

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=8,
        le_arch="cleantrunk_deep8",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        trunk3_depth = max(4, int(trunk3_depth))
        basic_block = functools.partial(arch_util.ResBlock_noSFT, nf=nf)
        self.recon_trunk3 = arch_util.make_layer(basic_block, trunk3_depth)
        self.trunk3_depth = trunk3_depth
        self.le_arch = le_arch

    def remap_loaded_state_dict(self, state_dict, prefix=""):
        mapped = dict(state_dict)
        depth = 4
        while f"{prefix}recon_trunk3.{depth}.conv1.weight" in mapped:
            depth += 1
        target_depth = int(getattr(self, "trunk3_depth", 8))
        for new_idx in range(depth, target_depth):
            source = self._EXTRA_TRUNK_SOURCES.get(
                new_idx,
                f"recon_trunk3.{(new_idx - 4) % 4}",
            )
            for suffix in ("weight", "bias"):
                src_key = f"{prefix}{source}.conv1.{suffix}"
                dst_key = f"{prefix}recon_trunk3.{new_idx}.conv1.{suffix}"
                if src_key in mapped and dst_key not in mapped:
                    mapped[dst_key] = mapped[src_key].clone()

                template_key = f"{prefix}{source}.conv2.{suffix}"
                dst_key = f"{prefix}recon_trunk3.{new_idx}.conv2.{suffix}"
                if template_key in mapped and dst_key not in mapped:
                    mapped[dst_key] = torch.zeros_like(mapped[template_key])
        return mapped

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = self.remap_loaded_state_dict(state_dict, prefix="")
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class HDRUNet3T1CleanTrunkWideExtra(HDRUNet3T1CleanTrunk):
    """Clean trunk with identity-initialized wide h/8 residual capacity."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        extra_blocks=4,
        wide_nf=64,
        le_arch="cleantrunk_wide64x4",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        layers = []
        for _ in range(4):
            layers.append(arch_util.ResBlock_noSFT(nf=nf))
        for _ in range(max(1, int(extra_blocks))):
            layers.append(arch_util.ResBlock_noSFTWide(nf=nf, wide_nf=wide_nf))
        self.recon_trunk3 = nn.Sequential(*layers)
        self.trunk3_extra_blocks = int(extra_blocks)
        self.trunk3_wide_nf = int(wide_nf)
        self.le_arch = le_arch

    def remap_loaded_state_dict(self, state_dict, prefix=""):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            full_key = f"{prefix}{key}"
            if full_key not in mapped:
                mapped[full_key] = value.clone()
        return mapped

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = self.remap_loaded_state_dict(state_dict, prefix="")
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class HDRUNet3T1FlatTrunk(HDRUNet3T1CleanTrunk):
    """Speed-first clean trunk with Conv/ReLU chains and macro skips only."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=8,
        wide_nf=None,
        flatten_all=False,
        le_arch="cleantrunk_flat8",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        self.trunk3_depth = int(trunk3_depth)
        self.trunk3_wide_nf = int(wide_nf or nf)
        self.flatten_all_trunks = bool(flatten_all)
        if flatten_all:
            self.recon_trunk1 = arch_util.TuplePlainConvTrunk(nf=nf, depth=2)
            self.recon_trunk2 = arch_util.TuplePlainConvTrunk(nf=nf, depth=2)
            self.recon_trunk4 = arch_util.TuplePlainConvTrunk(nf=nf, depth=2)
            self.recon_trunk5 = arch_util.TuplePlainConvTrunk(nf=nf, depth=2)
        if wide_nf and int(wide_nf) > int(nf):
            self.recon_trunk3 = arch_util.TupleWidePlainConvTrunk(
                nf=nf,
                wide_nf=int(wide_nf),
                depth=trunk3_depth,
            )
        else:
            self.recon_trunk3 = arch_util.TuplePlainConvTrunk(
                nf=nf,
                depth=trunk3_depth,
            )
        self.le_arch = le_arch

    def remap_loaded_state_dict(self, state_dict, prefix=""):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            full_key = f"{prefix}{key}"
            if full_key not in mapped:
                mapped[full_key] = value.clone()
        return mapped

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = self.remap_loaded_state_dict(state_dict, prefix="")
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class HDRUNet3T1PlainFlat(HDRUNet3T1FlatTrunk):
    """Speed-first plain-conv LE path that skips LE-side SFT conditioning."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=8,
        wide_nf=None,
        linear_skips=False,
        le_arch="plainflatall8",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
            trunk3_depth=trunk3_depth,
            wide_nf=wide_nf,
            flatten_all=True,
            le_arch=le_arch,
        )
        self.linear_skips = bool(linear_skips)
        self.SFT_layer1 = nn.Identity()
        self.SFT_layer2 = nn.Identity()
        self.cond_first = nn.Identity()
        self.CondNet1 = nn.Identity()
        self.CondNet2 = nn.Identity()
        self.CondNet3 = nn.Identity()
        self.CondNet4 = nn.Identity()
        self.le_arch = le_arch

    def _forward_plain(self, x, assume_aligned=False):
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0]

        fea0 = self.act(self.conv_first(x[0]))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, None))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, None))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, None))
        if not self.linear_skips:
            out = out + fea3

        up = self.act(self.up_conv1(out))
        if not assume_aligned and up.shape[-2:] != fea2.shape[-2:]:
            up = self._align_to(up, fea2)
        out = up if self.linear_skips else up + fea2
        out, _ = self.recon_trunk4((out, None))

        up = self.act(self.up_conv2(out))
        if not assume_aligned and up.shape[-2:] != fea1.shape[-2:]:
            up = self._align_to(up, fea1)
        out = up if self.linear_skips else up + fea1
        out, _ = self.recon_trunk5((out, None))

        up = self.act(self.up_conv3(out))
        if not assume_aligned and up.shape[-2:] != fea0.shape[-2:]:
            up = self._align_to(up, fea0)
        out = up if self.linear_skips else up + fea0

        out = self.act(self.HR_conv2(out))
        out = self.conv_last(out)
        if not assume_aligned and out.shape[-2:] != mask_out.shape[-2:]:
            out = self._align_to(out, mask_out)
        out = mask_out + out
        return out, x[0]

    def _forward_assume_aligned(self, x):
        return self._forward_plain(x, assume_aligned=True)

    def _forward_safe_aligned(self, x):
        return self._forward_plain(x, assume_aligned=False)


class HDRUNet3T1PlainBottleneck(HDRUNet3T1PlainFlat):
    """Compiler-first plain LE path with one low-resolution compute island."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=16,
        wide_nf=128,
        bottleneck_scale=8,
        le_arch="plainbottleneckh8wide128x16",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
            trunk3_depth=1,
            wide_nf=None,
            linear_skips=True,
            le_arch=le_arch,
        )
        self.recon_trunk1 = nn.Identity()
        self.recon_trunk2 = nn.Identity()
        self.recon_trunk4 = nn.Identity()
        self.recon_trunk5 = nn.Identity()
        self.bottleneck_scale = 16 if int(bottleneck_scale) >= 16 else 8
        self.trunk3_depth = int(trunk3_depth)
        self.trunk3_wide_nf = int(wide_nf or nf)
        if self.bottleneck_scale >= 16:
            self.down_conv4 = nn.Conv2d(nf, nf, 3, 2, 1)
            self.up_conv0 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        else:
            self.down_conv4 = nn.Identity()
            self.up_conv0 = nn.Identity()
        self.recon_trunk3 = arch_util.TupleWidePlainConvTrunk(
            nf=nf,
            wide_nf=self.trunk3_wide_nf,
            depth=self.trunk3_depth,
        )
        self.le_arch = le_arch

    def _forward_bottleneck(self, x):
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0]

        fea0 = self.act(self.conv_first(x[0]))
        fea0 = self.act(self.HR_conv1(fea0))
        fea1 = self.act(self.down_conv1(fea0))
        fea2 = self.act(self.down_conv2(fea1))
        fea3 = self.act(self.down_conv3(fea2))

        bottleneck = fea3
        if self.bottleneck_scale >= 16:
            bottleneck = self.act(self.down_conv4(bottleneck))
        out, _ = self.recon_trunk3((bottleneck, None))

        if self.bottleneck_scale >= 16:
            out = self.act(self.up_conv0(out))
            if out.shape[-2:] != fea3.shape[-2:]:
                out = self._align_to(out, fea3)

        out = self.act(self.up_conv1(out))
        if out.shape[-2:] != fea2.shape[-2:]:
            out = self._align_to(out, fea2)
        out = self.act(self.up_conv2(out))
        if out.shape[-2:] != fea1.shape[-2:]:
            out = self._align_to(out, fea1)
        out = self.act(self.up_conv3(out))
        if out.shape[-2:] != fea0.shape[-2:]:
            out = self._align_to(out, fea0)

        out = self.act(self.HR_conv2(out))
        out = self.conv_last(out)
        if out.shape[-2:] != mask_out.shape[-2:]:
            out = self._align_to(out, mask_out)
        out = mask_out + out
        return out, x[0]

    def _forward_assume_aligned(self, x):
        return self._forward_bottleneck(x)

    def _forward_safe_aligned(self, x):
        return self._forward_bottleneck(x)


class HDRUNet3T1PlainDirect(HDRUNet3T1PlainFlat):
    """Direct low-res residual path for TensorRT-friendly INT8 islands."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=16,
        wide_nf=128,
        bottleneck_scale=8,
        le_arch="plaindirecth8wide128x16",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
            trunk3_depth=1,
            wide_nf=None,
            linear_skips=True,
            le_arch=le_arch,
        )
        self.SFT_layer1 = nn.Identity()
        self.SFT_layer2 = nn.Identity()
        self.cond_first = nn.Identity()
        self.CondNet1 = nn.Identity()
        self.CondNet2 = nn.Identity()
        self.CondNet3 = nn.Identity()
        self.CondNet4 = nn.Identity()
        self.conv_first = nn.Identity()
        self.HR_conv1 = nn.Identity()
        self.down_conv1 = nn.Identity()
        self.down_conv2 = nn.Identity()
        self.down_conv3 = nn.Identity()
        self.up_conv1 = nn.Identity()
        self.up_conv2 = nn.Identity()
        self.up_conv3 = nn.Identity()
        self.HR_conv2 = nn.Identity()
        self.conv_last = nn.Identity()
        self.recon_trunk1 = nn.Identity()
        self.recon_trunk2 = nn.Identity()
        self.recon_trunk4 = nn.Identity()
        self.recon_trunk5 = nn.Identity()
        self.bottleneck_scale = max(2, int(bottleneck_scale))
        self.trunk3_depth = int(trunk3_depth)
        self.trunk3_wide_nf = int(wide_nf or nf)
        self.low_in = nn.Conv2d(in_nc, self.trunk3_wide_nf, 1, 1, 0)
        layers = []
        for _ in range(max(1, self.trunk3_depth)):
            layers.append(nn.Conv2d(self.trunk3_wide_nf, self.trunk3_wide_nf, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
        self.recon_trunk3 = nn.Sequential(*layers)
        self.low_out = nn.Conv2d(
            self.trunk3_wide_nf,
            out_nc * self.bottleneck_scale * self.bottleneck_scale,
            1,
            1,
            0,
        )
        self.up = nn.PixelShuffle(self.bottleneck_scale)
        arch_util.initialize_weights([self.low_in, self.recon_trunk3], 0.1)
        nn.init.zeros_(self.low_out.weight)
        if self.low_out.bias is not None:
            nn.init.zeros_(self.low_out.bias)
        self.le_arch = le_arch

    def remap_loaded_state_dict(self, state_dict, prefix=""):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            full_key = f"{prefix}{key}"
            if full_key not in mapped:
                mapped[full_key] = value.clone()
        return mapped

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = self.remap_loaded_state_dict(state_dict, prefix="")
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    def _forward_direct(self, x):
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0]

        low = F.avg_pool2d(x[0], self.bottleneck_scale, self.bottleneck_scale)
        out = self.act(self.low_in(low))
        out = self.recon_trunk3(out)
        out = self.low_out(out)
        out = self.up(out)
        if out.shape[-2:] != mask_out.shape[-2:]:
            out = self._align_to(out, mask_out)
        out = mask_out + out
        return out, x[0]

    def _forward_assume_aligned(self, x):
        return self._forward_direct(x)

    def _forward_safe_aligned(self, x):
        return self._forward_direct(x)


class HDRUNet3T1CondDirect(HDRUNet3T1PlainDirect):
    """Direct low-res residual path that keeps the spatial condition signal."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        trunk3_depth=16,
        wide_nf=128,
        bottleneck_scale=16,
        le_arch="conddirecth16wide128x16",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
            trunk3_depth=trunk3_depth,
            wide_nf=wide_nf,
            bottleneck_scale=bottleneck_scale,
            le_arch=le_arch,
        )
        self.low_in = nn.Conv2d(in_nc + 3, self.trunk3_wide_nf, 1, 1, 0)
        arch_util.initialize_weights([self.low_in], 0.1)
        self.le_arch = le_arch

    def _forward_direct(self, x):
        img = x[0]
        if self.weighting_network:
            mask = self.mask_est(img)
            mask_out = mask * img
        else:
            mask_out = img

        cond = x[1] if isinstance(x, (tuple, list)) and len(x) > 1 else img
        cond = cond.to(device=img.device, dtype=img.dtype)
        low_img = F.avg_pool2d(img, self.bottleneck_scale, self.bottleneck_scale)
        cond_stride = max(1, self.bottleneck_scale // 4)
        low_cond = F.avg_pool2d(cond, cond_stride, cond_stride)
        if low_cond.shape[-2:] != low_img.shape[-2:]:
            low_cond = self._align_to(low_cond, low_img)
        low = torch.cat((low_img, low_cond), dim=1)

        out = self.act(self.low_in(low))
        out = self.recon_trunk3(out)
        out = self.low_out(out)
        out = self.up(out)
        if out.shape[-2:] != mask_out.shape[-2:]:
            out = self._align_to(out, mask_out)
        out = mask_out + out
        return out, img


class HDRUNet3T1BottleneckHeavy(HDRUNet3T1):
    """Quantization-first LE variant with residual work moved to h/8.

    The original HR topology has residual blocks at h/2, h/4, h/8, then h/4
    and h/2 again.  This variant keeps the same down/up scaffold and SFT
    entry/exit conditioning, but makes the residual compute one long clean
    bottleneck trunk. TensorRT can then quantize a larger low-resolution
    Conv/ReLU region without paying Q/DQ costs on wider feature maps.
    """

    _EXTRA_TRUNK_SOURCES = {
        4: "recon_trunk1.0",
        5: "recon_trunk2.0",
        6: "recon_trunk4.0",
        7: "recon_trunk5.0",
    }

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu', weighting_network=True):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        empty_block = functools.partial(arch_util.ResBlock_noSFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(empty_block, 0)
        self.recon_trunk2 = arch_util.make_layer(empty_block, 0)
        self.recon_trunk4 = arch_util.make_layer(empty_block, 0)
        self.recon_trunk5 = arch_util.make_layer(empty_block, 0)
        bottleneck_block = functools.partial(arch_util.ResBlock_noSFT, nf=nf)
        self.recon_trunk3 = arch_util.make_layer(bottleneck_block, 8)
        self.le_arch = "bottleneck_heavy"

    @classmethod
    def remap_loaded_state_dict(cls, state_dict, prefix=""):
        mapped = dict(state_dict)
        for new_idx, source in cls._EXTRA_TRUNK_SOURCES.items():
            for conv_name in ("conv1", "conv2"):
                for suffix in ("weight", "bias"):
                    src_key = f"{prefix}{source}.{conv_name}.{suffix}"
                    dst_key = f"{prefix}recon_trunk3.{new_idx}.{conv_name}.{suffix}"
                    if src_key in mapped and dst_key not in mapped:
                        mapped[dst_key] = mapped[src_key]
        return mapped

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = self.remap_loaded_state_dict(state_dict, prefix="")
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class HDRUNet3T1SelectiveSFT(HDRUNet3T1):
    """Quantization-friendly LE variant with SFT kept only where requested.

    Keeping SFT at lower spatial scales preserves much of the original HR
    modulation behavior while avoiding repeated high-resolution scale/shift
    islands that fragment TensorRT INT8 regions.
    """

    _TRUNK_DEPTHS = {
        "recon_trunk1": 1,
        "recon_trunk2": 1,
        "recon_trunk3": 4,
        "recon_trunk4": 1,
        "recon_trunk5": 1,
    }

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        act_type='relu',
        weighting_network=True,
        sft_trunks=("recon_trunk3",),
        le_arch="selective_sft",
    ):
        super().__init__(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            act_type=act_type,
            weighting_network=weighting_network,
        )
        self.sft_trunks = tuple(sft_trunks)
        for trunk_name, depth in self._TRUNK_DEPTHS.items():
            block_cls = (
                arch_util.ResBlock_with_SFT
                if trunk_name in self.sft_trunks
                else arch_util.ResBlock_noSFT
            )
            block = functools.partial(block_cls, nf=nf)
            setattr(self, trunk_name, arch_util.make_layer(block, depth))
        self.le_arch = le_arch

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key in list(mapped.keys()):
            if key not in current and ".sft" in key:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)
