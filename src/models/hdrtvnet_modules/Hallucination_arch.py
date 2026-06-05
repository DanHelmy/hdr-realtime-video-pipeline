import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(net, init_type="normal"):
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(f"initialization method [{init_type}] is not implemented")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif "Linear" in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, norm=True, act=True):
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding)
    n = nn.BatchNorm2d(out_nc, affine=True) if norm else None
    a = nn.ReLU(inplace=True) if act else None
    layers = [c, n, a]
    return nn.Sequential(*[x for x in layers if x is not None])


def up_block(in_nc, out_nc, sf=2, kernel_size=3, stride=1):
    c = nn.Conv2d(in_nc, out_nc * sf ** 2, kernel_size=kernel_size, stride=stride, padding=1)
    s = nn.PixelShuffle(sf)
    a = nn.ReLU(inplace=True)
    return nn.Sequential(c, s, a)


def resize_conv_block(in_nc, out_nc, sf=2, kernel_size=3, stride=1):
    u = nn.Upsample(scale_factor=sf, mode="nearest")
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=1)
    a = nn.ReLU(inplace=True)
    return nn.Sequential(u, c, a)


def fused_conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, act=True):
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding)
    a = nn.ReLU(inplace=True) if act else None
    layers = [c, a]
    return nn.Sequential(*[x for x in layers if x is not None])


class Hallucination_Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, sf=2):
        super().__init__()

        self.maxpool = nn.MaxPool2d(sf)

        self.conv1 = conv_block(in_nc, nf)
        self.conv2 = conv_block(nf, 2 * nf)

        self.conv3_1 = conv_block(2 * nf, 4 * nf)
        self.conv3_2 = conv_block(4 * nf, 4 * nf)

        self.conv4_1 = conv_block(4 * nf, 8 * nf)
        self.conv4_2 = conv_block(8 * nf, 8 * nf)

        self.conv5_1 = conv_block(8 * nf, 8 * nf)
        self.conv5_2 = conv_block(8 * nf, 8 * nf)

        self.conv_code1 = conv_block(8 * nf, 8 * nf)
        self.conv_code2 = conv_block(8 * nf, 8 * nf)

        self.Up_conv1 = up_block(8 * nf, 8 * nf, sf=2)
        self.conv6 = nn.Conv2d(16 * nf, 8 * nf, 1, 1)

        self.Up_conv2 = up_block(8 * nf, 8 * nf, sf=2)
        self.conv7 = nn.Conv2d(16 * nf, 4 * nf, 1, 1)

        self.Up_conv3 = up_block(4 * nf, 4 * nf, sf=2)
        self.conv8 = nn.Conv2d(8 * nf, 2 * nf, 1, 1)

        self.Up_conv4 = up_block(2 * nf, 2 * nf, sf=2)
        self.conv9 = nn.Conv2d(4 * nf, nf, 1, 1)

        self.Up_conv5 = up_block(nf, nf, sf=2)
        self.conv10 = nn.Conv2d(2 * nf, out_nc, 1, 1)

        self.conv_last = nn.Conv2d(2 * out_nc, out_nc, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        img = x[0]
        mask = x[1]

        conv1_out = self.conv1(img)

        conv2_out = self.maxpool(conv1_out)
        conv2_out = self.conv2(conv2_out)

        conv3_out = self.maxpool(self.conv3_1(conv2_out))
        conv3_out = self.conv3_2(conv3_out)

        conv4_out = self.maxpool(self.conv4_1(conv3_out))
        conv4_out = self.conv4_2(conv4_out)

        conv5_out = self.maxpool(self.conv5_1(conv4_out))
        conv5_out = self.conv5_2(conv5_out)

        conv_code = self.maxpool(self.conv_code1(conv5_out))
        conv_code = self.conv_code2(conv_code)

        conv6_out = torch.cat((self.Up_conv1(conv_code), conv5_out), 1)
        conv6_out = self.conv6(conv6_out)

        conv7_out = torch.cat((self.Up_conv2(conv6_out), conv4_out), 1)
        conv7_out = self.conv7(conv7_out)

        conv8_out = torch.cat((self.Up_conv3(conv7_out), conv3_out), 1)
        conv8_out = self.conv8(conv8_out)

        conv9_out = torch.cat((self.Up_conv4(conv8_out), conv2_out), 1)
        conv9_out = self.conv9(conv9_out)

        conv10_out = torch.cat((self.Up_conv5(conv9_out), conv1_out), 1)
        conv10_out = self.conv10(conv10_out)

        out = torch.cat((conv10_out, img), 1)
        out = self.conv_last(out)

        out = mask * out + img
        return out


class Hallucination_Generator_ResizeConv(Hallucination_Generator):
    """HG variant that avoids PixelShuffle transpose kernels for TensorRT tests."""

    def __init__(self, in_nc=3, out_nc=3, nf=64, sf=2):
        super().__init__(in_nc=in_nc, out_nc=out_nc, nf=nf, sf=sf)
        self.Up_conv1 = resize_conv_block(8 * nf, 8 * nf, sf=sf)
        self.Up_conv2 = resize_conv_block(8 * nf, 8 * nf, sf=sf)
        self.Up_conv3 = resize_conv_block(4 * nf, 4 * nf, sf=sf)
        self.Up_conv4 = resize_conv_block(2 * nf, 2 * nf, sf=sf)
        self.Up_conv5 = resize_conv_block(nf, nf, sf=sf)

        for module in (
            self.Up_conv1[1],
            self.Up_conv2[1],
            self.Up_conv3[1],
            self.Up_conv4[1],
            self.Up_conv5[1],
        ):
            init_weights(module, init_type="kaiming")

    @staticmethod
    def _pixelshuffle_to_resizeconv_weight(weight, target_shape):
        if not hasattr(weight, "shape") or len(weight.shape) != 4:
            return weight
        out_c, in_c, kh, kw = target_shape
        if tuple(weight.shape[1:]) != (in_c, kh, kw):
            return weight
        if int(weight.shape[0]) != int(out_c) * 4:
            return weight
        return weight.reshape(out_c, 4, in_c, kh, kw).mean(dim=1).contiguous()

    @staticmethod
    def _pixelshuffle_to_resizeconv_bias(bias, target_shape):
        if bias is None or not hasattr(bias, "shape"):
            return bias
        out_c = int(target_shape[0])
        if int(bias.shape[0]) != out_c * 4:
            return bias
        return bias.reshape(out_c, 4).mean(dim=1).contiguous()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = dict(state_dict)
        for name in ("Up_conv1", "Up_conv2", "Up_conv3", "Up_conv4", "Up_conv5"):
            old_w = f"{name}.0.weight"
            old_b = f"{name}.0.bias"
            new_w = f"{name}.1.weight"
            new_b = f"{name}.1.bias"
            if old_w in mapped and new_w not in mapped:
                mapped[new_w] = self._pixelshuffle_to_resizeconv_weight(
                    mapped[old_w],
                    self.state_dict()[new_w].shape,
                )
                mapped.pop(old_w, None)
            if old_b in mapped and new_b not in mapped:
                mapped[new_b] = self._pixelshuffle_to_resizeconv_bias(
                    mapped[old_b],
                    self.state_dict()[new_b].shape,
                )
                mapped.pop(old_b, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class Hallucination_Generator_FusedBN(Hallucination_Generator):
    """HG variant with eval-time BatchNorm folded into adjacent conv weights."""

    _BN_BLOCKS = (
        "conv1",
        "conv2",
        "conv3_1",
        "conv3_2",
        "conv4_1",
        "conv4_2",
        "conv5_1",
        "conv5_2",
        "conv_code1",
        "conv_code2",
    )

    def __init__(self, in_nc=3, out_nc=3, nf=64, sf=2):
        super().__init__(in_nc=in_nc, out_nc=out_nc, nf=nf, sf=sf)

        self.conv1 = fused_conv_block(in_nc, nf)
        self.conv2 = fused_conv_block(nf, 2 * nf)

        self.conv3_1 = fused_conv_block(2 * nf, 4 * nf)
        self.conv3_2 = fused_conv_block(4 * nf, 4 * nf)

        self.conv4_1 = fused_conv_block(4 * nf, 8 * nf)
        self.conv4_2 = fused_conv_block(8 * nf, 8 * nf)

        self.conv5_1 = fused_conv_block(8 * nf, 8 * nf)
        self.conv5_2 = fused_conv_block(8 * nf, 8 * nf)

        self.conv_code1 = fused_conv_block(8 * nf, 8 * nf)
        self.conv_code2 = fused_conv_block(8 * nf, 8 * nf)

        for block_name in self._BN_BLOCKS:
            init_weights(getattr(self, block_name)[0], init_type="kaiming")

    @staticmethod
    def _fold_bn_into_conv(mapped, block_name, eps=1e-5):
        conv_w_key = f"{block_name}.0.weight"
        conv_b_key = f"{block_name}.0.bias"
        bn_w_key = f"{block_name}.1.weight"
        bn_b_key = f"{block_name}.1.bias"
        bn_mean_key = f"{block_name}.1.running_mean"
        bn_var_key = f"{block_name}.1.running_var"
        if not all(k in mapped for k in (conv_w_key, bn_w_key, bn_b_key, bn_mean_key, bn_var_key)):
            return

        conv_w = mapped[conv_w_key].float()
        if conv_b_key in mapped and mapped[conv_b_key] is not None:
            conv_b = mapped[conv_b_key].float()
        else:
            conv_b = conv_w.new_zeros(conv_w.shape[0])
        bn_w = mapped[bn_w_key].float()
        bn_b = mapped[bn_b_key].float()
        running_mean = mapped[bn_mean_key].float()
        running_var = mapped[bn_var_key].float()

        inv_std = torch.rsqrt(running_var + eps)
        scale = bn_w * inv_std
        mapped[conv_w_key] = (conv_w * scale.reshape(-1, 1, 1, 1)).to(mapped[conv_w_key].dtype)
        mapped[conv_b_key] = ((conv_b - running_mean) * scale + bn_b).to(mapped[conv_w_key].dtype)

        for suffix in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            mapped.pop(f"{block_name}.1.{suffix}", None)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = dict(state_dict)
        for block_name in self._BN_BLOCKS:
            self._fold_bn_into_conv(mapped, block_name)
        return super().load_state_dict(mapped, strict=strict, assign=assign)


class Hallucination_Generator_Direct(nn.Module):
    """Compiler-first HG path with one low-resolution masked residual island."""

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        sf=2,
        trunk_depth=8,
        wide_nf=64,
        bottleneck_scale=8,
    ):
        super().__init__()
        self.bottleneck_scale = max(2, int(bottleneck_scale))
        self.trunk_depth = int(trunk_depth)
        self.trunk_wide_nf = int(wide_nf or nf)
        self.low_in = nn.Conv2d(in_nc + 1, self.trunk_wide_nf, 1, 1, 0)
        layers = []
        for _ in range(max(1, self.trunk_depth)):
            layers.append(nn.Conv2d(self.trunk_wide_nf, self.trunk_wide_nf, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.low_out = nn.Conv2d(
            self.trunk_wide_nf,
            out_nc * self.bottleneck_scale * self.bottleneck_scale,
            1,
            1,
            0,
        )
        self.up = nn.PixelShuffle(self.bottleneck_scale)
        for module in (self.low_in, self.trunk):
            init_weights(module, init_type="kaiming")
        nn.init.zeros_(self.low_out.weight)
        if self.low_out.bias is not None:
            nn.init.zeros_(self.low_out.bias)

    @staticmethod
    def _align_to(x, ref):
        xh, xw = x.shape[-2:]
        rh, rw = ref.shape[-2:]
        if xh > rh:
            top = (xh - rh) // 2
            x = x[..., top:top + rh, :]
        if xw > rw:
            left = (xw - rw) // 2
            x = x[..., :, left:left + rw]
        xh, xw = x.shape[-2:]
        ph = rh - xh
        pw = rw - xw
        if ph > 0 or pw > 0:
            pt = ph // 2
            pb = ph - pt
            pl = pw // 2
            pr = pw - pl
            x = F.pad(x, (pl, pr, pt, pb), mode="replicate")
        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            if key not in mapped:
                mapped[key] = value.clone()
        for key in list(mapped.keys()):
            if key not in current:
                mapped.pop(key, None)
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    def forward(self, x):
        img = x[0]
        mask = x[1].to(device=img.device, dtype=img.dtype)
        cond = torch.cat((img, mask), 1)
        low = F.avg_pool2d(cond, self.bottleneck_scale, self.bottleneck_scale)
        out = F.relu(self.low_in(low), inplace=True)
        out = self.trunk(out)
        out = self.low_out(out)
        out = self.up(out)
        if out.shape[-2:] != img.shape[-2:]:
            out = self._align_to(out, img)
        return mask * out + img
