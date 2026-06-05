import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def color_block(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


# Release Version
class Color_Condition(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


# Function ablation of Condition
# -------------------------------------------------------------------------------
class Color_Condition_woDropout(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_woDropout, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            # nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_woIN(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_woIN, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=False),
            *color_block(16, 32, normalization=False),
            *color_block(32, 64, normalization=False),
            *color_block(64, 128, normalization=False),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)
# -------------------------------------------------------------------------------


def normalize_classifier_name(classifier):
    name = str(classifier or "color_condition").strip().lower().replace("-", "_")
    spatial_mix_global_cnn_match = re.fullmatch(
        r"(?:agcm_)?spatial(?:_?mix_?global_?cnn|_?mixgcnn|_?global_?mix_?cnn)_?h?([0-9]+)?(?:_?wide_?([0-9]+))?(?:_?x_?([0-9]+))?",
        name,
    )
    if spatial_mix_global_cnn_match:
        scale = spatial_mix_global_cnn_match.group(1) or "16"
        width = spatial_mix_global_cnn_match.group(2) or "32"
        depth = spatial_mix_global_cnn_match.group(3) or "4"
        return f"agcm_spatialmixglobalcnnh{scale}wide{width}x{depth}"
    spatial_mix_global_match = re.fullmatch(
        r"(?:agcm_)?spatial(?:_?mix_?global|_?mixg|_?global_?mix)_?h?([0-9]+)?(?:_?wide_?([0-9]+))?(?:_?x_?([0-9]+))?",
        name,
    )
    if spatial_mix_global_match:
        scale = spatial_mix_global_match.group(1) or "16"
        width = spatial_mix_global_match.group(2) or "32"
        depth = spatial_mix_global_match.group(3) or "4"
        return f"agcm_spatialmixglobalh{scale}wide{width}x{depth}"
    spatial_mix_match = re.fullmatch(
        r"(?:agcm_)?spatial(?:_?mix|_?matrix|_?mat)_?h?([0-9]+)?(?:_?wide_?([0-9]+))?(?:_?x_?([0-9]+))?",
        name,
    )
    if spatial_mix_match:
        scale = spatial_mix_match.group(1) or "16"
        width = spatial_mix_match.group(2) or "32"
        depth = spatial_mix_match.group(3) or "4"
        return f"agcm_spatialmixh{scale}wide{width}x{depth}"
    spatial_match = re.fullmatch(
        r"(?:agcm_)?spatial(?:_?affine)?_?h?([0-9]+)?(?:_?wide_?([0-9]+))?(?:_?x_?([0-9]+))?",
        name,
    )
    if spatial_match:
        scale = spatial_match.group(1) or "16"
        width = spatial_match.group(2) or "32"
        depth = spatial_match.group(3) or "4"
        return f"agcm_spatialh{scale}wide{width}x{depth}"
    lite_match = re.fullmatch(r"(?:agcm_)?lite(?:_?agcm)?_?([0-9]+)?", name)
    if lite_match:
        width = lite_match.group(1) or "16"
        return f"agcm_lite{width}"
    lowrank_match = re.fullmatch(r"(?:agcm_)?lowrank_?([0-9]+)?", name)
    if lowrank_match:
        rank = lowrank_match.group(1) or "16"
        return f"agcm_lowrank{rank}"
    aliases = {
        "color": "color_condition",
        "default": "color_condition",
        "color_condition_noin": "color_condition_woin",
        "color_condition_wo_in": "color_condition_woin",
        "color_condition_woIN": "color_condition_woin",
        "color_condition_no_in": "color_condition_woin",
        "base": "agcm_plain",
        "base3": "agcm_plain",
        "plain": "agcm_plain",
        "plain3": "agcm_plain",
        "plain_agcm": "agcm_plain",
        "plain_agcm3": "agcm_plain",
        "agcm_base": "agcm_plain",
        "agcm_base3": "agcm_plain",
        "affine": "agcm_affine",
        "adaptive_affine": "agcm_affine",
    }
    return aliases.get(name, name)


def agcm_lite_width(classifier):
    match = re.fullmatch(r"agcm_lite([0-9]+)", normalize_classifier_name(classifier))
    if not match:
        return None
    return max(4, min(64, int(match.group(1))))


def agcm_lowrank_rank(classifier):
    match = re.fullmatch(r"agcm_lowrank([0-9]+)", normalize_classifier_name(classifier))
    if not match:
        return None
    return max(1, min(64, int(match.group(1))))


def agcm_spatial_config(classifier):
    match = re.fullmatch(
        r"agcm_spatial(?:mixglobalcnn|mixgcnn|mixglobal|mixg|mix)?h([0-9]+)wide([0-9]+)x([0-9]+)",
        normalize_classifier_name(classifier),
    )
    if not match:
        return None
    return (
        max(2, min(32, int(match.group(1)))),
        max(4, min(128, int(match.group(2)))),
        max(1, min(16, int(match.group(3)))),
    )


def is_plain_agcm_classifier(classifier):
    name = normalize_classifier_name(classifier)
    return (
        name in {"agcm_plain", "agcm_affine"}
        or agcm_lite_width(name) is not None
        or agcm_lowrank_rank(name) is not None
        or agcm_spatial_config(name) is not None
    )


def remap_condition_state_dict(state_dict, prefix="", classifier="color_condition"):
    classifier = normalize_classifier_name(classifier)
    lowrank_rank = agcm_lowrank_rank(classifier)
    if lowrank_rank is not None:
        mapped = dict(state_dict)
        w_key = f"{prefix}HRconv.weight"
        b_key = f"{prefix}HRconv.bias"
        if (
            w_key in mapped
            and f"{prefix}HRconv_reduce.weight" not in mapped
            and f"{prefix}HRconv_expand.weight" not in mapped
        ):
            weight = mapped.pop(w_key)
            bias = mapped.pop(b_key, None)
            matrix = weight.detach().float().reshape(weight.shape[0], weight.shape[1])
            try:
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                rank = min(int(lowrank_rank), int(s.numel()))
                sqrt_s = torch.sqrt(torch.clamp(s[:rank], min=0.0))
                reduce_weight = (sqrt_s[:, None] * vh[:rank, :]).reshape(
                    rank,
                    weight.shape[1],
                    1,
                    1,
                )
                expand_weight = (u[:, :rank] * sqrt_s[None, :]).reshape(
                    weight.shape[0],
                    rank,
                    1,
                    1,
                )
            except Exception:
                rank = min(int(lowrank_rank), int(weight.shape[0]), int(weight.shape[1]))
                reduce_weight = torch.zeros(rank, weight.shape[1], 1, 1)
                expand_weight = torch.zeros(weight.shape[0], rank, 1, 1)
                eye = torch.eye(rank)
                reduce_weight[:, :rank, 0, 0] = eye
                expand_weight[:rank, :, 0, 0] = eye
            mapped[f"{prefix}HRconv_reduce.weight"] = reduce_weight.to(dtype=weight.dtype)
            mapped[f"{prefix}HRconv_reduce.bias"] = torch.zeros(
                rank,
                dtype=weight.dtype,
                device=weight.device,
            )
            mapped[f"{prefix}HRconv_expand.weight"] = expand_weight.to(dtype=weight.dtype)
            if bias is None:
                bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
            mapped[f"{prefix}HRconv_expand.bias"] = bias
        return mapped

    lite_width = agcm_lite_width(classifier)
    if lite_width is not None:
        mapped = dict(state_dict)

        def _slice(key, *slices):
            full_key = f"{prefix}{key}"
            value = mapped.get(full_key)
            if value is None or not hasattr(value, "ndim"):
                return
            mapped[full_key] = value[tuple(slices)].clone()

        for name in (
            "cond_scale_first.weight",
            "cond_scale_HR.weight",
            "cond_shift_first.weight",
            "cond_shift_HR.weight",
        ):
            _slice(name, slice(0, lite_width), slice(None))
        for name in (
            "cond_scale_first.bias",
            "cond_scale_HR.bias",
            "cond_shift_first.bias",
            "cond_shift_HR.bias",
            "conv_first.weight",
            "conv_first.bias",
            "HRconv.bias",
        ):
            _slice(name, slice(0, lite_width))
        _slice("HRconv.weight", slice(0, lite_width), slice(0, lite_width), slice(None), slice(None))
        _slice("conv_last.weight", slice(None), slice(0, lite_width), slice(None), slice(None))
        return mapped

    if classifier != "color_condition_woin":
        return dict(state_dict)

    mapped = dict(state_dict)
    root = f"{prefix}classifier.model."
    if not any(key.startswith(f"{root}20.") for key in mapped):
        return mapped
    drop_prefixes = (
        f"{root}3.",
        f"{root}7.",
        f"{root}11.",
        f"{root}15.",
    )
    index_map = {
        "4": "3",
        "8": "6",
        "12": "9",
        "16": "12",
        "20": "16",
    }
    for key in list(mapped.keys()):
        if key.startswith(drop_prefixes):
            mapped.pop(key, None)
            continue
        for old, new in index_map.items():
            old_prefix = f"{root}{old}."
            if key.startswith(old_prefix):
                mapped[f"{root}{new}.{key[len(old_prefix):]}"] = mapped.pop(key)
                break
    return mapped



# Depth Comparison of Condition
# -------------------------------------------------------------------------------
# Release Version
class Color_Condition_3layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_3layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 32),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_4layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_4layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 64),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_Condition_6layer(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition_6layer, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 256, normalization=True),
            *color_block(256, 256),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)

# -------------------------------------------------------------------------------


class ConditionNet(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet, self).__init__()
        self.classifier_name = normalize_classifier_name(classifier)
        self.agcm_mode = "dynamic"
        self.GFM_nf = nf

        if self.classifier_name == 'agcm_plain':
            self.agcm_mode = "plain"
            self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
            self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
            self.act = nn.ReLU(inplace=True)
            return
        if self.classifier_name == 'agcm_affine':
            self.agcm_mode = "affine"
            hidden = max(8, min(int(nf), 32))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.affine = nn.Sequential(
                nn.Conv2d(3, hidden, 1, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 6, 1, 1, bias=True),
            )
            nn.init.zeros_(self.affine[-1].weight)
            nn.init.zeros_(self.affine[-1].bias)
            return
        spatial_config = agcm_spatial_config(self.classifier_name)
        if spatial_config is not None:
            self.agcm_mode = (
                "spatial_mix"
                if self.classifier_name.startswith("agcm_spatialmix")
                else "spatial_affine"
            )
            self.spatial_global = self.classifier_name.startswith(
                ("agcm_spatialmixglobal", "agcm_spatialmixg")
            )
            self.spatial_global_cnn = self.classifier_name.startswith(
                ("agcm_spatialmixglobalcnn", "agcm_spatialmixgcnn")
            )
            self.spatial_scale, self.spatial_width, self.spatial_depth = spatial_config
            self.spatial_in = nn.Conv2d(3, self.spatial_width, 1, 1, bias=True)
            layers = []
            for _ in range(self.spatial_depth):
                layers.append(nn.Conv2d(self.spatial_width, self.spatial_width, 3, 1, 1, bias=True))
                layers.append(nn.ReLU(inplace=True))
            self.spatial_trunk = nn.Sequential(*layers)
            spatial_out_channels = 12 if self.agcm_mode == "spatial_mix" else 6
            self.spatial_out = nn.Conv2d(self.spatial_width, spatial_out_channels, 1, 1, bias=True)
            if self.spatial_global:
                global_width = max(16, min(128, self.spatial_width))
                if self.spatial_global_cnn:
                    self.global_net = nn.Sequential(
                        nn.Conv2d(3, global_width, 1, 1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(global_width, global_width, 3, 2, 1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(global_width, global_width, 3, 2, 1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(global_width, spatial_out_channels, 1, 1, bias=True),
                    )
                else:
                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                    self.global_in = nn.Conv2d(3, global_width, 1, 1, bias=True)
                    self.global_out = nn.Conv2d(global_width, spatial_out_channels, 1, 1, bias=True)
            self.act = nn.ReLU(inplace=True)
            nn.init.zeros_(self.spatial_out.weight)
            nn.init.zeros_(self.spatial_out.bias)
            if self.spatial_global:
                if self.spatial_global_cnn:
                    nn.init.zeros_(self.global_net[-1].weight)
                    nn.init.zeros_(self.global_net[-1].bias)
                else:
                    nn.init.zeros_(self.global_out.weight)
                    nn.init.zeros_(self.global_out.bias)
            return
        lite_width = agcm_lite_width(self.classifier_name)
        if lite_width is not None:
            self.agcm_mode = "lite"
            self.GFM_nf = int(lite_width)
            self.classifier = Color_Condition(out_c=cond_c)
            self.cond_scale_first = nn.Linear(cond_c, self.GFM_nf)
            self.cond_scale_HR = nn.Linear(cond_c, self.GFM_nf)
            self.cond_scale_last = nn.Linear(cond_c, 3)
            self.cond_shift_first = nn.Linear(cond_c, self.GFM_nf)
            self.cond_shift_HR = nn.Linear(cond_c, self.GFM_nf)
            self.cond_shift_last = nn.Linear(cond_c, 3)
            self.conv_first = nn.Conv2d(3, self.GFM_nf, 1, 1)
            self.HRconv = nn.Conv2d(self.GFM_nf, self.GFM_nf, 1, 1)
            self.conv_last = nn.Conv2d(self.GFM_nf, 3, 1, 1)
            self.act = nn.ReLU(inplace=True)
            return
        lowrank_rank = agcm_lowrank_rank(self.classifier_name)
        if lowrank_rank is not None:
            self.agcm_mode = "lowrank"
            self.lowrank_rank = int(lowrank_rank)
            self.classifier = Color_Condition(out_c=cond_c)
            self.cond_scale_first = nn.Linear(cond_c, nf)
            self.cond_scale_HR = nn.Linear(cond_c, nf)
            self.cond_scale_last = nn.Linear(cond_c, 3)
            self.cond_shift_first = nn.Linear(cond_c, nf)
            self.cond_shift_HR = nn.Linear(cond_c, nf)
            self.cond_shift_last = nn.Linear(cond_c, 3)
            self.conv_first = nn.Conv2d(3, nf, 1, 1)
            self.HRconv_reduce = nn.Conv2d(nf, self.lowrank_rank, 1, 1)
            self.HRconv_expand = nn.Conv2d(self.lowrank_rank, nf, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 1, 1)
            self.act = nn.ReLU(inplace=True)
            return
        if self.classifier_name == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        elif self.classifier_name == 'color_condition_woin':
            self.classifier = Color_Condition_woIN(out_c=cond_c)
        elif self.classifier_name == 'color_condition_wodropout':
            self.classifier = Color_Condition_woDropout(out_c=cond_c)
        elif self.classifier_name == 'color_condition_3layer':
            self.classifier = Color_Condition_3layer(out_c=cond_c)
        elif self.classifier_name == 'color_condition_4layer':
            self.classifier = Color_Condition_4layer(out_c=cond_c)
        elif self.classifier_name == 'color_condition_6layer':
            self.classifier = Color_Condition_6layer(out_c=cond_c)
        else:
            raise ValueError(f"Unsupported classifier '{classifier}'")

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        if getattr(self, "agcm_mode", "dynamic") == "plain":
            out = self.conv_first(content)
            out = self.act(out)
            out = self.HRconv(out)
            out = self.act(out)
            out = self.conv_last(out)
            return out, x
        if getattr(self, "agcm_mode", "dynamic") == "affine":
            params = self.affine(self.pool(content))
            scale, shift = params[:, :3], params[:, 3:]
            out = content * (scale + 1.0) + shift
            return out, x
        if getattr(self, "agcm_mode", "dynamic") == "spatial_affine":
            low = F.avg_pool2d(content, self.spatial_scale, self.spatial_scale)
            params = self.act(self.spatial_in(low))
            params = self.spatial_trunk(params)
            params = self.spatial_out(params)
            if getattr(self, "spatial_global", False):
                if getattr(self, "spatial_global_cnn", False):
                    global_params = self.global_net(low)
                else:
                    global_params = self.global_out(self.act(self.global_in(self.global_pool(content))))
                params = params + global_params
            if params.shape[-2:] != content.shape[-2:]:
                params = F.interpolate(
                    params,
                    size=content.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            scale, shift = params[:, :3], params[:, 3:]
            out = content * (scale + 1.0) + shift
            return out, x
        if getattr(self, "agcm_mode", "dynamic") == "spatial_mix":
            low = F.avg_pool2d(content, self.spatial_scale, self.spatial_scale)
            params = self.act(self.spatial_in(low))
            params = self.spatial_trunk(params)
            params = self.spatial_out(params)
            if getattr(self, "spatial_global", False):
                if getattr(self, "spatial_global_cnn", False):
                    global_params = self.global_net(low)
                else:
                    global_params = self.global_out(self.act(self.global_in(self.global_pool(content))))
                params = params + global_params
            if params.shape[-2:] != content.shape[-2:]:
                params = F.interpolate(
                    params,
                    size=content.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            delta = params[:, :9]
            shift = params[:, 9:]
            r = content[:, 0:1]
            g = content[:, 1:2]
            b = content[:, 2:3]
            out_r = r * (delta[:, 0:1] + 1.0) + g * delta[:, 1:2] + b * delta[:, 2:3] + shift[:, 0:1]
            out_g = r * delta[:, 3:4] + g * (delta[:, 4:5] + 1.0) + b * delta[:, 5:6] + shift[:, 1:2]
            out_b = r * delta[:, 6:7] + g * delta[:, 7:8] + b * (delta[:, 8:9] + 1.0) + shift[:, 2:3]
            return torch.cat((out_r, out_g, out_b), dim=1), x

        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        if getattr(self, "agcm_mode", "dynamic") == "lowrank":
            out = self.HRconv_expand(self.HRconv_reduce(out))
        else:
            out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out, x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = remap_condition_state_dict(
            state_dict,
            prefix="",
            classifier=getattr(self, "classifier_name", "color_condition"),
        )
        return super().load_state_dict(mapped, strict=strict, assign=assign)


# 3 layers base model
class BaseModel(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last(out)

        return out


# Depth Comparison of Base Model
# -------------------------------------------------------------------------------
class BaseModel2layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel2layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.conv_last(out)

        return out


class BaseModel4layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel4layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last1(out)
        out = self.act(out)
        out = self.conv_last2(out)

        return out


class BaseModel5layer(nn.Module):
    def __init__(self, nf=64):
        super(BaseModel5layer, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]

        out = self.conv_first(content)
        out = self.act(out)
        out = self.HRconv(out)
        out = self.act(out)
        out = self.conv_last1(out)
        out = self.act(out)
        out = self.conv_last2(out)
        out = self.act(out)
        out = self.conv_last3(out)
        return out


# Depth Comparison of Condition
# -------------------------------------------------------------------------------
class ConditionNet2Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet2Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        print(self.classifier)
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, 3, 1, 1) + shift_HR.view(-1, 3, 1, 1) + out

        return out


class ConditionNet4Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet4Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise
        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last1 = nn.Linear(cond_c, nf)
        self.cond_scale_last2 = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last1 = nn.Linear(cond_c, nf)
        self.cond_shift_last2 = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last1 = self.cond_scale_last1(fea)
        shift_last1 = self.cond_shift_last1(fea)

        scale_last2 = self.cond_scale_last2(fea)
        shift_last2 = self.cond_shift_last2(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last1(out)
        out = out * scale_last1.view(-1, self.GFM_nf, 1, 1) + shift_last1.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last2(out)
        out = out * scale_last2.view(-1, 3, 1, 1) + shift_last2.view(-1, 3, 1, 1) + out

        return out


class ConditionNet5Layer(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet5Layer, self).__init__()
        if classifier == 'color_condition':
            self.classifier = Color_Condition(out_c=cond_c)
        else:
            raise
        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last1 = nn.Linear(cond_c, nf)
        self.cond_scale_last2 = nn.Linear(cond_c, nf)
        self.cond_scale_last3 = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last1 = nn.Linear(cond_c, nf)
        self.cond_shift_last2 = nn.Linear(cond_c, nf)
        self.cond_shift_last3 = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last1 = self.cond_scale_last1(fea)
        shift_last1 = self.cond_shift_last1(fea)

        scale_last2 = self.cond_scale_last2(fea)
        shift_last2 = self.cond_shift_last2(fea)

        scale_last3 = self.cond_scale_last3(fea)
        shift_last3 = self.cond_shift_last3(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last1(out)
        out = out * scale_last1.view(-1, self.GFM_nf, 1, 1) + shift_last1.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last2(out)
        out = out * scale_last2.view(-1, self.GFM_nf, 1, 1) + shift_last2.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last3(out)
        out = out * scale_last3.view(-1, 3, 1, 1) + shift_last3.view(-1, 3, 1, 1) + out

        return out
# -------------------------------------------------------------------------------



