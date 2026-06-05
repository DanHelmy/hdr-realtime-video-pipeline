import os
import re

import torch
import torch.nn as nn
from .Condition_arch import ConditionNet, is_plain_agcm_classifier, remap_condition_state_dict
from .HDRUNet3T1_arch import (
    HDRUNet3T1,
    HDRUNet3T1BottleneckHeavy,
    HDRUNet3T1CleanTrunk,
    HDRUNet3T1CleanTrunkDeep,
    HDRUNet3T1CleanTrunkWideExtra,
    HDRUNet3T1CondDirect,
    HDRUNet3T1FlatTrunk,
    HDRUNet3T1PlainBottleneck,
    HDRUNet3T1PlainDirect,
    HDRUNet3T1PlainFlat,
    HDRUNet3T1SelectiveSFT,
)


class Ensemble_AGCM_LE(nn.Module):
    def __init__(self, classifier='color_condition', cond_c=6, in_nc=3, out_nc=3, nf=32, act_type='relu', weighting_network=False, le_arch=None):
        super(Ensemble_AGCM_LE, self).__init__()
        self.AGCM = ConditionNet(classifier=classifier, cond_c=cond_c)
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
            cond_direct_match = re.fullmatch(
                r"conddirecth(4|8|16|32)wide([0-9]+)x([0-9]+)",
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
        elif le_arch in {"cleantrunk", "clean-trunk", "qfriendly", "quant-friendly"}:
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
        return LE_output[0], condition_output

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
