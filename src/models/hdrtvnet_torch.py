import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_HAS_COMPILE = hasattr(torch, "compile")          # PyTorch >= 2.0
_HAS_CUDA_GRAPHS = hasattr(torch.cuda, "CUDAGraph")  # PyTorch >= 1.10
_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Enable TF32 on Ampere+ GPUs — harmless no-op on AMD
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class HDRTVNetTorch:
    """PyTorch inference wrapper for HDRTVNet with platform-aware optimizations.

    Optimizations applied automatically:
      * torch.inference_mode() (lower overhead than no_grad).
      * Pre-allocated GPU tensors to avoid per-frame allocation.
      * cudnn.benchmark + channels_last on NVIDIA (skipped on ROCm where
        MIOpen NHWC can be slower for small models with mixed ops).
      * torch.compile() when Triton is available.
      * Optional CUDA-graph replay for static-shape inputs.
    """

    def __init__(self, model_path, device="auto", precision="auto",
                 compile_model=True, force_compile=False, use_cuda_graphs=False,
                 force_channels_last=False):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.precision = self._resolve_precision(precision, self.device)
        self._use_cuda = self.device.type == "cuda"
        self._dtype = {"fp16": torch.float16}.get(
            self.precision, torch.float32
        )

        self.model = self._load_model(model_path)

        # ---- Platform-specific: channels_last + cudnn.benchmark -----------
        # Auto-enabled on NVIDIA (cuDNN benefits); skipped on ROCm where
        # MIOpen NHWC can hurt.  --channels-last forces it on for A/B testing.
        if self._use_cuda and (not _IS_ROCM or force_channels_last):
            if not _IS_ROCM:
                torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)
            self._use_channels_last = True
            if _IS_ROCM:
                print("ROCm + --channels-last: channels_last forced ON")
            else:
                print("NVIDIA detected — cudnn.benchmark + channels_last enabled")
        else:
            self._use_channels_last = False
            if self._use_cuda and _IS_ROCM:
                print("ROCm detected — skipping channels_last and "
                      "cudnn.benchmark (MIOpen works better without them "
                      "for this model).  Use --channels-last to override.")

        # ---- torch.compile (PyTorch 2.x) ----------------------------------
        self._compiled = False
        if compile_model and _HAS_COMPILE and self._use_cuda:
            if _IS_ROCM and not force_compile:
                print("torch.compile skipped on ROCm \u2014 Triton ROCm "
                      "codegen needs HIP SDK on Windows.\n"
                      "  Use --force-compile to try anyway (requires HIP SDK)")
            elif not _HAS_TRITON:
                print("torch.compile skipped \u2014 Triton not installed.\n"
                      "  pip install triton")
            else:
                if _IS_ROCM and force_compile:
                    print("--force-compile: attempting torch.compile on ROCm...")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune",
                        fullgraph=False,
                    )
                    self._compiled = True
                    print("torch.compile enabled (max-autotune)")
                except Exception as exc:
                    print(f"torch.compile setup failed: {exc}")

        self.expected_hw = None
        self.is_static_input_model = False

        # ---- Pre-allocated buffer state ------------------------------------
        self._buf_hw = None
        self._gpu_input = None       # persistent GPU tensor (1,3,H,W)
        self._gpu_cond = None        # persistent GPU tensor (1,3,H//4,W//4)

        # ---- CUDA graph state (optional) -----------------------------------
        self._use_cuda_graphs = (
            use_cuda_graphs and self._use_cuda and _HAS_CUDA_GRAPHS
        )
        self._graph = None
        self._graph_input = None
        self._graph_cond = None
        self._graph_output = None
        self._graph_hw = None

        print(f"PyTorch device : {self.device}")
        print(f"PyTorch precision: {self.precision}")

    # -----------------------------------------------------------------------
    # Device / precision helpers
    # -----------------------------------------------------------------------
    def _resolve_device(self, device):
        mode = device.lower()
        if mode == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm device not available for PyTorch.")
            return torch.device("cuda:0")
        if mode == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be one of: auto, cuda, cpu")

    def _resolve_precision(self, precision, device):
        p = precision.lower()
        if p not in {"auto", "fp16", "fp32"}:
            raise ValueError("precision must be one of: auto, fp16, fp32")
        if p == "auto":
            return "fp16" if device.type == "cuda" else "fp32"
        if device.type != "cuda" and p == "fp16":
            return "fp32"
        return p

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    def _load_model(self, model_path):
        ext = os.path.splitext(model_path)[1].lower()
        if ext in {".pt", ".ts"}:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
        else:
            model = Ensemble_AGCM_LE(
                classifier="color_condition",
                cond_c=6,
                in_nc=3,
                out_nc=3,
                nf=32,
                act_type="relu",
                weighting_network=False,
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            cleaned = {}
            for key, value in state_dict.items():
                cleaned[key[7:] if key.startswith("module.") else key] = value
            model.load_state_dict(cleaned, strict=True)
            model.eval()

        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half()
        else:
            model = model.float()

        return model

    # -----------------------------------------------------------------------
    # Buffer management — allocate once, reuse every frame
    # -----------------------------------------------------------------------
    def _ensure_buffers(self, h, w):
        """Allocate / reallocate persistent GPU tensors when resolution changes.
        For the common case (fixed resolution) this is a no-op after the first
        frame."""
        if self._buf_hw == (h, w):
            return
        self._buf_hw = (h, w)
        cond_h, cond_w = max(1, h // 4), max(1, w // 4)

        # Persistent GPU tensors — avoids torch.empty() + .to(device) per frame
        mem_fmt = (torch.channels_last if self._use_channels_last
                   else torch.contiguous_format)
        self._gpu_input = torch.empty(
            (1, 3, h, w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)
        self._gpu_cond = torch.empty(
            (1, 3, cond_h, cond_w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)

        # Invalidate any cached CUDA graph on resolution change
        self._graph = None
        self._graph_hw = None

    # -----------------------------------------------------------------------
    # Preprocess — keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._ensure_buffers(h, w)

        # Upload raw uint8 BGR frame to GPU, then do all math there
        # from_numpy is zero-copy; the .to(device) does the actual H2D
        raw = torch.from_numpy(frame_bgr).to(
            device=self.device, non_blocking=self._use_cuda
        )
        # (H,W,3) uint8 → BGR→RGB via channel flip → CHW → add batch → fp → /255
        raw = raw.flip(2)                                  # BGR → RGB
        raw = raw.permute(2, 0, 1).unsqueeze(0)           # (1,3,H,W)
        self._gpu_input.copy_(
            raw.to(dtype=self._dtype).mul_(1.0 / 255.0),
            non_blocking=self._use_cuda,
        )

        # Condition tensor (0.25× spatial)
        self._gpu_cond.copy_(
            F.interpolate(
                self._gpu_input,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ),
            non_blocking=self._use_cuda,
        )

        return self._gpu_input, self._gpu_cond

    # -----------------------------------------------------------------------
    # Inference — CUDA graph replay when possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def infer(self, input_cond):
        tensor, cond = input_cond

        # --- CUDA graph path (static shapes only) -------------------------
        if self._use_cuda_graphs and self._use_cuda:
            h, w = tensor.shape[2], tensor.shape[3]
            if self._graph is not None and self._graph_hw == (h, w):
                self._graph_input.copy_(tensor)
                self._graph_cond.copy_(cond)
                self._graph.replay()
                return self._graph_output
            else:
                self._graph_hw = (h, w)
                self._graph_input = tensor.clone()
                self._graph_cond = cond.clone()

                # Warmup (required before recording)
                for _ in range(3):
                    _ = self.model((self._graph_input, self._graph_cond))
                torch.cuda.synchronize()

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._graph_output = self.model(
                        (self._graph_input, self._graph_cond)
                    )
                self._graph_input.copy_(tensor)
                self._graph_cond.copy_(cond)
                self._graph.replay()
                return self._graph_output

        # --- Eager / compiled path -----------------------------------------
        try:
            return self.model((tensor, cond))
        except Exception as exc:
            if self._compiled:
                print(f"torch.compile failed at runtime, reverting to eager: {exc}")
                self.model = self.model._orig_mod
                self._compiled = False
                return self.model((tensor, cond))
            raise

    # -----------------------------------------------------------------------
    # Postprocess — keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def postprocess(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        # All math on GPU: clamp, scale, quantize, channel-flip (RGB→BGR)
        t = output.squeeze(0)                      # (3,H,W)  fp16/fp32
        t = t.clamp_(0.0, 1.0).mul_(255.0)        # [0,255] still on GPU
        t = t.to(dtype=torch.uint8)                # quantize on GPU
        t = t.flip(0)                              # RGB→BGR via channel flip
        t = t.permute(1, 2, 0).contiguous()        # CHW → HWC, contiguous for cv2
        return t.cpu().numpy()                     # single D2H copy of uint8

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def process(self, frame_bgr):
        tensor, cond = self.preprocess(frame_bgr)
        out = self.infer((tensor, cond))
        return self.postprocess(out)

    @torch.inference_mode()
    def process_timed(self, frame_bgr):
        t0 = time.perf_counter()
        tensor, cond = self.preprocess(frame_bgr)
        if self._use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = self.infer((tensor, cond))
        if self._use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        output = self.postprocess(out)
        t3 = time.perf_counter()

        return output, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0

    def end_profiling(self):
        return None
