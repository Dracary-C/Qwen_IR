from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from module.degradation_prompt import AssessPriorAdapter
from module.layout_prompt import QwenPromptPriorAdapter

from ..base import MethodAdapter, SourceRef, strip_module_prefix
from ..paths import PROJECT_ROOT, default_repo_root, push_cwd, push_sys_path, require_exists


TPGDIFF_WEIGHT_ROOT = Path(os.environ.get("QWEN_IR_TPGDIFF_WEIGHT_ROOT", str(PROJECT_ROOT / "weights" / "tpgdiff"))).expanduser()


def _resolve_checkpoint(path: Optional[Path | str], repo_root: Path, label: str) -> Optional[Path]:
    if path is None:
        return None

    raw = Path(path).expanduser()
    candidates = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(raw)
    if raw.is_absolute():
        add_candidate(repo_root / "pretrained" / raw.name)
        if len(raw.parts) > 2 and raw.parts[1] == "pretrained":
            add_candidate(repo_root / Path(*raw.parts[1:]))
    else:
        add_candidate(repo_root / raw)
        add_candidate(repo_root / "pretrained" / raw.name)
    add_candidate(TPGDIFF_WEIGHT_ROOT / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Checked: {checked}")


class TPGDiffPriorAdapter(MethodAdapter):
    name = "tpgdiff-prior"
    capabilities = ("prior", "encode", "score")
    source_refs = (
        SourceRef(
            repo="TPGDiff",
            entrypoints=(
                "universal-restoration/open_clip/factory.py",
                "universal-restoration/open_clip/prior_stage_model.py",
                "universal-restoration/open_clip/tpgd_model.py",
            ),
            notes="Reusable prior-stage CLIP and degradation encoder",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        checkpoint_path: Optional[Path] = None,
        options_path: Optional[Path] = None,
        num_degradations: int = 5,
        device: Optional[str] = None,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("tpgdiff")
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.options_path = Path(options_path).resolve() if options_path else None
        self.num_degradations = num_degradations
        self.device = device or "cuda"
        self.precision = precision
        self.clip_model = None
        self.preprocess = None

    def load(self) -> "TPGDiffPriorAdapter":
        repo = self.repo_root / "universal-restoration"
        require_exists(repo, "TPGDiff universal-restoration repo")

        with push_sys_path(repo), push_cwd(repo):
            import open_clip
            from open_clip.prior_stage_model import PriorStageModel

            base_model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                precision=self.precision,
                device=self.device,
            )
            teacher_encoder = base_model.visual
            student_encoder = deepcopy(base_model.visual)
            deg_backbone = deepcopy(base_model.visual)

            if hasattr(base_model.visual, "output_dim"):
                embed_dim = base_model.visual.output_dim
            elif hasattr(base_model, "embed_dim"):
                embed_dim = base_model.embed_dim
            else:
                raise RuntimeError("Cannot infer embed_dim from TPGDiff CLIP base model.")

            prior_model = PriorStageModel(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                deg_backbone=deg_backbone,
                embed_dim=embed_dim,
                num_degradations=self.num_degradations,
                content_loss_weight=1.0,
                deg_loss_weight=1.0,
                use_cosine_distill=True,
                normalize_embedding=True,
                freeze_teacher=True,
                freeze_deg_backbone=True,
            ).to(self.device)

            if self.checkpoint_path is not None:
                require_exists(self.checkpoint_path, "TPGDiff prior checkpoint")
                ckpt = torch.load(self.checkpoint_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                prior_model.load_state_dict(strip_module_prefix(state), strict=True)

            self.model = prior_model.eval()
            self.clip_model = base_model.eval()
            self.preprocess = preprocess
            self.loaded = True
        return self

    def get_content_prior(self, img_lq):
        return self.ensure_loaded().model.get_content_prior(img_lq)

    def get_degradation_prior(self, img_lq, as_prob: bool = True):
        return self.ensure_loaded().model.get_degradation_prior(img_lq, as_prob=as_prob)

    def forward(self, img_gt, img_lq, deg_label, return_embeddings: bool = True):
        return self.ensure_loaded().model(img_gt, img_lq, deg_label, return_embeddings=return_embeddings)

    def source_command(self) -> tuple[Path, tuple[str, ...]]:
        config_dir = self.repo_root / "universal-restoration" / "config" / "tpgd-sde"
        opt = self.options_path or (config_dir / "options" / "test.yml")
        return config_dir, ("python", "test.py", "-opt", str(opt))


class TPGDiffRuntimeAdapter(MethodAdapter):
    name = "tpgdiff-runtime"
    capabilities = ("restore", "prior", "demo")
    source_refs = (
        SourceRef(
            repo="TPGDiff",
            entrypoints=(
                "universal-restoration/config/tpgd-sde/app.py",
                "universal-restoration/config/tpgd-sde/test.py",
                "universal-restoration/config/tpgd-sde/models/denoising_model.py",
            ),
            notes="Full TPGDiff restoration runtime with content, degradation, and structure priors",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        options_path: Optional[Path] = None,
        restoration_checkpoint: Optional[Path] = None,
        tpgd_checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        sampling_mode: Optional[str] = None,
        assessment_checkpoint: Optional[Path] = None,
        adapter_hidden_dim: int = 1024,
        adapter_pool: str = "mean",
        adapter_dropout: float = 0.0,
        assessment_hidden_dim: int = 4096,
        strict_load: bool = False,
        image_range: str = "zero_one",
        boundary_pad: int = 32,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("tpgdiff")
        self.options_path = Path(options_path).resolve() if options_path else None
        restoration_checkpoint = restoration_checkpoint or os.environ.get("METHODHUB_TPGDIFF_RESTORE_CKPT")
        tpgd_checkpoint = tpgd_checkpoint or os.environ.get("METHODHUB_TPGDIFF_TPGD_CKPT")
        self.restoration_checkpoint = Path(restoration_checkpoint).expanduser() if restoration_checkpoint else None
        self.tpgd_checkpoint = Path(tpgd_checkpoint).expanduser() if tpgd_checkpoint else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_mode = sampling_mode
        self.assessment_checkpoint = Path(assessment_checkpoint).expanduser() if assessment_checkpoint else None
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_pool = adapter_pool
        self.adapter_dropout = float(adapter_dropout)
        self.assessment_hidden_dim = int(assessment_hidden_dim)
        self.strict_load = bool(strict_load)
        self.boundary_pad = max(0, int(boundary_pad or 0))
        self.image_range = str(image_range or "zero_one").strip().lower().replace("-", "_")
        if self.image_range in {"0_1", "01", "zero_one"}:
            self.image_range = "zero_one"
        elif self.image_range in {"_1_1", "minus_one_one", "neg_one_one", "negative_one_one"}:
            self.image_range = "minus_one_one"
        else:
            raise ValueError("image_range must be zero_one or minus_one_one")

        self.assess_prior: Any = None
        self.qwen_prompt_adapter: Any = None
        self.clip_model: Any = None
        self.prior_model: Any = None
        self.sde: Any = None
        self.opt: Any = None
        self.util: Any = None

    @staticmethod
    def _checkpoint_has_prefix(checkpoint_path: Path, prefix: str) -> bool:
        checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
        if not isinstance(checkpoint, dict):
            return False
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            return False
        return any(str(key).startswith(prefix) for key in model_state.keys())

    def load(self) -> "TPGDiffRuntimeAdapter":
        repo = require_exists(self.repo_root / "universal-restoration", "TPGDiff universal-restoration repo")
        config_dir = require_exists(repo / "config" / "tpgd-sde", "TPGDiff tpgd-sde config dir")
        opt_path = self.options_path or (config_dir / "options" / "test.yml")
        require_exists(opt_path, "TPGDiff runtime options")

        restoration_checkpoint = _resolve_checkpoint(
            self.restoration_checkpoint or (TPGDIFF_WEIGHT_ROOT / "universal-ir.pth"),
            self.repo_root,
            "TPGDiff restoration checkpoint",
        )
        tpgd_checkpoint = _resolve_checkpoint(
            self.tpgd_checkpoint or (TPGDIFF_WEIGHT_ROOT / "tpgd_ViT-B-32.pt"),
            self.repo_root,
            "TPGDiff prior/control checkpoint",
        )

        with push_sys_path(config_dir, repo), push_cwd(config_dir):
            import open_clip
            import options as option
            import utils as util
            from models import create_model
            from open_clip.prior_stage_model import PriorStageModel

            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            opt = option.parse(str(opt_path), is_train=False)
            if visible_devices:
                # TPGDiff's option parser rewrites CUDA_VISIBLE_DEVICES from YAML.
                # Keep the device selected by the launcher, and use gpu_ids as logical ids.
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            opt = option.dict_to_nonedict(opt)
            opt["gpu_ids"] = [0] if self.device.startswith("cuda") else None
            opt.setdefault("path", {})
            checkpoint_has_backbone = (
                self.assessment_checkpoint is not None
                and self._checkpoint_has_prefix(self.assessment_checkpoint, "backbone.")
            )
            opt["path"]["pretrain_model_G"] = None if checkpoint_has_backbone else str(restoration_checkpoint)
            opt["path"]["tpgd"] = str(tpgd_checkpoint)
            opt["path"]["prior"] = str(tpgd_checkpoint)

            model = create_model(opt)
            base_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                precision="fp32",
                device=self.device,
            )
            teacher_encoder = base_model.visual
            student_encoder = deepcopy(base_model.visual)
            deg_backbone = deepcopy(base_model.visual)

            if hasattr(base_model.visual, "output_dim"):
                embed_dim = base_model.visual.output_dim
            elif hasattr(base_model, "embed_dim"):
                embed_dim = base_model.embed_dim
            else:
                raise RuntimeError("Cannot infer embed_dim from TPGDiff CLIP base model.")

            prior_model = PriorStageModel(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                deg_backbone=deg_backbone,
                embed_dim=embed_dim,
                num_degradations=len(opt["distortion"]),
                content_loss_weight=1.0,
                deg_loss_weight=1.0,
                use_cosine_distill=True,
                normalize_embedding=True,
                freeze_teacher=True,
                freeze_deg_backbone=True,
            )
            ckpt = torch.load(tpgd_checkpoint, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            prior_model.load_state_dict(strip_module_prefix(state), strict=True)
            prior_model = prior_model.to(self.device).eval()

            sde = util.IRSDE(
                max_sigma=opt["sde"]["max_sigma"],
                T=opt["sde"]["T"],
                schedule=opt["sde"]["schedule"],
                eps=opt["sde"]["eps"],
                device=model.device,
            )
            sde.set_model(model.model)

            self.model = model
            self.clip_model = base_model.eval()
            self.prior_model = prior_model
            self.sde = sde
            self.opt = opt
            self.util = util
            self.sampling_mode = self.sampling_mode or opt["sde"]["sampling_mode"]
            if self.assessment_checkpoint:
                self.assess_prior, self.qwen_prompt_adapter = self._load_checkpoint_priors()
            else:
                self.assess_prior = None
                self.qwen_prompt_adapter = None
            self.loaded = True
        return self

    def _load_checkpoint_priors(self) -> tuple[Any, Any]:
        require_exists(self.assessment_checkpoint, "Assess-TPGD checkpoint")
        network_setting = ((self.opt or {}).get("network_G", {}) or {}).get("setting", {}) or {}
        structure_setting = ((self.opt or {}).get("structure_prior", {}) or {}).get("setting", {}) or {}
        context_dim = int(network_setting.get("context_dim", 512))
        struct_context_dim = int(network_setting.get("struct_context_dim", structure_setting.get("token_dim", 128)))
        num_struct_tokens = int(structure_setting.get("num_latent_tokens", 32))
        use_layout_tokens = bool(network_setting.get("use_struct_context", True))

        checkpoint = torch.load(self.assessment_checkpoint, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported Assess-TPGD checkpoint payload: {type(checkpoint)!r}")

        model_state = checkpoint.get("model")
        if isinstance(model_state, dict):
            backbone_state = {
                key[len("backbone."):]: value
                for key, value in model_state.items()
                if key.startswith("backbone.")
            }
            if backbone_state and hasattr(self.model, "model"):
                target_backbone = self.model.model.module if hasattr(self.model.model, "module") else self.model.model
                incompatible_backbone = target_backbone.load_state_dict(backbone_state, strict=self.strict_load)
                print(
                    f"assessment_backbone_checkpoint={self.assessment_checkpoint} "
                    f"missing={len(incompatible_backbone.missing_keys)} unexpected={len(incompatible_backbone.unexpected_keys)}",
                    flush=True,
                )

        assess_state = checkpoint.get("assess_prior")
        if not isinstance(assess_state, dict) and isinstance(model_state, dict):
            assess_state = {
                key[len("assess_prior."):]: value
                for key, value in model_state.items()
                if key.startswith("assess_prior.")
            }

        qwen_prompt_state = checkpoint.get("qwen_prompt_prior")
        qwen_prompt_key = "qwen_prompt_prior" if isinstance(qwen_prompt_state, dict) else "structured_prior"
        if not isinstance(qwen_prompt_state, dict):
            qwen_prompt_state = checkpoint.get("structured_prior")
        if not isinstance(qwen_prompt_state, dict) and isinstance(model_state, dict):
            qwen_prompt_state = {
                key[len("structured_prior."):]: value
                for key, value in model_state.items()
                if key.startswith("structured_prior.")
            }
            qwen_prompt_key = "structured_prior"

        assess_prior = None
        if isinstance(assess_state, dict) and assess_state:
            assess_prior = AssessPriorAdapter(
                input_dim=self.assessment_hidden_dim,
                output_dim=context_dim,
                hidden_dim=self.adapter_hidden_dim,
                pool=self.adapter_pool,
                dropout=self.adapter_dropout,
            ).to(self.device)
            incompatible = assess_prior.load_state_dict(strip_module_prefix(assess_state), strict=self.strict_load)
            print(
                f"assessment_checkpoint={self.assessment_checkpoint} key=assess_prior "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )
            assess_prior.eval()
            for param in assess_prior.parameters():
                param.requires_grad = False

        qwen_prompt_prior = None
        if isinstance(qwen_prompt_state, dict) and qwen_prompt_state:
            qwen_prompt_prior = QwenPromptPriorAdapter(
                context_dim=context_dim,
                struct_context_dim=struct_context_dim,
                num_struct_tokens=num_struct_tokens,
                hidden_dim=self.adapter_hidden_dim,
                use_layout_tokens=use_layout_tokens,
            ).to(self.device)
            incompatible = qwen_prompt_prior.load_state_dict(strip_module_prefix(qwen_prompt_state), strict=self.strict_load)
            print(
                f"assessment_checkpoint={self.assessment_checkpoint} key={qwen_prompt_key} "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )
            qwen_prompt_prior.eval()
            for param in qwen_prompt_prior.parameters():
                param.requires_grad = False

        if assess_prior is None and qwen_prompt_prior is None:
            raise KeyError(
                "Assess-TPGD checkpoint must contain assess_prior/qwen_prompt_prior, "
                "or compatible model keys prefixed with assess_prior./structured_prior/."
            )
        return assess_prior, qwen_prompt_prior

    def _load_assess_prior(self) -> AssessPriorAdapter:
        require_exists(self.assessment_checkpoint, "Assess-TPGD checkpoint")
        network_setting = ((self.opt or {}).get("network_G", {}) or {}).get("setting", {}) or {}
        context_dim = int(network_setting.get("context_dim", 512))
        adapter = AssessPriorAdapter(
            input_dim=self.assessment_hidden_dim,
            output_dim=context_dim,
            hidden_dim=self.adapter_hidden_dim,
            pool=self.adapter_pool,
            dropout=self.adapter_dropout,
        ).to(self.device)
        checkpoint = torch.load(self.assessment_checkpoint, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported Assess-TPGD checkpoint payload: {type(checkpoint)!r}")
        state = checkpoint.get("assess_prior")
        if not isinstance(state, dict) and isinstance(checkpoint.get("model"), dict):
            state = {
                key[len("assess_prior."):]: value
                for key, value in checkpoint["model"].items()
                if key.startswith("assess_prior.")
            }
        if not isinstance(state, dict) or not state:
            raise KeyError("Assess-TPGD checkpoint must contain 'assess_prior' or model keys prefixed with 'assess_prior.'.")
        model_state = checkpoint.get("model")
        if isinstance(model_state, dict):
            backbone_state = {
                key[len("backbone."):]: value
                for key, value in model_state.items()
                if key.startswith("backbone.")
            }
            if backbone_state and hasattr(self.model, "model"):
                target_backbone = self.model.model.module if hasattr(self.model.model, "module") else self.model.model
                incompatible_backbone = target_backbone.load_state_dict(backbone_state, strict=self.strict_load)
                print(
                    f"assessment_backbone_checkpoint={self.assessment_checkpoint} "
                    f"missing={len(incompatible_backbone.missing_keys)} unexpected={len(incompatible_backbone.unexpected_keys)}",
                    flush=True,
                )
        state = strip_module_prefix(state)
        incompatible = adapter.load_state_dict(state, strict=self.strict_load)
        print(
            f"assessment_checkpoint={self.assessment_checkpoint} "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
        adapter.eval()
        for param in adapter.parameters():
            param.requires_grad = False
        return adapter

    def _assessment_deg_context(self, assessment_hidden: torch.Tensor) -> torch.Tensor:
        if self.assess_prior is None:
            raise RuntimeError("assessment_hidden was provided, but no assessment_checkpoint was configured.")
        hidden = assessment_hidden.detach()
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3:
            raise ValueError(f"Expected assessment_hidden [T,C] or [B,T,C], got {tuple(hidden.shape)}")
        hidden = hidden.to(self.device)
        with torch.no_grad():
            return self.assess_prior(hidden).float()

    def _qwen_prompt_contexts(self, qwen_prompt_prior: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.qwen_prompt_adapter is None:
            raise RuntimeError("qwen_prompt_prior was provided, but no qwen_prompt adapter was loaded from checkpoint.")
        prior = torch.as_tensor(qwen_prompt_prior, dtype=torch.float32, device=self.device)
        if prior.ndim == 1:
            prior = prior.unsqueeze(0)
        if prior.ndim != 2 or prior.shape[-1] != 21:
            raise ValueError(f"Expected qwen_prompt_prior [21] or [B,21], got {tuple(prior.shape)}")
        with torch.no_grad():
            contexts = self.qwen_prompt_adapter(prior)
        struct_tokens = contexts.struct_tokens.float() if contexts.struct_tokens is not None else None
        return contexts.deg_context.float(), struct_tokens

    @staticmethod
    def _to_numpy_rgb(image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif torch.is_tensor(image):
            tensor = image.detach().float().cpu()
            if tensor.dim() == 4:
                if tensor.shape[0] != 1:
                    raise ValueError("TPGDiffRuntimeAdapter.restore only supports one image at a time.")
                tensor = tensor[0]
            if tensor.dim() != 3:
                raise ValueError(f"Expected image tensor with shape [C,H,W], got {tuple(tensor.shape)}.")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor[:3]
                if tensor.shape[0] == 1:
                    tensor = tensor.expand(3, -1, -1)
                array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
                array = (array * 255.0).round().astype(np.uint8)
            else:
                raise ValueError(f"Expected channel-first tensor, got shape {tuple(tensor.shape)}.")
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.ndim != 3:
                raise ValueError(f"Expected image array with shape [H,W,C], got {array.shape}.")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.dtype != np.uint8:
                array = array.astype(np.float32)
                if array.max() <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).round().astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _clip_transform(np_image: np.ndarray, resolution: int = 224) -> torch.Tensor:
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

        pil_image = Image.fromarray(np_image)
        transform = Compose(
            [
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(resolution),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return transform(pil_image)

    def restore(self, image: Any, *, assessment_hidden: torch.Tensor | None = None, qwen_prompt_prior: Any = None, structured_prior: Any = None) -> Image.Image:
        self.ensure_loaded()
        np_rgb = self._to_numpy_rgb(image)
        if self.image_range == "minus_one_one":
            np_float = np_rgb.astype(np.float32) / 127.5 - 1.0
            output_minmax = (-1, 1)
        else:
            np_float = np_rgb.astype(np.float32) / 255.0
            output_minmax = (0, 1)

        img4clip = self._clip_transform(np_rgb).unsqueeze(0).to(self.device)
        amp_enabled = self.device.startswith("cuda") and torch.cuda.is_available()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            content_context = self.prior_model.get_content_prior(img4clip).float()
            struct_tokens = None
            effective_qwen_prompt = qwen_prompt_prior if qwen_prompt_prior is not None else structured_prior
            if effective_qwen_prompt is not None:
                deg_context, struct_tokens = self._qwen_prompt_contexts(effective_qwen_prompt)
            elif assessment_hidden is not None:
                deg_context = self._assessment_deg_context(assessment_hidden)
            else:
                deg_context = self.prior_model.encode_for_degradation(img4clip).float()

        lq_tensor = torch.from_numpy(np_float).permute(2, 0, 1).unsqueeze(0).float()
        orig_h, orig_w = lq_tensor.shape[-2:]
        pad_h = min(self.boundary_pad, max(orig_h - 1, 0))
        pad_w = min(self.boundary_pad, max(orig_w - 1, 0))
        if pad_h > 0 or pad_w > 0:
            lq_model = F.pad(lq_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        else:
            lq_model = lq_tensor

        noisy_tensor = self.sde.noise_state(lq_model)
        self.model.feed_data(
            noisy_tensor,
            lq_model,
            deg_context=deg_context,
            content_context=content_context,
        )
        self.model.test(self.sde, mode=self.sampling_mode, save_states=False, struct_tokens=struct_tokens)
        visuals = self.model.get_current_visuals(need_GT=False)
        output_tensor = visuals["Output"]
        if pad_h > 0:
            output_tensor = output_tensor[..., pad_h:pad_h + orig_h, :]
        if pad_w > 0:
            output_tensor = output_tensor[..., :, pad_w:pad_w + orig_w]
        output_bgr = self.util.tensor2img(output_tensor.squeeze(), min_max=output_minmax)
        output_rgb = output_bgr[:, :, [2, 1, 0]]
        return Image.fromarray(output_rgb)

    def process(self, image: Any, name: str = "sample") -> Image.Image:
        return self.restore(image)

    def source_command(self) -> tuple[Path, tuple[str, ...]]:
        config_dir = self.repo_root / "universal-restoration" / "config" / "tpgd-sde"
        opt = self.options_path or (config_dir / "options" / "test.yml")
        return config_dir, ("python", "app.py", "-opt", str(opt))
