import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


def _to_pil_list(x: torch.Tensor):
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    x = x.clamp(0.0, 1.0)
    x_cpu = x.detach().cpu()
    pil_list = [TF.to_pil_image(img) for img in x_cpu]
    return pil_list


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]

    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    pos_x = grid[..., 0:1] * omega  # [H, W, dim//4]
    pos_y = grid[..., 1:2] * omega  # [H, W, dim//4]

    sin_x = torch.sin(pos_x)
    cos_x = torch.cos(pos_x)
    sin_y = torch.sin(pos_y)
    cos_y = torch.cos(pos_y)

    pos_embed = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  # [H, W, dim]
    pos_embed = pos_embed.view(h * w, dim)
    return pos_embed  # [H*W, dim]



class SegFormerWrapper(nn.Module):
    def __init__(self,
                 model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        self._model_device = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self._model_device != device:
            self.model.to(device)
            self._model_device = device

        pil_list = _to_pil_list(x)
        inputs = self.processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        num_labels = logits.shape[1]
        labels = logits.argmax(dim=1, keepdim=True).float()  # [B,1,h',w']
        seg_norm = labels / max(num_labels - 1, 1)
        return seg_norm



class StructEncoder(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 base_ch: int = 32,
                 num_blocks: int = 8,
                 token_dim: int = 256):
        super().__init__()

        layers = []
        ch_in = in_ch
        ch_out = base_ch

        for i in range(num_blocks):
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out

            if i % 2 == 1:
                next_ch = min(ch_out * 2, token_dim)
                layers.append(
                    nn.Conv2d(ch_in, next_ch, kernel_size=3, stride=2, padding=1)
                )
                layers.append(nn.ReLU(inplace=True))
                ch_in = next_ch
                ch_out = next_ch
            else:
                ch_out = min(ch_out * 2, token_dim)

        self.conv = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch_in, token_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.proj(feat)
        return feat


class StructuralConnector(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 num_latent_tokens: int = 64,
                 nhead: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.d_model = d_model

        self.latent_tokens = nn.Parameter(
            torch.randn(1, num_latent_tokens, d_model)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        self.self_attn1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.self_attn2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        hidden = int(d_model * mlp_ratio)

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        self.mlp2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_self1 = nn.LayerNorm(d_model)
        self.ln_self2 = nn.LayerNorm(d_model)

    def forward(self, tokens_seg):
        tokens_struct = tokens_seg  # [B, N_seg, d_model]

        B = tokens_struct.size(0)
        latent = self.latent_tokens.expand(B, -1, -1)  # [B, N_m, d_model]

        # Cross-Attention: Q = latent, K/V = tokens_struct
        q = self.ln_q(latent)
        kv = self.ln_kv(tokens_struct)
        cross_out, _ = self.cross_attn(q, kv, kv)
        latent = latent + cross_out
        latent = latent + self.mlp1(latent)

        # Self-Attention1 on latent tokens
        s1 = self.ln_self1(latent)
        self_out1, _ = self.self_attn1(s1, s1, s1)
        latent = latent + self_out1
        latent = latent + self.mlp2(latent)

        # Self-Attention2 on latent tokens
        s2 = self.ln_self2(latent)
        self_out2, _ = self.self_attn2(s2, s2, s2)
        latent = latent + self_out2
        latent = latent + self.mlp3(latent)

        return latent  # [B, N_m, d_model]


class StructurePriorModule(nn.Module):

    def __init__(
        self,
        in_nc: int = 3,
        base_channels: int = 32,
        num_blocks: int = 8,
        token_dim: int = 256,
        image_size: int = 256,
        num_latent_tokens: int = 64,
        depth_model_name: str = "LiheYoung/depth-anything-small-hf",
        seg_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    ):
        super().__init__()

        self.in_nc = in_nc
        self.token_dim = token_dim
        self.image_size = image_size

        self.seg_extractor = SegFormerWrapper(seg_model_name)

        for p in self.seg_extractor.parameters():
            p.requires_grad = False

        self.struct_encoder = StructEncoder(
            in_ch=1,
            base_ch=base_channels,
            num_blocks=num_blocks,
            token_dim=token_dim,
        )

        # Keep modality embedding for API consistency, but only segmentation is used.
        self.modality_embed = nn.Parameter(torch.randn(1, token_dim))

        self.connector = StructuralConnector(
            d_model=token_dim,
            num_latent_tokens=num_latent_tokens,
            nhead=8,
            mlp_ratio=4.0,
        )

        self._cached_pos_shape = None
        self._cached_pos_embed = None

    def _get_pos_embed(self, h: int, w: int, device):

        if (
            self._cached_pos_shape is not None
            and self._cached_pos_shape == (h, w)
            and self._cached_pos_embed is not None
            and self._cached_pos_embed.device == device
        ):
            return self._cached_pos_embed

        pos = _build_2d_sincos_pos_embed(h, w, self.token_dim, device=device)
        pos = pos.unsqueeze(0)  # [1, H*W, C]
        self._cached_pos_shape = (h, w)
        self._cached_pos_embed = pos
        return pos

    def _encode_struct_map(self, x: torch.Tensor):

        feat = self.struct_encoder(x)  # [B, C, H', W']
        B, C, Hf, Wf = feat.shape

        tokens = feat.view(B, C, Hf * Wf).transpose(1, 2)  # [B, N, C]

        tokens = tokens + self.modality_embed[0].view(1, 1, C)

        pos = self._get_pos_embed(Hf, Wf, feat.device)  # [1, N, C]
        tokens = tokens + pos

        return tokens  # [B, N, C]

    def forward(self, lq: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            seg_map = self.seg_extractor(lq)      # [B,1,h_s,w_s]
        target_size = (self.image_size, self.image_size)

        seg_map = F.interpolate(seg_map, size=target_size, mode="nearest")

        tokens_seg = self._encode_struct_map(seg_map)

        struct_tokens = self.connector(tokens_seg)
        # struct_tokens: [B, N_m, token_dim]

        return struct_tokens
