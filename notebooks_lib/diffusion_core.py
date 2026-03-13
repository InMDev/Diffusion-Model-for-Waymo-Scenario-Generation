from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default constants aligned with train.ipynb defaults.
SEED = 24
HIST_DIM = 13
NBR_DIM = 10
MAP_SCALAR_DIM = 5
MAP_POINT_DIM = 11
MAP_SIGNAL_DIM = 10
MAP_POLYLINE_POINTS = 64
MAP_POLYLINE_BUDGET = 256
STATIC_DIM = 7
FUTURE_STEPS = 80
TARGET_DIM = FUTURE_STEPS * 4
COND_DIM = 256
DIT_MODEL_DIM = 384
DIT_DEPTH = 8
DIT_HEADS = 8
POS_TOKEN_COUNT = 512
TRAJ_TOKEN_COUNT = 1024
TRAJ_TOKEN_STEPS = 5
TOKEN_BUILD_MAX_SHARDS = 32
TOKEN_BUILD_MAX_SAMPLES = 200_000
TOKEN_KMEANS_ITERS = 18
EMA_DECAY = 0.999
CFG_DROPOUT = 0.1


def load_shard(shard_path: str) -> dict:
    """Load a cached shard from disk onto CPU."""
    return torch.load(shard_path, map_location="cpu")


def make_cosine_schedule(T: int, device: torch.device) -> dict[str, torch.Tensor]:
    s = 0.008
    x = torch.linspace(0, T, T + 1, device=device)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 1e-4, 0.999)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "T": torch.tensor(T, device=device),
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def _balanced_subsample(x: torch.Tensor, max_samples: int, seed: int = SEED) -> torch.Tensor:
    if x.shape[0] <= max_samples:
        return x
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:max_samples]
    return x[idx]


def _run_kmeans(data: torch.Tensor, k: int, iters: int = TOKEN_KMEANS_ITERS, seed: int = SEED) -> torch.Tensor:
    # data: [N, D] on CPU
    n = data.shape[0]
    d = data.shape[1]
    if n == 0:
        return torch.zeros((k, d), dtype=torch.float32)
    if n < k:
        reps = math.ceil(k / n)
        data = data.repeat((reps, 1))[:k]
        return data.clone().float()

    g = torch.Generator(device=data.device)
    g.manual_seed(seed)
    init_idx = torch.randperm(n, generator=g, device=data.device)[:k]
    centers = data[init_idx].clone()

    chunk = 4096
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros((k,), dtype=torch.float32, device=data.device)

        for start in range(0, n, chunk):
            batch = data[start : start + chunk]
            dist = ((batch.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
            assign = dist.argmin(dim=1)
            sums.index_add_(0, assign, batch)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32))

        nonzero = counts > 0
        centers[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(-1)
        missing = (~nonzero).nonzero(as_tuple=False).flatten()
        if missing.numel() > 0:
            refill_idx = torch.randint(0, n, (missing.numel(),), generator=g, device=data.device)
            centers[missing] = data[refill_idx]

    return centers.float()


def build_token_tables_from_shards(
    shard_paths: list[str],
    pos_k: int = POS_TOKEN_COUNT,
    traj_k: int = TRAJ_TOKEN_COUNT,
    traj_steps: int = TRAJ_TOKEN_STEPS,
    max_shards: int = TOKEN_BUILD_MAX_SHARDS,
    max_samples: int = TOKEN_BUILD_MAX_SAMPLES,
) -> dict[str, torch.Tensor]:
    pos_samples = []
    traj_samples = []

    for shard_path in shard_paths[: max(1, max_shards)]:
        shard = load_shard(shard_path)
        target = shard["target"].float()
        valid = shard["masks"]["target_valid"].float()

        first_ok = valid[:, 0] > 0
        if bool(first_ok.any()):
            pos_samples.append(target[first_ok, 0, :2].reshape(-1, 2))

        traj_ok = valid[:, :traj_steps].sum(dim=1) >= traj_steps
        if bool(traj_ok.any()):
            traj_frag = target[traj_ok, :traj_steps, :2].reshape(-1, traj_steps * 2)
            traj_samples.append(traj_frag)

    if pos_samples:
        pos_data = torch.cat(pos_samples, dim=0)
        pos_data = _balanced_subsample(pos_data, max_samples=max_samples)
    else:
        pos_data = torch.zeros((1, 2), dtype=torch.float32)

    if traj_samples:
        traj_data = torch.cat(traj_samples, dim=0)
        traj_data = _balanced_subsample(traj_data, max_samples=max_samples)
    else:
        traj_data = torch.zeros((1, traj_steps * 2), dtype=torch.float32)

    print(f"Building position token table from {int(pos_data.shape[0])} samples...")
    pos_centers = _run_kmeans(pos_data, k=pos_k, iters=TOKEN_KMEANS_ITERS)
    print(f"Building trajectory token table from {int(traj_data.shape[0])} samples...")
    traj_centers = _run_kmeans(traj_data, k=traj_k, iters=TOKEN_KMEANS_ITERS)

    return {
        "position_tokens": pos_centers,
        "trajectory_tokens": traj_centers.view(traj_k, traj_steps, 2),
    }


def nearest_token_indices(values: torch.Tensor, token_table: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    # values: [N, D], token_table: [K, D]
    out = []
    for start in range(0, values.shape[0], chunk_size):
        batch = values[start : start + chunk_size]
        dist = ((batch.unsqueeze(1) - token_table.unsqueeze(0)) ** 2).sum(dim=-1)
        out.append(dist.argmin(dim=1))
    return torch.cat(out, dim=0)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x * w).sum(dim=1) / denom


def _safe_key_padding_mask(valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # valid_mask: [B, T] where True means valid.
    pad_mask = ~valid_mask.bool()
    all_invalid = pad_mask.all(dim=1)
    if bool(all_invalid.any()):
        pad_mask = pad_mask.clone()
        pad_mask[all_invalid, 0] = False
    return pad_mask, all_invalid


class MapPointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(MAP_POINT_DIM, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.signal_mlp = nn.Sequential(
            nn.Linear(MAP_SIGNAL_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

    def forward(
        self,
        map_polyline: torch.Tensor,
        map_point_valid: torch.Tensor,
        map_signal: torch.Tensor,
    ) -> torch.Tensor:
        # map_polyline: [B, M, P, C]
        # map_point_valid: [B, M, P]
        points = self.point_mlp(map_polyline)
        pv = map_point_valid.unsqueeze(-1).to(points.dtype)
        pooled = (points * pv).sum(dim=2) / pv.sum(dim=2).clamp_min(1.0)
        signal = self.signal_mlp(map_signal)
        return torch.cat([pooled, signal], dim=-1)


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        hist_dim = 128
        agent_dim = 128
        map_dim = 128

        self.hist_in = nn.Sequential(
            nn.Linear(HIST_DIM, hist_dim),
            nn.GELU(),
            nn.Linear(hist_dim, hist_dim),
            nn.GELU(),
        )
        self.hist_gru = nn.GRU(hist_dim, hist_dim, batch_first=True)
        self.hist_token_proj = nn.Linear(hist_dim, cond_dim)

        self.nbr_in = nn.Sequential(
            nn.Linear(NBR_DIM, agent_dim),
            nn.GELU(),
            nn.Linear(agent_dim, agent_dim),
            nn.GELU(),
        )
        self.nbr_self_attn = nn.MultiheadAttention(agent_dim, num_heads=4, batch_first=True)
        self.nbr_cross_attn = nn.MultiheadAttention(agent_dim, num_heads=4, batch_first=True)

        self.map_encoder = MapPointEncoder()
        self.map_in = nn.Sequential(
            nn.Linear(96 + 32, map_dim),
            nn.GELU(),
            nn.Linear(map_dim, map_dim),
            nn.GELU(),
        )
        self.map_self_attn = nn.MultiheadAttention(map_dim, num_heads=4, batch_first=True)
        self.map_cross_attn = nn.MultiheadAttention(map_dim, num_heads=4, batch_first=True)

        self.map_scalar_mlp = nn.Sequential(
            nn.Linear(MAP_SCALAR_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(STATIC_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.query_hist_to_nbr = nn.Linear(hist_dim, agent_dim)
        self.query_nbr_to_map = nn.Linear(agent_dim, map_dim)
        self.nbr_token_proj = nn.Linear(agent_dim, cond_dim)
        self.map_token_proj = nn.Linear(map_dim, cond_dim)

        self.summary_proj = nn.Sequential(
            nn.Linear(hist_dim + agent_dim + map_dim + 64 + 64, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def _ensure_map_fields(
        self,
        cond: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = cond["hist"].shape[0]
        device = cond["hist"].device
        dtype = cond["hist"].dtype
        map_scalar = cond.get("map_scalar", cond.get("map", None))
        if map_scalar is None:
            map_scalar = torch.zeros((b, MAP_SCALAR_DIM), device=device, dtype=dtype)
        map_polyline = cond.get("map_polyline", None)
        if map_polyline is None:
            map_polyline = torch.zeros((b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS, MAP_POINT_DIM), device=device, dtype=dtype)
        map_point_valid = cond.get("map_point_valid", None)
        if map_point_valid is None:
            map_point_valid = torch.zeros((b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS), device=device, dtype=dtype)
        map_polyline_valid = cond.get("map_polyline_valid", cond.get("masks", {}).get("map_polyline_valid", None))
        if map_polyline_valid is None:
            map_polyline_valid = torch.zeros((b, MAP_POLYLINE_BUDGET), device=device, dtype=dtype)
        map_signal = cond.get("map_signal", None)
        if map_signal is None:
            map_signal = torch.zeros((b, MAP_POLYLINE_BUDGET, MAP_SIGNAL_DIM), device=device, dtype=dtype)
        return map_scalar, map_polyline, map_point_valid, map_polyline_valid, map_signal

    def forward(self, cond: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_valid = cond["masks"]["hist_valid"].float()
        nbr_valid = cond["masks"]["nbr_valid"].float()

        map_scalar, map_polyline, map_point_valid, map_polyline_valid, map_signal = self._ensure_map_fields(cond)

        hist_x = self.hist_in(cond["hist"])
        hist_out, _ = self.hist_gru(hist_x)
        hist_pool = _masked_mean(hist_out, hist_valid)
        hist_tokens = self.hist_token_proj(hist_out)

        nbr_x = self.nbr_in(cond["nbr"])
        nbr_pad_mask, nbr_all_invalid = _safe_key_padding_mask(nbr_valid > 0.5)
        nbr_attn, _ = self.nbr_self_attn(
            nbr_x,
            nbr_x,
            nbr_x,
            key_padding_mask=nbr_pad_mask,
            need_weights=False,
        )
        if bool(nbr_all_invalid.any()):
            nbr_attn = nbr_attn.clone()
            nbr_attn[nbr_all_invalid] = 0.0
        nbr_query = self.query_hist_to_nbr(hist_pool).unsqueeze(1)
        nbr_ctx, _ = self.nbr_cross_attn(
            nbr_query,
            nbr_attn,
            nbr_attn,
            key_padding_mask=nbr_pad_mask,
            need_weights=False,
        )
        nbr_ctx = nbr_ctx.squeeze(1)
        nbr_ctx[nbr_all_invalid] = 0.0
        nbr_tokens = self.nbr_token_proj(nbr_attn)

        map_raw = self.map_encoder(
            map_polyline=map_polyline,
            map_point_valid=map_point_valid,
            map_signal=map_signal,
        )
        map_x = self.map_in(map_raw)
        map_pad_mask, map_all_invalid = _safe_key_padding_mask(map_polyline_valid > 0.5)
        map_attn, _ = self.map_self_attn(
            map_x,
            map_x,
            map_x,
            key_padding_mask=map_pad_mask,
            need_weights=False,
        )
        if bool(map_all_invalid.any()):
            map_attn = map_attn.clone()
            map_attn[map_all_invalid] = 0.0
        map_query = self.query_nbr_to_map(nbr_ctx).unsqueeze(1)
        map_ctx, _ = self.map_cross_attn(
            map_query,
            map_attn,
            map_attn,
            key_padding_mask=map_pad_mask,
            need_weights=False,
        )
        map_ctx = map_ctx.squeeze(1)
        map_ctx[map_all_invalid] = 0.0
        map_tokens = self.map_token_proj(map_attn)

        static_emb = self.static_mlp(cond["static"])
        map_scalar_emb = self.map_scalar_mlp(map_scalar)

        summary = self.summary_proj(
            torch.cat([hist_pool, nbr_ctx, map_ctx, static_emb, map_scalar_emb], dim=-1)
        )

        cond_tokens = torch.cat([hist_tokens, nbr_tokens, map_tokens], dim=1)
        cond_mask = torch.cat(
            [
                (hist_valid > 0.5),
                (nbr_valid > 0.5),
                (map_polyline_valid > 0.5),
            ],
            dim=1,
        )
        return summary, cond_tokens, cond_mask


class DiTBlock(nn.Module):
    def __init__(self, dim: int = DIT_MODEL_DIM, heads: int = DIT_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.modulation = nn.Linear(dim, dim * 3)

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_mask: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        m1, m2, m3 = self.modulation(global_cond).chunk(3, dim=-1)

        h1 = self.norm1(x) + m1.unsqueeze(1)
        x = x + self.self_attn(h1, h1, h1, need_weights=False)[0]

        if cond_tokens.numel() > 0:
            h2 = self.norm2(x) + m2.unsqueeze(1)
            cond_pad_mask, cond_all_invalid = _safe_key_padding_mask(cond_mask > 0.5)
            cross = self.cross_attn(
                h2,
                cond_tokens,
                cond_tokens,
                key_padding_mask=cond_pad_mask,
                need_weights=False,
            )[0]
            if bool(cond_all_invalid.any()):
                cross = cross.clone()
                cross[cond_all_invalid] = 0.0
            x = x + cross

        h3 = self.norm3(x) + m3.unsqueeze(1)
        x = x + self.mlp(h3)
        return x


class ChunkDiffusionModel(nn.Module):
    def __init__(
        self,
        target_dim: int = TARGET_DIM,
        cond_dim: int = COND_DIM,
        model_dim: int = DIT_MODEL_DIM,
        depth: int = DIT_DEPTH,
        heads: int = DIT_HEADS,
        pos_vocab_size: int = 0,
        traj_vocab_size: int = 0,
    ):
        super().__init__()
        self.encoder = ConditionEncoder(cond_dim=cond_dim)
        self.target_dim = target_dim
        self.future_steps = target_dim // 4
        if self.future_steps <= 0 or (target_dim % 4) != 0:
            raise ValueError(f"target_dim must be divisible by 4, got {target_dim}")

        self.token_in = nn.Linear(4, model_dim)
        self.token_out = nn.Linear(model_dim, 4)
        self.pos_embed = nn.Parameter(torch.randn(1, self.future_steps, model_dim) * 0.02)
        self.cond_token_proj = nn.Linear(cond_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.global_cond_proj = nn.Sequential(
            nn.Linear(cond_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim=model_dim, heads=heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(model_dim)

        self.pos_head = nn.Linear(cond_dim, pos_vocab_size) if pos_vocab_size > 0 else None
        self.traj_head = nn.Linear(cond_dim, traj_vocab_size) if traj_vocab_size > 0 else None

    def encode_condition(self, cond: dict) -> torch.Tensor:
        summary, _, _ = self.encoder(cond)
        return summary

    def predict_token_logits(self, cond_emb: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pos_logits = self.pos_head(cond_emb) if self.pos_head is not None else None
        traj_logits = self.traj_head(cond_emb) if self.traj_head is not None else None
        return pos_logits, traj_logits

    def _predict_noise_from_encoded(
        self,
        x_noisy: torch.Tensor,
        t_norm: torch.Tensor,
        cond_emb: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = x_noisy.shape[0]
        t_emb = self.time_mlp(t_norm)
        global_cond = self.global_cond_proj(torch.cat([cond_emb, t_emb], dim=-1))

        x = x_noisy.view(b, self.future_steps, 4)
        x = self.token_in(x) + self.pos_embed[:, : self.future_steps, :]
        cond_tokens = self.cond_token_proj(cond_tokens)

        for block in self.blocks:
            x = block(x, cond_tokens, cond_mask, global_cond)

        x = self.token_out(self.final_norm(x))
        return x.reshape(b, self.target_dim)

    def forward(self, x_noisy: torch.Tensor, t_norm: torch.Tensor, cond: dict) -> torch.Tensor:
        cond_emb, cond_tokens, cond_mask = self.encoder(cond)
        return self._predict_noise_from_encoded(x_noisy, t_norm, cond_emb, cond_tokens, cond_mask)

    def forward_with_aux(
        self,
        x_noisy: torch.Tensor,
        t_norm: torch.Tensor,
        cond: dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        cond_emb, cond_tokens, cond_mask = self.encoder(cond)
        pred_noise = self._predict_noise_from_encoded(x_noisy, t_norm, cond_emb, cond_tokens, cond_mask)
        pos_logits, traj_logits = self.predict_token_logits(cond_emb)
        return pred_noise, pos_logits, traj_logits, cond_emb


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = EMA_DECAY) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))


@torch.no_grad()
def copy_model_params(src: nn.Module, dst: nn.Module) -> None:
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(p_src.data)


def _zero_cond_tensor_like(cond: dict, key: str, fallback_shape: tuple[int, ...]) -> torch.Tensor:
    if key in cond and isinstance(cond[key], torch.Tensor):
        return torch.zeros_like(cond[key])
    ref_tensor = None
    if isinstance(cond, dict):
        if "hist" in cond and isinstance(cond["hist"], torch.Tensor):
            ref_tensor = cond["hist"]
        else:
            for value in cond.values():
                if isinstance(value, torch.Tensor):
                    ref_tensor = value
                    break
    if ref_tensor is None:
        return torch.zeros(fallback_shape, dtype=torch.float32)
    device = ref_tensor.device
    dtype = ref_tensor.dtype
    return torch.zeros(fallback_shape, device=device, dtype=dtype)


def zero_condition_like(cond: dict) -> dict:
    b = cond["hist"].shape[0]
    return {
        "hist": torch.zeros_like(cond["hist"]),
        "nbr": torch.zeros_like(cond["nbr"]),
        "map_scalar": _zero_cond_tensor_like(cond, "map_scalar", (b, MAP_SCALAR_DIM)),
        "map": _zero_cond_tensor_like(cond, "map", (b, MAP_SCALAR_DIM)),
        "map_polyline": _zero_cond_tensor_like(cond, "map_polyline", (b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS, MAP_POINT_DIM)),
        "map_point_valid": _zero_cond_tensor_like(cond, "map_point_valid", (b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS)),
        "map_polyline_valid": _zero_cond_tensor_like(cond, "map_polyline_valid", (b, MAP_POLYLINE_BUDGET)),
        "map_signal": _zero_cond_tensor_like(cond, "map_signal", (b, MAP_POLYLINE_BUDGET, MAP_SIGNAL_DIM)),
        "static": torch.zeros_like(cond["static"]),
        "masks": {
            "hist_valid": torch.zeros_like(cond["masks"]["hist_valid"]),
            "target_valid": cond["masks"]["target_valid"],
            "nbr_valid": torch.zeros_like(cond["masks"]["nbr_valid"]),
            "map_valid": torch.zeros_like(cond["masks"]["map_valid"]),
            "map_polyline_valid": _zero_cond_tensor_like(cond["masks"], "map_polyline_valid", (b, MAP_POLYLINE_BUDGET)),
            "map_point_valid": _zero_cond_tensor_like(cond["masks"], "map_point_valid", (b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS)),
        },
    }


def apply_cfg_dropout(cond: dict, p_drop: float = CFG_DROPOUT) -> dict:
    b = cond["hist"].shape[0]
    keep = (torch.rand((b, 1), device=cond["hist"].device) > p_drop).float()
    keep_b = keep
    keep_bt = keep.unsqueeze(-1)
    keep_btp = keep.unsqueeze(-1).unsqueeze(-1)
    device = cond["hist"].device
    dtype = cond["hist"].dtype

    def _get_tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
        val = cond.get(key, None)
        if isinstance(val, torch.Tensor):
            return val
        return torch.zeros(shape, device=device, dtype=dtype)

    map_scalar = cond.get("map_scalar", cond.get("map", None))
    if map_scalar is None:
        map_scalar = torch.zeros((b, MAP_SCALAR_DIM), device=device, dtype=dtype)

    map_legacy = cond.get("map", cond.get("map_scalar", None))
    if map_legacy is None:
        map_legacy = torch.zeros((b, MAP_SCALAR_DIM), device=device, dtype=dtype)

    map_polyline = _get_tensor("map_polyline", (b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS, MAP_POINT_DIM))
    map_point_valid = _get_tensor("map_point_valid", (b, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS))
    map_polyline_valid = _get_tensor("map_polyline_valid", (b, MAP_POLYLINE_BUDGET))
    map_signal = _get_tensor("map_signal", (b, MAP_POLYLINE_BUDGET, MAP_SIGNAL_DIM))

    mask_map_polyline_valid = cond["masks"].get("map_polyline_valid", map_polyline_valid)
    mask_map_point_valid = cond["masks"].get("map_point_valid", map_point_valid)
    return {
        "hist": cond["hist"] * keep_bt,
        "nbr": cond["nbr"] * keep_bt,
        "map_scalar": map_scalar * keep_b,
        "map": map_legacy * keep_b,
        "map_polyline": map_polyline * keep_btp,
        "map_point_valid": map_point_valid * keep_btp.squeeze(-1),
        "map_polyline_valid": map_polyline_valid * keep_bt.squeeze(-1),
        "map_signal": map_signal * keep_bt,
        "static": cond["static"] * keep_b,
        "masks": {
            "hist_valid": cond["masks"]["hist_valid"] * keep_b,
            "target_valid": cond["masks"]["target_valid"],
            "nbr_valid": cond["masks"]["nbr_valid"] * keep_b,
            "map_valid": cond["masks"]["map_valid"] * keep_b,
            "map_polyline_valid": mask_map_polyline_valid * keep_b,
            "map_point_valid": mask_map_point_valid * keep_bt,
        },
    }


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: dict) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    sqrt_ab = schedule["sqrt_alphas_cumprod"][t].unsqueeze(-1)
    sqrt_omb = schedule["sqrt_one_minus_alphas_cumprod"][t].unsqueeze(-1)
    x_t = sqrt_ab * x0 + sqrt_omb * noise
    return x_t, noise


@torch.no_grad()
def _build_init_token_prior(model_for_sampling: nn.Module, cond: dict, diffusion_cfg: dict) -> torch.Tensor | None:
    pos_tokens = diffusion_cfg.get("position_tokens", None)
    traj_tokens = diffusion_cfg.get("trajectory_tokens", None)
    target_mean = diffusion_cfg["target_mean"].view(1, 1, 4)
    target_std = diffusion_cfg["target_std"].view(1, 1, 4)

    if pos_tokens is None or traj_tokens is None:
        return None
    if getattr(model_for_sampling, "pos_head", None) is None or getattr(model_for_sampling, "traj_head", None) is None:
        return None

    b = cond["hist"].shape[0]
    cond_emb = model_for_sampling.encode_condition(cond)
    pos_logits, traj_logits = model_for_sampling.predict_token_logits(cond_emb)
    if pos_logits is None or traj_logits is None:
        return None

    pos_tokens = pos_tokens.to(cond_emb.device)
    traj_tokens = traj_tokens.to(cond_emb.device)

    pos_idx = pos_logits.argmax(dim=-1)
    topk = min(16, traj_logits.shape[-1])
    topk_idx = traj_logits.topk(topk, dim=-1).indices

    recent_v = cond["hist"][:, -1, 2:4]
    traj_first = traj_tokens[:, 0, :]

    chosen_traj_idx = []
    for i in range(b):
        candidates = topk_idx[i]
        c_first = traj_first[candidates]
        dist = ((c_first - recent_v[i].unsqueeze(0)) ** 2).sum(dim=-1)
        chosen = candidates[dist.argmin()]
        chosen_traj_idx.append(chosen)
    chosen_traj_idx = torch.stack(chosen_traj_idx, dim=0)

    prior = torch.zeros((b, FUTURE_STEPS, 4), device=cond_emb.device)
    prior[:, 0, :2] = pos_tokens[pos_idx]

    k_steps = min(TRAJ_TOKEN_STEPS, FUTURE_STEPS)
    traj_path = traj_tokens[chosen_traj_idx][:, :k_steps, :]
    prior[:, :k_steps, :2] = 0.5 * prior[:, :k_steps, :2] + 0.5 * traj_path

    heading_xy = prior[:, :k_steps, :2].clone()
    heading_xy[:, 1:, :] = prior[:, 1:k_steps, :2] - prior[:, :k_steps - 1, :2]
    yaw = torch.atan2(heading_xy[..., 1], heading_xy[..., 0] + 1e-6)
    prior[:, :k_steps, 2] = torch.sin(yaw)
    prior[:, :k_steps, 3] = torch.cos(yaw)

    prior_norm = (prior - target_mean) / target_std.clamp_min(1e-6)
    return prior_norm.reshape(b, -1)


@torch.no_grad()
def sample_future_chunk(
    model: nn.Module,
    cond: dict,
    diffusion_cfg: dict,
    use_ema: bool = True,
    sample_steps: int | None = None,
    init_token_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    model_for_sampling = diffusion_cfg["ema_model"] if (use_ema and diffusion_cfg.get("ema_model") is not None) else model
    schedule = diffusion_cfg["schedule"]
    T = int(schedule["T"].item())
    sample_steps = int(sample_steps if sample_steps is not None else diffusion_cfg["sample_steps"])
    guidance_scale = diffusion_cfg["guidance_scale"]
    target_mean = diffusion_cfg["target_mean"].view(1, 1, 4)
    target_std = diffusion_cfg["target_std"].view(1, 1, 4)

    b = cond["hist"].shape[0]
    x = torch.randn((b, TARGET_DIM), device=cond["hist"].device)

    if init_token_prior is None:
        init_token_prior = _build_init_token_prior(model_for_sampling, cond, diffusion_cfg)
    if init_token_prior is not None:
        x = 0.85 * x + 0.15 * init_token_prior.to(x.device)

    time_grid = torch.linspace(T - 1, 0, sample_steps, device=cond["hist"].device).long()

    was_training = model_for_sampling.training
    model_for_sampling.eval()

    for i, t_tensor in enumerate(time_grid):
        t_idx = int(t_tensor.item())
        t_norm = torch.full((b, 1), t_idx / float(T), device=cond["hist"].device)

        eps_cond = model_for_sampling(x, t_norm, cond)
        eps_uncond = model_for_sampling(x, t_norm, zero_condition_like(cond))
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_bar_t = schedule["alphas_cumprod"][t_idx]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

        if i == (len(time_grid) - 1):
            x = x0_pred
            continue

        t_next = int(time_grid[i + 1].item())
        alpha_bar_next = schedule["alphas_cumprod"][t_next]
        x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps

    if was_training:
        model_for_sampling.train()

    x = x.view(b, FUTURE_STEPS, 4)
    return x * target_std + target_mean
