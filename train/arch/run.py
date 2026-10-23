"""Boost -> Cut with independent input-K/gate and future-token probes (#31), trained at batch 4 on two GPUs per
sequence with the backprop baseline's recipe, and merged at the end along its own trajectory.

The own-query and value-path measurements of PR #23 remain the starting point.
Input-K/gate probes supply additional input credit to linear consumers. Separate
single-source hidden/Q/embedding probes are scored at later, original measured
attention hubs. Corrected credits never become another probe's scoring target.

This record adds to #31 (nothing here changes what is measured or how):
  - Record #11's recipe: Cut at batch 4, each sequence split over two GPUs (each takes half of every cycle of moves
    and draws; the pair adds its parts, equal to one GPU), 0.7x rates, Adam betas (0.5, 0.98), a warmdown over the
    last 40% that ends at 30% of each rate (the end merge anneals the rest), and the LM head initialized at std 0.02.
  - Future probes: 128 central pairs per source token (#31: 512), in larger token chunks.
  - An end merge along the trajectory. The last 32 training sequences feed no update: they only score the merge. From
    60% of the epoch, every 5 updates, nine exponential means of the weights are kept (weight 0.5 ... 0.01 on the
    newest). After the last update, for each part (each block matrix, each table, the scalars) a candidate moves the
    final weights toward one mean, in full inside the part's 64 top input-covariance directions (for a table, the rows
    of the 512 most frequent tokens; both from forward passes on the held-out rows) and a fraction 0, 1/2 or 1 outside,
    times 0.75, 1 or 1.25: 81 candidates and no move. The best single candidate for every part comes first, then two
    coordinate sweeps over the parts (the second over each part's best 8) keep any change that lowers the held-out
    loss. About 5,000 forward passes on 32 sequences (~1 min), inside the timed region.

This recipe uses 20% Boost (global batch 4), then Cut (global batch 4), consuming
every training sequence but the last 32 once (those score the end merge). Future probes use all
permitted receivers and the first four hidden/Q layers plus the embedding.
No receiver sampling or active-space direction encoding is used.

    uv run torchrun --standalone --nproc_per_node=8 train/arch/run.py
"""
import argparse
from dataclasses import asdict, dataclass, field
from functools import cache, partial
from math import gcd
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Track 3: the model is this folder's model.py, which entries may change; utils/eval.py scores it with --model.
MODEL_FILE = Path(__file__).resolve().with_name('model.py')
from model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate

SEED = 0
EPS = torch.finfo(torch.float32).eps
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


def model_config():
    """The fixed GPT topology used by the current val experiment."""
    return GPTConfig(sequence_len=2048, vocab_size=4096, n_layer=8, n_head=8,
                     n_kv_head=8, n_embd=512, window_pattern='SSSL')


@dataclass
class Settings:
    """Own-query/value-path credit, full hidden FD, and optional independent input-K probes."""
    model: GPTConfig = field(default_factory=model_config)
    estimator: str = 'cut'
    boost_hub: str = field(default='projected', init=False)  # persisted algorithm version, not an ablation flag
    mlp_output_credit: str = field(default='next_mid', init=False)
    switch: float = 0.2
    boost_batch: int = 4
    cut_batch: int = 4
    projection_std: float = 1.5
    lr: float = 0.0021
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.4
    final: float = 0.3  # the warmdown ends at this fraction of each rate; the end merge anneals the rest
    head_std: float = 0.02  # the LM head's init std (the fixed init's 0.001)
    anvil_lr: float = 0.026
    anvil_wd: float = 2.25
    fast_weight: float = 1.0  # ANVIL fast rail
    polar_margin: float = 1.1
    sigma: float = 0.005
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    query_central: bool = False  # same directions, paired +/- attention evaluations
    nrep: int = 64
    emb_chunk: int = 16
    scalar_step: float = 0.01
    chunk: int = 2 ** 27
    chain_sigma: float = 0.01
    route_k: int = 512
    chain_tokens: int = 512
    chain_precision: str = 'bf16'
    own_query: bool = True  # PR #19's measured own-query paths in later attention blocks
    value_path: bool = True  # PR #23: one RMSNorm step, then only linear writers/V/VE
    path_scalars: bool = True  # also contract input-path credit into the linear mixing coefficients
    key_path_draws: int = 512  # independent input-K central pairs per token
    key_path_sigma: float = 0.01  # relative input RMS step
    key_path_tokens: int = 64  # source-token chunk; every allowed query contributes
    gate_path: bool = True  # independent full-cycle input probes through the VE gate
    gate_path_sigma: float = 0.01  # relative input RMS step
    future_draws: int = 128  # independent hidden/Q/WTE probes scored at every later raw hub
    future_layers: int = 4  # first blocks whose hidden and Q outputs receive this correction
    future_tokens: int = 128
    future_chunk: int = 512
    future_sigma: float = 0.01  # source-relative RMS; Q includes its own complete MLP response
    future_moments: bool = True  # score future probes with degree-2 moments over the receiving queries
    future_routed: bool = True  # future Q moves through the chain's routed keys (query_keys), not every key
    future_stacked: bool = True  # score every receiver of a source chunk at once (with future_moments)
    query_keys: int = 256   # moving key logits per head; 0 reruns all allowed logits
    key_sigma: float = 0.003
    key_chunk: int = 32
    key_moments: bool = True  # score input-K probes with degree-2 moments over the receiving queries
    own_query_precision: str = 'bf16'  # the chain's own-query move products (the record: 'fp32', TF32 products)

    def __post_init__(self):
        width, hidden = self.model.n_embd, 4 * self.model.n_embd
        if self.estimator not in ('boost', 'cut') or not 0 <= self.switch <= 1:
            raise ValueError('Invalid stage estimator or Boost fraction')
        if not 0 <= self.fast_weight <= 1:
            raise ValueError('ANVIL fast weight must be in [0,1]')
        if width < 1 or width & (width - 1):
            raise ValueError('Full cycles require power-of-two model width')
        if not 1 <= self.route_k <= hidden or self.chain_precision not in ('bf16', 'fp32'):
            raise ValueError('Invalid sparse route budget or chain precision')
        if self.query_draws != width // self.model.n_head:
            raise ValueError('Queries use one complete cycle per head')
        if not 0 < self.query_sigma < float('inf'):
            raise ValueError('Query step must be finite and positive')
        if not 0 < self.gate_path_sigma < float('inf') or (self.gate_path and self.estimator != 'cut'):
            raise ValueError('Gate input probes require Cut and a finite positive step')
        if (type(self.future_draws) is not int or not 0 <= self.future_draws <= width
                or (self.future_draws and self.estimator != 'cut')):
            raise ValueError('Future probes require Cut and a budget no greater than model width')
        if (type(self.future_layers) is not int or self.future_layers < 1
                or (self.future_draws and self.future_layers > self.model.n_layer)
                or type(self.future_tokens) is not int or self.future_tokens < 1
                or type(self.future_chunk) is not int or self.future_chunk < 1
                or not 0 < self.future_sigma < float('inf')):
            raise ValueError('Invalid future source layers, batching limits or finite step')
        if not isinstance(self.query_keys, int) or self.query_keys < 0:
            raise ValueError('Own-query key count must be a nonnegative integer')
        if type(self.key_path_draws) is not int or not 0 <= self.key_path_draws <= width:
            raise ValueError('Input-K pairs must be an integer between zero and model width')
        if (type(self.key_path_tokens) is not int or self.key_path_tokens < 1
                or not 0 < self.key_path_sigma < float('inf')):
            raise ValueError('Input-K chunk and finite step must be positive')
        if any(v <= 0 for v in (self.nrep, self.emb_chunk, self.chain_tokens, self.key_chunk, self.chunk,
                               self.sigma, self.fc_sigma, self.chain_sigma, self.key_sigma, self.scalar_step)):
            raise ValueError('Probe sizes and batching limits must be positive')

    @property
    def chain_dtype(self):
        return torch.bfloat16 if self.chain_precision == 'bf16' else torch.float32


# ---------------------------------------------------------------------------------------------------------- helpers


@torch.compile
def coordinate_differences(raw, y, sigma):
    """Finite differences of each token's loss along each of its own logits: for every token and every logit j,
    the loss with raw logit j moved up by sigma minus the loss with it moved down by sigma, over 2 sigma.

    Each of the two is a full forward evaluation of the soft-capped cross-entropy at the moved logit, written in
    closed form: moving one logit changes the partition function Z = sum_k exp(z_k) by one term, so the moved
    loss differs from the clean loss by log(Z - exp(z_j) + exp(z_j')) - log Z = log1p(exp(z_j) / Z * expm1(z_j' - z_j))
    in the partition and by -(z_j' - z_j) at the target logit. Nothing is differentiated: exp(z_j) / Z are the
    clean forward's probabilities, used only to evaluate the perturbed losses. Zero at ignored tokens (y < 0)."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)  # exp(z_j) / Z of the clean forward
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))  # loss(z_j -> new) - loss(z), partition
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))  # the target logit's own term
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1, built once per process."""
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)  # +-1 is exact in bfloat16
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


def attention_state(qraw, kraw, values, cos, sin, nhead, window):
    """Dense clean forward context for independently scored local attention trials."""
    def heads(raw):
        shaped = raw.reshape(*raw.shape[:2], nhead, -1)
        return norm(apply_rotary_emb(shaped, cos, sin)).transpose(1, 2)
    q, k = heads(qraw), heads(kraw)
    pos = torch.arange(qraw.shape[1], device=qraw.device)
    lag = pos[:, None] - pos[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -.5).masked_fill(
        ~allowed, -torch.inf).softmax(-1)
    return q, k, a, a @ values


def chunk_size(draws, cap):
    """Largest divisor of draws up to cap."""
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    """The attention each token would see if only its own state moved: every query attends to the other positions'
    clean keys and values and to its own moved ones, by swapping its own clean key and value through the
    log-sum-exp of one attention over the clean keys and values. Tensors are [batch, token, head, dim]."""
    if not q.is_cuda:
        t = q.shape[1]
        lag = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
        logits = q.transpose(1, 2) @ k_clean.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** .5
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), -torch.inf)
        own = (q * k_own).sum(-1).transpose(1, 2) / q.shape[-1] ** .5
        logits = logits.diagonal_scatter(own, dim1=-2, dim2=-1)
        weights = logits.softmax(-1)
        out = (weights @ v_clean.transpose(1, 2)).transpose(1, 2)
        return out + weights.diagonal(dim1=-2, dim2=-1).transpose(1, 2).unsqueeze(-1) * (v_own-v_clean)
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = flex(q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask,
                    return_lse=True)
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)

frozen_attention_compiled = torch.compile(frozen_attention)

@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)

@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)

@torch.compile(dynamic=False)
def query_probe_heads(raw, a, sigma, cos, sin):
    """Preserve the original finite norm calculation, then group probe heads."""
    moved = moved_heads(raw, a, sigma, cos, sin)
    batch, length, heads, width = raw.shape
    if raw.is_cuda and batch > 1:
        return moved.view(len(a), batch, length, heads, width).permute(1, 3, 0, 2, 4).reshape(
            batch, heads*len(a), length, width)
    return moved


def query_probe_attention(moved, keys, values, window, count, batch):
    """All probes share clean K/V; group probe heads instead of copying K/V."""
    _, length, heads, width = keys.shape
    if not moved.is_cuda or batch == 1:
        repeat = lambda z: z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*batch, length, heads, width)
        return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)
    out = flex(moved, keys.transpose(1, 2), values.transpose(1, 2),
               block_mask=window_mask(window, length, moved.device), enable_gqa=True)
    return out.view(batch, heads, count, length, width).permute(2, 0, 3, 1, 4)

@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    """Score either moved-minus-clean or plus-minus-minus by ordinary broadcasting."""
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)

@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores, directions):
    """Sum scalar probe scores times directions; the caller owns budget normalization.

    scores [..., P], directions [..., P, D] (or shared [P, D]).
    Keep this reduction inside the caller's compiled kernel, without creating
    a materialized score-times-direction tensor.
    """
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def decode_local_response(scores, steps, directions, central=True):
    """Decode independent trials; one-sided scores must already subtract clean."""
    if central:
        probes = directions.shape[-2]
        directional = (scores[..., :probes] - scores[..., probes:]) / (2 * steps)
    else:
        directional = scores / steps
    return decode_local_scores(directional, directions)


@dataclass(frozen=True)
class LocalExperiment:
    """A prepared forward response and its fixed-credit scalar score.

    response(rows, steps) returns concatenated +/- finite responses (central),
    or moved-minus-clean responses (one-sided). score(response) returns
    [..., 2P] or [..., P] independent scalar scores. No gradient or
    recursive credit propagation is implicit in this interface. Exact linear
    and separable local objectives can keep their specialized collapsed form.
    """
    response: Callable
    score: Callable
    central: bool = True

    def __call__(self, rows, steps):
        return decode_local_response(self.score(self.response(rows, steps)), steps, rows, self.central)

@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    """Collapse the full Hadamard central-difference cycle for the separable hidden objective.

    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's
    scalar finite difference. Keep that finite step, including ReLU crossings;
    contract in FP32 instead of rounding every direction's projected response to BF16.
    u and out_weight contain all units (or a GPU's contiguous slice).
    """
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope

@torch.compile(dynamic=False)
def embedding_moves(e, steps, rows, sign, wq, wk, wv, cos, sin):
    n, (b, t, width) = rows.shape[0], e.shape
    hd = 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + sign * steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)
    heads = lambda z: z.view(n * b, t, -1, hd)
    q = norm(apply_rotary_emb(heads(F.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(F.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(F.linear(x0, wv))

@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)

@torch.compile(dynamic=False)
def stage_readout(x0, out, mix, wproj, weight, y):
    xmid = mix * x0 + F.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)

def norm_difference(x, delta):
    """norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves."""
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / x.shape[-1] + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    """PR #23's central forward RMSNorm response along v.

    RMSNorm's Jacobian is symmetric: this finite directional response tends
    to its credit pullback. This is one nonlinear step, not an analytic VJP.
    The tiny floor also defines a finite response for a zero input or credit.
    """
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
                 / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    """Token chunks share the direction projections and squared lengths."""
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ rows.T) / x.shape[-1]
    direction_square = row_square.unsqueeze(0)
    base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
    direction = projected_rows.unsqueeze(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * direction_square + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * direction_square + EPS)
    zp = (base + steps.unsqueeze(-1) * direction) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * direction) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
    return decode_local_scores(response, rows) / len(rows)


def query_change(x, delta, pos, rel, w_q, keys, values, w_proj_t, cos, sin, window):
    """PR #19's finite own-query response; all keys and values stay at their clean values.

    Carry finite changes through both RMSNorms and softmax. The difference form
    preserves small probe responses without replacing them by a local derivative.
    """
    m, probes, _ = delta.shape
    heads, length, width = keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate(move_product(norm_difference(x, delta), w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    q0, dq = r0 * inv0, dr * inv + r0 * (inv - inv0)
    lag = rel[:, None] - torch.arange(length, device=rel.device)[None]
    allowed = ((lag >= 0) & (lag <= window))[:, None, None]
    scores = torch.einsum('mphd,htd->mpht', q0.to(keys.dtype), keys).float() * width ** -0.5
    p0 = torch.softmax(scores.masked_fill(~allowed, float('-inf')), -1)
    ds = torch.einsum('mphd,htd->mpht', dq.to(keys.dtype), keys).float() * width ** -0.5
    dlse = torch.log1p((p0 * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mpht,htd->mphd', (p0 * torch.expm1(ds - dlse)).to(values.dtype), values).float()
    return move_product(dout.flatten(-2), w_proj_t)


def move_product(move, weight):
    """A replayed move times a clean weight, in MOVE_DTYPE with an fp32 result."""
    return (move.to(MOVE_DTYPE) @ weight.to(MOVE_DTYPE)).float()


@cache
def finite_attention_kernels():
    """Finite-response reductions. Neither kernel differentiates an attention operation."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def routed(Q, K, V, P, Rest, Out,
               Probes: tl.constexpr, Heads: tl.constexpr, Keys: tl.constexpr, D: tl.constexpr,
               BP: tl.constexpr, BK: tl.constexpr):
        token, head = tl.program_id(1) // Heads, tl.program_id(1) % Heads
        probes = tl.program_id(0) * BP + tl.arange(0, BP)
        keys, features = tl.arange(0, BK), tl.arange(0, D)
        q = tl.load(Q + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                    probes[:, None] < Probes, 0)
        offsets = ((token * Heads + head) * Keys + keys[:, None]) * D + features[None, :]
        k = tl.load(K + offsets, keys[:, None] < Keys, 0)
        v = tl.load(V + offsets, keys[:, None] < Keys, 0)
        p = tl.load(P + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        # Match both BF16 matmul output roundings of the original implementation.
        ds = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * (D ** -.5)
        dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
        weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(Rest + (token * Heads + head) * D + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(Out + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                 moved, probes[:, None] < Probes)

    @triton.jit
    def key_response(Q, DK, A, C, Out,
                     H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                     QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                     AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                     CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                     SourceStart, QueryStart, Window: tl.constexpr,
                     BC: tl.constexpr, BQ: tl.constexpr):
        head = tl.program_id(1)
        col = tl.program_id(0) * BC + tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        changed = tl.load(DK + (head * P * M + col[None, :]) * D + features[:, None],
                          col[None, :] < P * M, 0)
        first = tl.maximum(0, SourceStart + tl.min(source, 0) - QueryStart)
        last = tl.minimum(T, SourceStart + tl.max(source, 0) + Window + 1 - QueryStart)
        first = first // BQ * BQ
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start + tl.arange(0, BQ)
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD,
                        query[:, None] < T, 0)
            delta = tl.dot(q, changed, input_precision='tf32x3') * (D ** -.5)
            valid = (query[:, None] < T) & (col[None, :] < P * M)
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
            change = a * libdevice.expm1(delta)
            result += tl.sum(change / (1 + change) * c, 0)
        tl.store(Out + head * P * M + col, result, col < P * M)

    return routed, key_response


@torch.library.custom_op('noise_record::routed_response', mutates_args=())
def routed_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    probabilities: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    import triton
    m, probes, heads, width = dq.shape
    count = keys.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device, dtype=torch.float32)
    finite_attention_kernels()[0][(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(), keys.contiguous(), values.contiguous(), probabilities.contiguous(), rest.contiguous(),
        out, probes, heads, count, width, 32, max(16, triton.next_power_of_2(count)), num_warps=4)
    return out


@routed_response.register_fake
def _(dq, keys, values, probabilities, rest):
    return torch.empty(dq.shape, device=dq.device, dtype=torch.float32)


@torch.library.custom_op('noise_record::key_response', mutates_args=())
def fused_key_response(q: torch.Tensor, dk: torch.Tensor, A: torch.Tensor, C: torch.Tensor,
                       source_start: int, query_start: int, window: int) -> torch.Tensor:
    import triton
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    finite_attention_kernels()[1][(triton.cdiv(probes * sources, 64), heads)](
        q, dk.contiguous(), A, C, out, heads, q.shape[1], sources, probes, width,
        *q.stride(), *A.stride(), *C.stride(), source_start, query_start, window,
        64, 64, num_warps=4, enable_fp_fusion=False)
    return out


@fused_key_response.register_fake
def _(q, dk, A, C, source_start, query_start, window):
    return torch.empty(dk.shape[:3], device=q.device, dtype=torch.float32)


def query_change_routed(x, delta, pos, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos, sin):
    """Move only the selected key logits; retain the exact rescaling of the held output."""
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate(move_product(norm_difference(x, delta), w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    return move_product(routed_output_change(dq, top_keys, top_values, top_p, rest).flatten(-2), w_proj_t)


def routed_output_change(dq, top_keys, top_values, top_p, rest):
    """The attention output's finite change [m, P, H, D] when the processed query moves by dq [m, P, H, D] and
    only its selected key logits move; the held remainder rescales exactly."""
    width = dq.shape[-1]
    if dq.is_cuda and top_keys.dtype == torch.bfloat16 and width in (32, 64) and top_keys.shape[2] <= 256:
        return routed_response(dq.to(top_keys.dtype), top_keys, top_values, top_p, rest)
    ds = torch.einsum('mphd,mhkd->mphk', dq.to(top_keys.dtype), top_keys).float() * width ** -0.5
    p = top_p.unsqueeze(1)
    dlse = torch.log1p((p * torch.expm1(ds)).sum(-1, keepdim=True))
    dout = torch.einsum('mphk,mhkd->mphd', (p * torch.expm1(ds - dlse)).to(top_values.dtype), top_values).float()
    return dout + torch.expm1(-dlse) * rest.unsqueeze(1)


@cache
def readout_kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(Delta, W, Base, Inv, Out, R, V, Probes: tl.constexpr, K: tl.constexpr,
               BR: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0)*BR+tl.arange(0, BR)
        live = rows < R
        token = rows // Probes
        inv = tl.load(Inv+rows, live, 0.)
        total = tl.zeros((BR,), tl.float32)
        for v0 in range(0, V, BV):
            cols = v0+tl.arange(0, BV)
            acc = tl.zeros((BR, BV), tl.float32)
            for k0 in tl.static_range(0, K, BK):
                ks = k0+tl.arange(0, BK)
                d = tl.load(Delta+rows[:, None]*K+ks[None, :], live[:, None], 0.)
                w = tl.load(W+ks[:, None]*V+cols[None, :], cols[None, :] < V, 0.)
                acc = tl.dot(d, w, acc)
            base = tl.load(Base+token[:, None]*V+cols[None, :], live[:, None] & (cols[None, :] < V), 0.)
            z = (base+acc)*inv[:, None]
            capped = 15*libdevice.tanh(z/15)
            total += tl.sum(tl.where(cols[None, :] < V, tl.exp(capped), 0.), 1)
        tl.store(Out+rows, tl.log(total), live)
    return kernel


@torch.library.custom_op('noise_record::readout_lse', mutates_args=())
def readout_lse(delta: torch.Tensor, weight_t: torch.Tensor, base: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """log sum_v exp(15 tanh(z_v / 15)), z = (base + delta @ weight_t) * inv, for delta [m, P, W] (bf16), weight_t
    [W, V] (bf16), base [m, V], inv [m, P]: the chain's moved readout without writing the [m, P, V] logits."""
    m, probes, width = delta.shape
    if not delta.is_cuda:
        z = (base.unsqueeze(1) + (delta @ weight_t).float()) * inv.unsqueeze(-1)
        return torch.exp(15 * torch.tanh(z / 15)).sum(-1).log()
    import triton
    rows, vocab = m * probes, weight_t.shape[1]
    out = torch.empty(rows, device=delta.device, dtype=torch.float32)
    readout_kernel()[(triton.cdiv(rows, 64),)](
        delta.contiguous(), weight_t.contiguous(), base.float().contiguous(), inv.float().contiguous(), out,
        rows, vocab, probes, width, 64, 128, 64, num_warps=8, num_stages=3)
    return out.view(m, probes)


@readout_lse.register_fake
def _(delta, weight_t, base, inv):
    return torch.empty(delta.shape[:2], device=delta.device, dtype=torch.float32)


def chain_chunk_eager(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                      target_logit_w, y, own=None, from_output=False, packed=False):
    """Forward differences of each token's loss along every move of one site, K/V held, MLPs restricted to
    each token's top-k units. site: -1 for the raw embedding, else the block whose mid stream moves. move (m, P, W);
    scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block (m, W); idx, hs per block
    (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block (H, W); weight_t (W, V);
    base (m, V) clean top-stream logits before the norm; target_logit_w (m, W).
    own optionally supplies PR #19's own-query responses in every entered attention.
    from_output starts at the block's MLP output instead: skip its MLP, then enter
    the next block through BOTH its residual scaling and its own-query response.
    Returns (m, P)."""
    L = len(mids)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site + int(from_output)
    for b in range(first, L):
        if b > first or from_output:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        if own is not None and (site < 0 or b > first or from_output):
            if own[0] == 'routed':
                _, pos, inputs, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos_sin = own
                delta = delta + query_change_routed(inputs[b], delta, pos, w_q[b], top_keys[b], top_values[b],
                                                    top_p[b], rest[b], w_proj_t[b], cos_sin[0], cos_sin[1])
            else:
                _, pos, rels, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
                delta = delta + query_change(inputs[b], delta, pos, rels[b], w_q[b], keys[b], values[b],
                                             w_proj_t[b], cos, sin, windows[b])
        dz = norm_difference(mids[b], delta)
        fc = w_fc[b] if packed else w_fc[b][idx[b]]
        wo = w_out_t[b] if packed else w_out_t[b][idx[b]]
        dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + EPS)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = readout_lse(delta.to(torch.bfloat16), weight_t, base, inv) - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)
chain_chunk_own = torch.compile(chain_chunk_eager, dynamic=True)
KEY_BUCKET = 256
MOVE_DTYPE = torch.bfloat16  # the own-query move products of the chain replay (Settings.own_query_precision)


@torch.compile(dynamic=False)
def key_scores(qp, kp, k_raw, steps, cos, sin, A, C, basis):
    """Single-key experiments, each evaluated exactly: the raw key at position t moves by steps_t e_j and nothing
    else moves, so each query's attention changes in one term, o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'),
    a' = a exp(dl). Scored against each query's target (C[s, t] = hub_s . (v_t - o_s)): one local step, as the
    record's coordinate head scores single-logit moves. The processed key's change is formed exactly before the
    logit product. qp, kp, k_raw (h, T, d); steps (h, T, 1); cos, sin (T, d/2); A, C (h, T, T), A zero outside the
    window; basis (J, d). Returns (h, T, J)."""
    d = k_raw.shape[-1]
    moved = k_raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    x1, x2 = moved[..., :d // 2], moved[..., d // 2:]
    moved = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    dk = moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)
    dl = torch.einsum('hsd,hjtd->hjst', qp, dk) / d ** 0.5
    change = A[:, None] * torch.expm1(dl)
    return (change / (1 + change) * C[:, None]).sum(2).transpose(1, 2) / steps


@torch.compile(dynamic=False)
def raw_key_moves(kp, raw, steps, cos, sin, basis):
    moved = raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    half = raw.shape[-1] // 2
    moved = torch.cat((moved[..., :half] * cos + moved[..., half:] * sin,
                       -moved[..., :half] * sin + moved[..., half:] * cos), -1)
    return moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)


@torch.compile(dynamic=False)
def prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin):
    """Same central experiments, reusing the linear projections of x and each direction.

    N(x +/- s*h) W - N(x) W =
        (inv_plusminus - inv0) * (x W) +/- s*inv_plusminus * (h W).
    This is a finite forward identity, including both RMSNorm responses.
    Positive and negative trials share one batched query/key product.
    """
    m, probes, width = x.shape[0], rows.shape[0], x.shape[-1]
    heads, _, dh = k_raw.shape
    square = x.square().sum(-1, keepdim=True)
    dots = x @ rows.T
    moved_square = square + steps.square() * rows.square().sum(-1)
    inv0 = torch.rsqrt(square / width + EPS)
    inv_up = torch.rsqrt((moved_square + 2 * steps * dots) / width + EPS)
    inv_down = torch.rsqrt((moved_square - 2 * steps * dots) / width + EPS)
    up = ((inv_up - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
          + (steps * inv_up).unsqueeze(-1) * projected_rows.unsqueeze(0))
    down = ((inv_down - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
            - (steps * inv_down).unsqueeze(-1) * projected_rows.unsqueeze(0))
    raw_change = torch.cat((up, down), dim=1).reshape(m, 2 * probes, heads, dh).permute(2, 1, 0, 3)
    rotate = lambda z: torch.cat([z[..., :dh//2]*cos + z[..., dh//2:]*sin,
                                  -z[..., :dh//2]*sin + z[..., dh//2:]*cos], -1)
    r0 = rotate(k_raw)
    dr = rotate(raw_change)
    dk = norm_difference(r0.reshape(heads*m, dh),
                         dr.permute(0, 2, 1, 3).reshape(heads*m, 2 * probes, dh))
    return dk.reshape(heads, m, 2 * probes, dh).permute(0, 2, 1, 3)


@torch.compile(dynamic=False)
def key_input_prepared_derivatives(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin, qp, A, C):
    dk = prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin)
    probes, dh = rows.shape[0], k_raw.shape[-1]
    dl = torch.einsum('hsd,hpmd->hpsm', qp, dk) / dh**.5
    change = A[:, None] * torch.expm1(dl)
    scores = (change / (1 + change) * C[:, None]).sum((0, 2)).T
    return (scores[:, :probes] - scores[:, probes:]) / (2 * steps)


def key_moments(qp, A, C):
    """Degree-2 moments over each source token's queries for keys alone (future_moments with no value change; A
    holds the queries that see each source, the source's own query included): key_moment_scores is the key
    kernel's change / (1 + change) * c to third order in the logit change."""
    dh = qp.shape[-1]
    ac = A*C
    mk = torch.einsum('hst,hsd->htd', ac, qp)*dh**-.5
    qq = (qp[..., :, None]*qp[..., None, :]).flatten(-2)
    mkk = torch.bmm((ac*(.5-A)).transpose(1, 2), qq).view(*mk.shape, dh)/dh
    return mk, mkk


@torch.compile(dynamic=False)
def key_moment_scores(dk, mk, mkk):
    """Summed key scores [H, P, M] of key changes dk [H, P, M, D] from the moments of their tokens M."""
    return (torch.einsum('hpmd,hmde->hpme', dk, mkk)*dk+dk*mk[:, None]).sum(-1)


@torch.compile(dynamic=False)
def decode_key_pairs(scores, steps, rows):
    return decode_local_response(scores.sum(0).T, steps, rows) / len(rows)


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames = {}
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.capturing = False
        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(layer)))

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output

        return hook

    def block_hook(self, layer):
        def hook(module, args):
            if self.capturing:
                self.blocks[layer] = args[0]

        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def signs(self, *shape, shared=False):
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws start..start + n of this step's world * draws at `site`: rows of a sign-flipped Hadamard matrix,
        orthogonal across chunks and GPUs within each cycle."""
        if site not in self.frames:
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
            # Track 3: the hooks kept the second pass, whose embedding is wte + n-grams + feedback(h[t-1]),
            # recomputed as the forward did; every embedding probe moves that sum, and h[t-1] (the first
            # pass) is a fixed input, like the token
            self.outputs['transformer.wte'] = self.model.embed(x, self.inputs.get('feedback'))
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma=0.1):
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def embedding_gradients(self, gradients, credit, sequence=None):
        """Add the gradients of everything summed into the embedding, from the credit there ([batch, token,
        width], or sequence `sequence`'s [token, width]): token and n-gram rows index it, feedback multiplies it
        by h[t-1]. None of them takes a step through a nonlinearity."""
        def inputs(site):
            return self.inputs[site] if sequence is None else self.inputs[site][sequence]

        def grad(site):
            if site + '.weight' not in gradients:
                gradients[site + '.weight'] = torch.zeros_like(self.modules[site].weight, dtype=torch.float32)
            return gradients[site + '.weight']

        credit = credit.reshape(-1, credit.shape[-1]).float()
        grad('transformer.wte').index_add_(0, inputs('transformer.wte').reshape(-1), credit)
        if 'ngram.table' in self.inputs:  # each order's heads hold the width side by side
            ids = inputs('ngram.table').reshape(credit.shape[0], -1)
            error = credit.unsqueeze(1).expand(-1, self.model.ngram.orders, -1)
            grad('ngram.table').index_add_(0, ids.reshape(-1), error.reshape(ids.numel(), -1))
        if 'feedback' in self.inputs:
            grad('feedback').addmm_(credit.T, inputs('feedback').reshape(credit.shape[0], -1))

class Boost:
    """Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        if model.config.n_head != model.config.n_kv_head or model.config.n_layer % 2:
            raise ValueError('Stage estimators require equal Q/KV heads and an even number of blocks')
        if any(k % split for k in (model.config.n_embd, settings.query_draws)):
            raise ValueError('Readout and query cycles must divide across the sequence group')
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.shared_generator = self.capture.shared_generator
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)
        self.maps = {}

    def close(self):
        self.capture.close()

    def share(self, total):
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor):
        if self.split > 1:
            torch.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def probe(self, x, y, mix=None, probed=None):
        """Each token's readout credit in the probed space, measured along a full cycle of the step's basis; the
        GPUs of a sequence each take a slice of the cycle and add their parts."""
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        offset = torch.randint(0, self.width, (n, 1), device=self.device, generator=self.generator)  # each token's
        # first row (used only by partial cycles; drawn either way so the random stream is the same)
        credit = torch.zeros(n, self.width, device=self.device)
        lo, hi = self.share(self.width)
        C, stream_dirs, dir_sq, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ Hs / self.width
        return self.group_sum(credit)

    def embedding_block0(self, e, y, x):
        """The embedding's credit for the first attention stage's readout along each token's own path (every query
        sees the other positions' clean keys and values and its own moved ones)."""
        model = self.model
        cfg = model.config
        attn = model.transformer.h[0].attn
        b, t = x.shape
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        window = model.window_sizes[0][0]
        mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
        weight = model.lm_head.weight.float()[:cfg.vocab_size]
        W = {name: getattr(attn, 'c_' + name).weight.float() for name in ('q', 'k', 'v', 'proj')}
        e = e.float().view(b, t, self.width)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, self.width, device=self.device)
        n = chunk_size(self.width // self.split, self.s.emb_chunk)
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, self.width).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, self.width), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(self.width)
        for j0 in range(lo, hi, n):
            rows = H[j0:j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / self.width).reshape(b * t, self.width)

    def attention_maps(self, i):
        """Clean attention A[b, h, query, source]; no matched-filter matrix is needed."""
        if i in self.maps:
            return self.maps[i]
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (norm(apply_rotary_emb(d.outputs[prefix + name].float().view(b, t, h, hd), cos, sin))
                for name in ('c_q', 'c_k'))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        window = model.window_sizes[i][0]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        A = logits.softmax(-1)
        del logits
        self.maps[i] = A.to(torch.bfloat16)
        return self.maps[i]

    def query_errors(self, draws, targets, layers=None):
        """A complete per-head query cycle, scored against each query's own output target."""
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        n = chunk = chunk_size(draws, self.s.nrep)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = None if self.s.query_central else attention(processed(q_raw).to(dt), kp.to(dt), values.to(dt), window)
            fixed, vals = kp.to(dt), values.to(dt)
            raw = q_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = query_probe_heads(raw, a, sigma, cos, sin)
                out = query_probe_attention(moved, fixed, vals, window, n, b)
                if self.s.query_central:
                    minus = query_probe_attention(
                        query_probe_heads(raw, a, -sigma, cos, sin), fixed, vals, window, n, b)
                    score = head_scores(out, minus, target)
                else:
                    score = head_scores(out, baseline, target)
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma * (2 if self.s.query_central else 1))).reshape(b, t, cfg.n_embd)
        return errors

    def key_context(self, i, sequence, target):
        """Clean FP32 single-key score context, shared by raw-K and input-K experiments."""
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd//2]*cos + z[..., hd//2:]*sin,
                                   -z[..., :hd//2]*sin + z[..., hd//2:]*cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix+name][sequence].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][sequence].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix+'ve_gate'][sequence].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        logits = (qp @ kp.transpose(1, 2)) / hd**.5
        A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
        hub = target[sequence].view(t, h, hd).transpose(0, 1)
        C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
        return qp, kp, k_raw, cos, sin, A, C

    def exact_key_errors(self, targets, layers=None):
        """Key credit, one local step from the targets, by single-key experiments (key_scores): exact up to the
        one-sided step, with no interference between positions. The GPUs of a sequence split the directions."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                credit_all[r] = self.key_errors_from_context(i, self.key_context(i, r, targets[i]))
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def key_errors_from_context(self, layer, context):
        """Raw-K finite experiments, sharing the input-K context and skipping masked query blocks."""
        qp, kp, raw, cos, sin, A, C = context
        h, t, hd = raw.shape
        lo, hi = self.share(hd)
        result = torch.zeros_like(raw)
        eye = torch.eye(hd, device=self.device)
        steps = self.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        for t0 in range(0, t, KEY_BUCKET):
            t1 = min(t0 + KEY_BUCKET, t)
            q1 = min(t, t1 + self.model.window_sizes[layer][0])
            source, query = slice(t0, t1), slice(t0, q1)
            if raw.is_cuda and hd in (32, 64):
                moves = raw_key_moves(kp[:, source], raw[:, source], steps[:, source],
                                      cos[source], sin[source], eye[lo:hi])
                scored = fused_key_response(qp[:, query], moves, A[:, query, source], C[:, query, source],
                                           t0, t0, self.model.window_sizes[layer][0])
                result[:, source, lo:hi] = scored.transpose(1, 2) / steps[:, source]
                continue
            for j0 in range(lo, hi, self.s.key_chunk):
                j1 = min(j0 + self.s.key_chunk, hi)
                result[:, source, j0:j1] = key_scores(
                    qp[:, query], kp[:, source], raw[:, source], steps[:, source],
                    cos[source], sin[source], A[:, query, source], C[:, query, source], eye[j0:j1])
        return result

    def value_credit(self, i, target):
        """Fixed-attention value credit; all query tokens contribute, with no discount."""
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        A = self.attention_maps(i)  # A[b, h, query, source]
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return (A.float().transpose(-1, -2) @ hub).transpose(1, 2)

    def exact_values(self, i, target, value_target=None, raw_credit=None):
        """Layer i's value and gate errors, the local step in closed form: a value move at t reaches query s's output
        as A[s, t] times the move, so the value credit is sum_s A[s, t] target_s; a gate move scales the value
        embedding by the gate's slope (the record's central difference, odd in the move)."""
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        credit = self.value_credit(i, target) if raw_credit is None else raw_credit
        # A corrected target has already crossed RMSNorm. Only V/VE may use it:
        # the gate's sigmoid experiment must retain the directly measured target.
        value_credit = credit if value_target is None else self.value_credit(i, value_target)
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """All hidden units' local central differences, partitioned across a sequence's GPUs."""
        # Preserve full-state checkpoint RNG advancement from the retired hidden cycle.
        self.signs(256)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        lo, hi = self.share(u.shape[-1])
        local = hidden_local_credit(u[:, lo:hi], steps, g, out_weight[:, lo:hi].float())
        return local if self.split == 1 else F.pad(local, (lo, u.shape[-1] - hi))

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def internals(self, gradients, targets, b, t, value_targets=None, key_errors=None, raw_values=None):
        """Q/K/gate use measured targets; only linear V/VE paths accept corrected targets."""
        d, s = self.capture, self.s
        errors = self.query_errors(s.query_draws // self.split, targets)
        errors = {site: error / self.split for site, error in errors.items()}
        errors.update(self.exact_key_errors(targets) if key_errors is None else key_errors)
        for i, target in targets.items():
            value_target = None if value_targets is None else value_targets[i]
            raw = None if raw_values is None else raw_values[i]
            errors.update({site: error / self.split
                           for site, error in self.exact_values(i, target, value_target, raw).items()})
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        x0 = d.x0
        targets, streams = {}, []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i]
            xmid = xin + d.outputs[p + 'attn.c_proj']
            xout = xmid + d.outputs[p + 'mlp.c_proj']
            streams.append((xin, xmid, xout))
            xm, xo = xmid.float().flatten(0, 1), xout.float().flatten(0, 1)
            block = model.transformer.h[i]
            att_in = d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            g_mid = self.probe(xm, yf)
            gradients[p + 'attn.c_proj.weight'] = g_mid.T @ att_in
            # One local linear step from the measured stream credit (PR #15).
            hub = g_mid @ block.attn.c_proj.weight.float()
            # Keep later probes on the same RNG stream as the retired hub cycle.
            self.signs(self.width)
            torch.randint(0, self.width, (xm.shape[0], 1), device=self.device, generator=self.generator)
            targets[i] = hub.view(b, t, self.width)
            g_out = self.probe(xo, yf)
            gradients[p + 'mlp.c_proj.weight'] = g_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, g_out, model.transformer.h[i].mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            if i < L - 1:  # the head learns at every intermediate readout too
                for stream in (xm, xo):
                    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + EPS)
                    error = coordinate_differences(normed @ weight.T, yf, 0.1)
                    gradients['lm_head.weight'][:weight.shape[0]] += error.T @ normed
        self.internals(gradients, targets, b, t)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        d.embedding_gradients(gradients, emb.view(b, t, self.width))
        scalars = torch.zeros(2, L, device=self.device)
        step = s.scalar_step
        for i in range(self.part, L, self.split):
            x_prev = x0 if i == 0 else streams[i - 1][2]
            ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
            lr_, l0 = model.resid_lambdas[i], model.x0_lambdas[i]
            for which in range(2):
                up = self.stage_losses(i, lr_ + step * (which == 0), l0 + step * (which == 1), x_prev, x0, ve, y)
                down = self.stage_losses(i, lr_ - step * (which == 0), l0 - step * (which == 1), x_prev, x0, ve, y)
                scalars[which, i] = (up - down).sum() / (2 * step)
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.part:  # computed in full by every GPU of the sequence: kept by its first GPU only
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight'):
                    gradients[name] = torch.zeros_like(gradients[name])
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

    def step(self, x, y, optimizer):
        gradients = self.estimate(x, y)
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        sequences = self.world // self.split
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if self.world > 1:
            torch.distributed.all_reduce(flat)
        flat /= sequences
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite estimate')
        offset = 0
        for name, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        self.maps.clear()
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()

class Cut(Boost):
    """Own-query forward loss probes, optionally augmented by PR #23's value path."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        # split > 1: the GPUs of a sequence each take a share of every cycle of moves and draws
        super().__init__(model, settings, seed, rank, world, split=split, group=group)

    def value_path_credit(self, chain, *, return_inputs=False, raw_values=None):
        """Independent measured mid credits -> V -> one RMSNorm step -> residual identities.

        Never feed accumulated extra credit into another norm or local nonlinear
        experiment. Reuse this capture's attention maps later for V/gate credit.
        """
        model, d = self.model, self.capture
        L = model.config.n_layer
        extra = {L - 1: torch.zeros_like(chain[0])}
        inputs = {}
        for i in reversed(range(0 if return_inputs else 1, L)):
            attn = model.transformer.h[i].attn
            ev = (self.value_credit(i, chain[i] @ attn.c_proj.weight.float())
                  if raw_values is None else raw_values[i]).reshape(-1, self.width)
            vp = norm_move(d.blocks[i].float().flatten(0, 1), ev @ attn.c_v.weight.float())
            inputs[i] = vp
            if i:
                extra[i - 1] = model.resid_lambdas[i].float() * (vp + extra[i])
        return (extra, inputs) if return_inputs else extra

    def key_input_credits(self, targets):
        """Independent central probes at each block input, moving only its K branch.

        Use a private copy of the saved probe RNG so these optional draws neither
        shift Q/hidden probes nor require another checkpoint RNG stream.
        Partial Hadamard cycles use randomized rows and column signs.
        """
        d, s = self.capture, self.s
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        raw_errors = {}
        for i, target in targets.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)[:s.key_path_draws]
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator)*2-1
            rows = frame[indices] * signs
            total_rows = len(rows)
            r0, r1 = self.share(total_rows)  # this GPU's share of the draws
            rows = rows[r0:r1]
            estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
            raw_credit = torch.empty_like(estimate)
            w_key = self.model.transformer.h[i].attn.c_k.weight.float()
            projected_rows = rows @ w_key.T
            for r in range(d.batch):
                context = self.key_context(i, r, target)
                qp, _, raw, cos, sin, A, C = context
                raw_credit[r] = self.key_errors_from_context(i, context).transpose(0, 1).reshape(d.length, self.width)
                moments = key_moments(qp, A, C) if s.key_moments and self.device.type == 'cuda' else None
                xin = d.blocks[i][r].float()
                projected_x = xin @ w_key.T
                steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                for t0 in range(0, d.length, s.key_path_tokens):
                    t1 = min(t0+s.key_path_tokens, d.length)
                    sl = slice(t0, t1)
                    # Queries outside this union of source windows have exactly
                    # zero response. Round the retained range to kernel buckets.
                    q1 = min(d.length, t1 + self.model.window_sizes[i][0])
                    qlength = min(d.length, -(-(q1-t0) // KEY_BUCKET) * KEY_BUCKET)
                    qs = slice(max(0, q1-qlength), q1)
                    if xin.is_cuda and raw.shape[-1] in (32, 64):
                        # All directions share a source tile. The fused scorer
                        # reduces query tiles without writing H*P*Q*M intermediates.
                        moves = prepared_key_moves(xin[sl], steps[sl], rows, projected_x[sl],
                                                   projected_rows, raw[:, sl], cos[sl], sin[sl])
                        if moments is not None:
                            scores = key_moment_scores(moves, moments[0][:, sl], moments[1][:, sl])
                        else:
                            scores = fused_key_response(qp[:, qs], moves, A[:, qs, sl], C[:, qs, sl],
                                                        t0, qs.start, self.model.window_sizes[i][0])
                        estimate[r, sl] = decode_key_pairs(scores, steps[sl], rows) * (len(rows) / total_rows)
                        continue
                    derivatives = torch.empty(len(xin[sl]), len(rows), device=self.device)
                    for j0 in range(0, len(rows), s.key_chunk):
                        js = slice(j0, min(j0+s.key_chunk, len(rows)))
                        derivatives[:, js] = key_input_prepared_derivatives(
                            xin[sl], steps[sl], rows[js], projected_x[sl], projected_rows[js],
                            raw[:, sl], cos[sl], sin[sl],
                            qp[:, qs], A[:, qs, sl], C[:, qs, sl])
                    estimate[r, sl] = derivatives @ rows / total_rows
            credits[i] = self.group_sum(estimate.flatten(0, 1))
            raw_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
        self._input_key_errors = raw_errors
        return credits

    def path_credits(self, chain, targets, raw_values=None):
        """Sum independent input responses, then transport only along residual identities."""
        L = self.model.config.n_layer
        inputs = {i: torch.zeros_like(chain[i]) for i in range(L)}
        if self.s.value_path:
            _, inputs = self.value_path_credit(chain, return_inputs=True, raw_values=raw_values)
        if self.s.key_path_draws:
            keys = self.key_input_credits(targets)
            inputs = {i: inputs[i] + keys[i] for i in range(L)}
        if self.s.gate_path:
            for i, credit in self.gate_input_credits(targets, raw_values).items():
                inputs[i] = inputs[i] + credit
        extra = {L-1: torch.zeros_like(chain[0])}
        for i in reversed(range(1, L)):
            extra[i-1] = self.model.resid_lambdas[i].float() * (inputs[i]+extra[i])
        return extra, inputs

    def gate_input_credits(self, targets, raw_values=None):
        """Independent gate-input experiments from raw hubs; only linear consumers follow."""
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        for i, target in targets.items():
            # Advance for every layer to retain the tested private pilot bases.
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            if not has_ve(i, self.model.config.n_layer):
                continue
            rows = frame[indices] * signs
            value = (self.value_credit(i, target) if raw_values is None else raw_values[i])
            value = value.reshape(-1, heads, self.width // heads)
            ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
            kappa = (value * ve).sum(-1)
            xin = d.blocks[i].float().flatten(0, 1)
            weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
            projected_rows = F.linear(rows[:, :weight.shape[1]], weight)
            row_square = rows.square().mean(-1)
            result = torch.empty_like(xin)
            for t0 in range(0, len(xin), s.chain_tokens):
                sl = slice(t0, t0 + s.chain_tokens)
                result[sl] = prepared_gate_input_response(
                    xin[sl], weight, kappa[sl], rows, projected_rows, row_square, s.gate_path_sigma)
            credits[i] = result
        return credits

    @staticmethod
    def path_scalar_credit(inputs, extra, x0, outs):
        """Linear mixing coefficients see input responses plus later identity-path credit."""
        credits = []
        for i in range(len(outs)):
            credit = inputs[i]+extra[i]
            previous = x0 if i == 0 else outs[i-1]
            credits.append(torch.stack([(credit*previous).sum(), (credit*x0).sum()]))
        return torch.stack(credits, dim=1)

    def chain_credits(self, e, x0, mids, outs, routes, y):
        """Measure embedding and mid-stream credits from loss.

        Each root uses a full orthogonal cycle. Mixing scalars retain their
        central mid-stream probes. MLP outputs reuse the next mid credit;
        the final output uses the existing head probe.
        """
        model, s = self.model, self.s
        L = model.config.n_layer
        vocab = model.config.vocab_size
        weight = model.lm_head.weight.float()[:vocab]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        # Draw bases in the original site order. Replay one sequence at a time
        # so routed K/V storage does not grow with the per-GPU batch size.
        probes = []
        lo, hi = self.share(self.width)  # this GPU's share of each cycle; the scalar moves on the first GPU
        P, scalars_here = hi - lo, self.part == 0
        for site in range(-1, L):
            H = self.basis()[lo:hi]
            probed = e if site < 0 else mids[site]
            steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            credit = torch.empty(n, self.width, device=self.device)
            lam = torch.zeros(2, device=self.device)
            probes.append((site, H, steps, credit, lam))
        length = self.capture.length if s.own_query else n
        for start in range(0, n, length):
            own = None
            if s.own_query:
                own = self.own_query_inputs(start // length)
                if s.query_keys:
                    own = self.strong_keys(own, s.query_keys)
                    if s.future_routed:
                        self.routed[start // length] = own
            for t0 in range(start, start + length, m):
                t1 = min(t0 + m, start + length)
                sl = slice(t0, t1)
                own_sl = None
                if own is not None:
                    begin, end = t0 - start, t1 - start
                    if own[0] == 'routed':
                        local = slice(begin, end)
                        own_sl = ('routed', own[1][local], [z[local] for z in own[2]], own[3]) + tuple(
                            [z[local] for z in part] for part in own[4:8]) + own[8:]
                    else:
                        own_sl = self.own_query_window(own, begin, end)
                kernel = chain_chunk_own if own_sl is not None and own_sl[0] == 'window' else chain_chunk
                # All roots use the same selected units. Gather once per token block,
                # not again for each of the 44 layer visits of the nine roots.
                selected_fc = [w[r[0][sl]] for w, r in zip(w_fc, routes)]
                selected_out = [w[r[0][sl]] for w, r in zip(w_out_t, routes)]
                for site, H, steps, credit, lam in probes:
                    move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                    scale = steps[sl].expand(-1, P)
                    if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1),
                                          -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                           torch.full_like(scale[:, :2], -h)], 1)
                    derivative = kernel(
                        site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, selected_fc, selected_out, weight_t,
                        base[sl], target_w[sl], y[sl], own_sl, packed=True)
                    credit[sl] = derivative[:, :P] @ H / self.width
                    if site >= 0 and scalars_here:
                        lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                del selected_fc, selected_out
            # Drop views as well as the owning cache before constructing the
            # next sequence's routed keys and values.
            del own_sl, own
        credits = {}
        for site, _, _, credit, lam in probes:
            credits['emb' if site < 0 else site] = self.group_sum(credit)
            if site >= 0:
                credits['lam', site] = self.group_sum(lam)
        return credits

    def own_query_inputs(self, sequence=0):
        """Clean keys/values for one sequence; preserve upstream math and token positions."""
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            k = d.outputs[prefix + 'c_k'][sequence].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            c, s = cos[:, None], sin[:, None]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][sequence].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][sequence].float().view(length, heads, width)
            inputs.append(d.blocks[i][sequence].float())
            w_q.append(block.attn.c_q.weight.float())
            keys.append(k.transpose(0, 1).to(torch.bfloat16).contiguous())
            values.append(v.transpose(0, 1).to(torch.bfloat16).contiguous())
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            windows.append(model.window_sizes[i][0])
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)

    @staticmethod
    def strong_keys(own, k):
        """Only top-k logits move; keep the remaining output for exact renormalization."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        length = len(pos)
        top_keys, top_values, top_p, rests = [], [], [], []
        for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
            heads, _, width = kk.shape
            c, s = cos[:, None], sin[:, None]
            r = (norm(x) @ wq.T).view(length, heads, width)
            q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                                r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            lag = pos[:, None] - pos[None]
            allowed = (lag >= 0) & (lag <= window)
            prob = torch.softmax((torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width ** -0.5)
                                 .masked_fill(~allowed, float('-inf')), -1)
            p, idx = prob.topk(min(k, length), dim=-1)
            gather = lambda z: z.unsqueeze(1).expand(-1, length, -1, -1).gather(
                2, idx.unsqueeze(-1).expand(-1, -1, -1, width))
            ktop, vtop = gather(kk), gather(vv)
            out = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
            rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
            top_keys.append(ktop.transpose(0, 1).contiguous())
            top_values.append(vtop.transpose(0, 1).contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
            rests.append(rest.transpose(0, 1).contiguous())
        return ('routed', pos, inputs, w_q, top_keys, top_values, top_p, rests, w_proj_t, (cos, sin))

    @staticmethod
    def own_query_window(own, t0, t1):
        """Slice clean context windows, with the same 256-token buckets as PR #19."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        total = keys[0].shape[1]
        rels, ks, vs = [], [], []
        for k, v, window in zip(keys, values, windows):
            need = t1 - max(0, t0 - window)
            length = min(total, -(-need // KEY_BUCKET) * KEY_BUCKET)
            start = max(0, t1 - length)
            rels.append(pos[t0:t1] - start)
            ks.append(k[:, start:start + length])
            vs.append(v[:, start:start + length])
        return ('window', pos[t0:t1], rels, [z[t0:t1] for z in inputs],
                w_q, ks, vs, w_proj_t, cos, sin, windows)

    def select_mlp_routes(self, layer, mid, hpre, activation):
        """Select the fixed replay units; experiments may override only this ranking."""
        weight = self.model.transformer.h[layer].mlp.c_proj.weight.float()
        return (activation * weight.norm(dim=0)).topk(self.s.route_k, dim=-1).indices

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        self._input_key_errors = None
        self.measured = {}  # Last step's diagnostic credits are not inputs to this estimate.
        self.routed = {}  # chain_credits' routed own-query contexts, reused by the future probes
        global MOVE_DTYPE  # read by the compiled chain replay
        MOVE_DTYPE = torch.bfloat16 if s.own_query_precision == 'bf16' else torch.float32
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()  # coordinate differences at the final readout
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        for i in range(L):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            top = self.select_mlp_routes(i, mid, hpre, r)
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        chain = self.chain_credits(e, x0, mids, outs, routes, yf)
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        # Reuse only directly measured hub credit, scoped to this capture.
        # Corrected targets still take a separate A.T @ target in exact_values.
        raw_values = {i: self.value_credit(i, target) for i, target in targets.items()}
        extra, inputs = self.path_credits(chain, targets, raw_values) if s.value_path or s.key_path_draws or s.gate_path else (None, None)
        value_targets = {}
        scalars = torch.zeros(2, L, device=self.device)
        for i in range(L):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            if i == L - 1:
                d_out = g_top
            else:
                d_out = lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = (d_mid, d_out) if extra is None else (d_mid + extra[i], d_out + extra[i])
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            if extra is not None:
                value_targets[i] = (x_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            # the internals' local step through the attention module, scored as c_proj sees the mid-stream credit
            scalars[:, i] = chain['lam', i]
        if s.path_scalars and extra is not None:
            scalars += self.path_scalar_credit(inputs, extra, x0, outs)
        self.measured = dict(chain=chain, g_top=g_top, targets=targets)
        self.internals(gradients, targets, b, t, value_targets or None, self._input_key_errors, raw_values)
        self._input_key_errors = None
        d.embedding_gradients(gradients, chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        if self.part:  # split: formed in full by every GPU of the sequence, kept by its first GPU (the
            # internals split or average their own: keys share dims, queries share draws, hidden units share units)
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight') or name in (
                        'transformer.wte.weight', 'feedback.weight', 'ngram.table.weight', 'resid_lambdas',
                        'x0_lambdas'):
                    gradients[name] = torch.zeros_like(gradients[name])
        if s.future_draws:
            # Use this capture's original measured hub targets, never corrected
            # targets. Only the source parameter gradients receive the addition.
            add_future_gradients(self, gradients)
        return gradients


# -------------------------------------------------------------------------------------------------- future probes


SCORE_COLUMNS = 32  # Smaller tiles avoid shared-memory/occupancy pressure on RTX 5090.


def norm_delta(x, delta):
    width = x.shape[-1]
    eps = torch.finfo(x.dtype).eps
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + eps)
    moved_inverse = torch.rsqrt(
        (square + 2 * (x * delta).sum(-1, keepdim=True)
         + delta.square().sum(-1, keepdim=True)) / width + eps)
    return delta * moved_inverse + x * (moved_inverse-inverse)


def rotate(x, cos, sin):
    half = x.shape[-1]//2
    return torch.cat((x[..., :half]*cos+x[..., half:]*sin,
                      -x[..., :half]*sin+x[..., half:]*cos), -1)


def receiver_changes(x, move, kraw, gate, ve, wkv, wg, cos, sin):
    """Finite K and effective-V changes of each independent source token."""
    m, p, width = move.shape
    heads, hd = kraw.shape[-2:]
    dn = norm_delta(x[:, None], move)
    projected = F.linear(dn, wkv).reshape(m, p, 2, heads, hd)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(kraw[:, None], c, s)
    dr = rotate(projected[:, :, 0], c, s)
    dk = norm_delta(r0, dr)
    dv = projected[:, :, 1]
    if gate is not None:
        dg = F.linear(dn[..., :wg.shape[-1]], wg)
        gate_change = 2*((gate[:, None]+dg).sigmoid()-gate[:, None].sigmoid())
        dv = dv + gate_change[..., None]*ve[:, None]
    return dk.permute(2, 1, 0, 3).contiguous(), dv.permute(2, 1, 0, 3).contiguous()


def joint_scores(q, target, dk, dv, a, contrast, source_start, window):
    """Exact single-column softmax update; output shape H,P,M."""
    dh = q.shape[-1]
    ds = torch.einsum("hsd,hpmd->hpsm", q, dk)*dh**-.5
    changed = a[:, None]*torch.expm1(ds)
    value_score = torch.einsum("hsd,hpmd->hpsm", target, dv)
    scores = (changed*contrast[:, None] + (a[:, None]+changed)*value_score)/(1+changed)
    s = torch.arange(q.shape[1], device=q.device)[:, None]
    t = source_start+torch.arange(dk.shape[2], device=q.device)[None]
    allowed = (s>t)&(s-t<=window)
    return (scores*allowed[None, None]).sum(2)


@cache
def score_kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(Q, G, DK, DV, A, C, Out,
               T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
               GH: tl.constexpr, GT: tl.constexpr, GD: tl.constexpr,
               AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
               CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
               Start, Window: tl.constexpr, BC: tl.constexpr, BQ: tl.constexpr):
        head = tl.program_id(1)
        col = tl.program_id(0)*BC+tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        offsets = (head*P*M+col[None, :])*D+features[:, None]
        dk = tl.load(DK+offsets, col[None, :]<P*M, 0)
        dv = tl.load(DV+offsets, col[None, :]<P*M, 0)
        first = (Start+tl.min(source, 0))//BQ*BQ
        last = tl.minimum(T, Start+tl.max(source, 0)+Window+1)
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start+tl.arange(0, BQ)
            q = tl.load(Q+head*QH+query[:, None]*QT+features[None, :]*QD,
                        query[:, None]<T, 0)
            g = tl.load(G+head*GH+query[:, None]*GT+features[None, :]*GD,
                        query[:, None]<T, 0)
            ds = tl.dot(q, dk, input_precision="tf32x3")*(D**-.5)
            vs = tl.dot(g, dv, input_precision="tf32x3")
            valid = (query[:, None]<T)&(col[None, :]<P*M)
            a = tl.load(A+head*AH+query[:, None]*AT+source[None, :]*AM, valid, 0)
            c = tl.load(C+head*CH+query[:, None]*CT+source[None, :]*CM, valid, 0)
            changed = a*libdevice.expm1(ds)
            allowed = (query[:, None]>Start+source[None, :])&(
                query[:, None]-Start-source[None, :]<=Window)
            response = (changed*c+(a+changed)*vs)/(1+changed)
            result += tl.sum(tl.where(valid&allowed, response, 0), 0)
        tl.store(Out+head*P*M+col, result, col<P*M)

    return kernel


@torch.library.custom_op("noise_record::future_scores", mutates_args=())
def fused_scores(q: torch.Tensor, target: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                 a: torch.Tensor, contrast: torch.Tensor, source_start: int,
                 window: int) -> torch.Tensor:
    if not q.is_cuda:
        return joint_scores(q, target, dk, dv, a, contrast, source_start, window)
    import triton
    h, p, m, d = dk.shape
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    score_kernel()[(triton.cdiv(p*m, SCORE_COLUMNS), h)](
        q, target, dk.contiguous(), dv.contiguous(), a, contrast, out,
        q.shape[1], m, p, d, *q.stride(), *target.stride(),
        *a.stride(), *contrast.stride(), source_start, window, SCORE_COLUMNS, 64,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@fused_scores.register_fake
def _(q, target, dk, dv, a, contrast, source_start, window):
    return torch.empty(dk.shape[:-1], device=q.device,
                       dtype=torch.float32 if q.is_cuda else q.dtype)


def future_moments(q, target, a, contrast, window):
    """Degree-2 moments over each source token's receiving queries (one layer, all heads), so that moment_scores
    gives joint_scores' exact single-column update to third order in the change, for every direction at once.

    With u = q_s.dk / sqrt(D), w = target_s.dv and changed = a (e^u - 1), the update
    (changed c + (a + changed) w) / (1 + changed) = a c u + (a c / 2 - a^2 c) u^2 + a w + (a - a^2) u w + O(3),
    so its sum over the queries s is linear and quadratic forms in (dk, dv) with per-source moments: the
    construction of the merged track 2 kernels (#29), here over the future probes' receivers.
    """
    h, t, dh = q.shape
    s = torch.arange(t, device=q.device)
    lag = s[:, None]-s[None, :]
    a = a*((lag > 0) & (lag <= window))  # [head, query, source], the queries that see the source
    ac = a*contrast
    mk = torch.einsum('hst,hsd->htd', ac, q)*dh**-.5
    mv = torch.einsum('hst,hsd->htd', a, target)
    qq = (q[..., :, None]*q[..., None, :]).flatten(-2)
    qg = (q[..., :, None]*target[..., None, :]).flatten(-2)
    mkk = torch.bmm((ac*(.5-a)).transpose(1, 2), qq).view(h, t, dh, dh)/dh
    mkv = torch.bmm((a-a*a).transpose(1, 2), qg).view(h, t, dh, dh)*dh**-.5
    return mk, mv, mkk, mkv


def moment_scores(dk, dv, mk, mv, mkk, mkv):
    """Summed future scores [P, M] of key and value changes [H, P, M, D] from the moments of their tokens M."""
    quadratic = torch.einsum('hpmd,hmde->hpme', dk, mkk)*dk+torch.einsum('hpmd,hmde->hpme', dk, mkv)*dv
    return (quadratic+dk*mk[:, None]+dv*mv[:, None]).sum((0, 3))


def signed_hidden_moves(raw, directions, step, weight):
    """Exact paired squared-ReLU responses for sign directions, with one large GEMM.

    The even component is independent of each +/-1 direction. This identity
    includes threshold crossings; it is not a derivative approximation.
    """
    base = raw.relu()
    plus, minus = (raw+step).relu(), (raw-step).relu()
    dp, dm = (plus-base)*(plus+base), (minus-base)*(minus+base)
    odd, even = (dp-dm)*.5, (dp+dm)*.5
    moved = F.linear(directions*odd[:, None], weight)
    center = F.linear(even, weight)[:, None]
    return torch.cat((center+moved, center-moved), 1)


def future_embedding_moves(raw, directions, step):
    delta = step[:, None]*directions
    return torch.cat((norm_delta(raw[:, None], delta), norm_delta(raw[:, None], -delta)), 1)


@cache
def query_kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(DQ, K, V, A, Base, Out,
               M: tl.constexpr, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               T: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
               VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
               AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
               OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
               BP: tl.constexpr, BK: tl.constexpr):
        token, head = tl.program_id(1) // H, tl.program_id(1) % H
        probe = tl.program_id(0)*BP+tl.arange(0, BP)
        feature = tl.arange(0, D)
        delta = tl.load(DQ+((token*P+probe[:, None])*H+head)*D+feature[None, :],
                        probe[:, None]<P, 0)
        numerator = tl.full((BP, D), 0, tl.float32)
        partition = tl.full((BP,), 0, tl.float32)
        for start in range(0, T, BK):
            key = start+tl.arange(0, BK)
            a = tl.load(A+head*AH+token*AM+key*AT, key<T, 0)
            # Causal/window-masked tiles are exactly zero, not a top-k truncation.
            if tl.sum(a, 0) > 0:
                k = tl.load(K+head*KH+key[None, :]*KT+feature[:, None]*KD, key[None, :]<T, 0)
                v = tl.load(V+head*VH+key[:, None]*VT+feature[None, :]*VD, key[:, None]<T, 0)
                ds = tl.dot(delta, k, input_precision="tf32x3")*(D**-.5)
                change = a[None, :]*libdevice.expm1(ds)
                partition += tl.sum(change, 1)
                numerator += tl.dot(change, v, input_precision="tf32x3")
        base = tl.load(Base+head*OH+token*OM+feature*OD)
        response = (numerator-partition[:, None]*base[None, :])/(1+partition[:, None])
        tl.store(Out+((token*P+probe[:, None])*H+head)*D+feature[None, :],
                 response, probe[:, None]<P)
    return kernel


@torch.library.custom_op("noise_record::future_query_response", mutates_args=())
def query_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   probabilities: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Stream the exact finite softmax response without a token/probe/key tensor."""
    if not dq.is_cuda:
        ds = torch.einsum("mphd,hsd->mphs", dq, keys)*dq.shape[-1]**-.5
        changed = probabilities.permute(1, 0, 2)[:, None]*torch.expm1(ds)
        partition = changed.sum(-1, keepdim=True)
        numerator = torch.einsum("mphs,hsd->mphd", changed, values)
        return (numerator-partition*base.transpose(0, 1)[:, None])/(1+partition)
    import triton
    m, p, h, d = dq.shape
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    query_kernel()[(triton.cdiv(p, 32), m*h)](
        dq.contiguous(), keys, values, probabilities, base, out, m, p, h, d, keys.shape[1],
        *keys.stride(), *values.stride(), *probabilities.stride(), *base.stride(),
        32, 64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@query_response.register_fake
def _(dq, keys, values, probabilities, base):
    return torch.empty_like(dq, memory_format=torch.contiguous_format)


def streamed_query_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight, base):
    m, p, width = directions.shape
    heads, _, hd = keys.shape
    delta = torch.cat((step[:, None]*directions, -step[:, None]*directions), 1)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = rotate(delta.reshape(m, 2*p, heads, hd), c, s)
    response = query_response(norm_delta(r0, dr), keys, values, probabilities, base)
    return F.linear(response.flatten(-2), weight)


def mlp_output_moves(mid, move, hidden, w_fc, w_out):
    """Carry each signed Q trial through its own block's complete MLP.

    Anchor at the captured hidden pre-activation, exactly as receiver_changes
    anchors K/V. This uses finite forward responses, including ReLU crossings.
    The incoming attention/residual move remains part of the block output.
    """
    dh = F.linear(norm_delta(mid[:, None], move), w_fc)
    base = hidden[:, None].relu()
    moved = (hidden[:, None] + dh).relu()
    return move + F.linear((moved-base)*(moved+base), w_out)


compiled_hidden = torch.compile(signed_hidden_moves, dynamic=False)
compiled_embedding = torch.compile(future_embedding_moves, dynamic=False)
compiled_query = torch.compile(streamed_query_moves, dynamic=False)


@torch.compile(dynamic=True)
def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin,
                    q, target, a, contrast, start, window):
    dk, dv = receiver_changes(x, move*coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return fused_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T


@torch.compile(dynamic=True)
def receiver_moment_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin, mk, mv, mkk, mkv):
    dk, dv = receiver_changes(x, move*coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return moment_scores(dk, dv, mk, mv, mkk, mkv).T


@torch.compile(dynamic=False)
def stacked_receiver_scores(move, coeffs, x, kraw, wkv, gate, ve, wg, cos, sin, mk, mv, mkk, mkv):
    """receiver_moment_scores summed over every receiver of one source chunk, with one projection of the move for
    all of them: norm_delta(x_j, c_j move) = c_j inv'_j move + (inv'_j - inv_j) x_j, so its K/V products are
    c_j inv'_j (move W_kv,j) + (inv'_j - inv_j) (x_j W_kv,j). Stacked per receiver j: coeffs [J], x [J, m, W],
    kraw [J, m, H, hd], wkv [J, 2W, W], gate [J, m, H], ve [J, m, H, hd], wg [J, H, G] (zeros for blocks without
    value embeddings, whose gate change is then zero), moments [J, H, m, ...]. Returns [m, P]."""
    J, m, width = x.shape
    p = move.shape[1]
    heads, hd = kraw.shape[-2:]
    eps = torch.finfo(x.dtype).eps
    c = coeffs[:, None, None]
    square = x.square().sum(-1)[..., None]  # [J, m, 1]
    inv = torch.rsqrt(square/width+eps)
    moved = torch.rsqrt((square+2*c*torch.einsum('jmw,mpw->jmp', x, move)+c*c*move.square().sum(-1))/width+eps)
    proj = torch.einsum('mpw,jow->jmpo', move, wkv)
    xproj = torch.einsum('jmw,jow->jmo', x, wkv)[:, :, None]
    dproj = ((c*moved)[..., None]*proj+(moved-inv)[..., None]*xproj).view(J, m, p, 2, heads, hd)
    cc, ss = cos[None, :, None, None], sin[None, :, None, None]
    dk = norm_delta(rotate(kraw[:, :, None], cc, ss), rotate(dproj[:, :, :, 0], cc, ss))  # [J, m, P, H, hd]
    g = wg.shape[-1]
    dn = (c*moved)[..., None]*move[None, ..., :g]+(moved-inv)[..., None]*x[:, :, None, :g]
    dg = torch.einsum('jmpg,jhg->jmph', dn, wg)
    dv = dproj[:, :, :, 1]+(2*((gate[:, :, None]+dg).sigmoid()-gate[:, :, None].sigmoid()))[..., None]*ve[:, :, None]
    quadratic = (torch.einsum('jmphd,jhmde->jmphe', dk, mkk)*dk+torch.einsum('jmphd,jhmde->jmphe', dk, mkv)*dv)
    linear = dk*mk.transpose(1, 2)[:, :, None]+dv*mv.transpose(1, 2)[:, :, None]
    return (quadratic+linear).sum((0, 3, 4))


@torch.compile(dynamic=False)
def query_mlp_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight,
                    base, mid, hidden, w_fc, w_out):
    move = streamed_query_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight, base)
    return mlp_output_moves(mid, move, hidden, w_fc, w_out)


@torch.compile(dynamic=False)
def routed_query_mlp_moves(raw, directions, step, cos, sin, top_keys, top_values, top_p, rest, weight,
                           mid, hidden, w_fc, w_out):
    """query_mlp_moves with the chain's routed own-query response (its query_keys strongest keys per head move,
    the rest rescale exactly) in place of every key."""
    m, p, width = directions.shape
    heads, hd = top_keys.shape[1], top_keys.shape[-1]
    delta = torch.cat((step[:, None]*directions, -step[:, None]*directions), 1)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = rotate(delta.reshape(m, 2*p, heads, hd), c, s)
    change = routed_output_change(norm_delta(r0, dr), top_keys, top_values, top_p, rest)
    return mlp_output_moves(mid, F.linear(change.flatten(-2), weight), hidden, w_fc, w_out)


class FiniteFuture:
    def __init__(self, estimator, source_layers=4, tokens=64, chunk=512, sigma=.01, seed=9181,
                 query_mlp=True, timing=True, moments=False, stacked=False):
        self.e = estimator
        self.moments, self.stacked = moments, stacked
        self.routed = None  # the chain's routed own-query context of this sequence, when set
        self.source_layers, self.tokens, self.chunk = source_layers, tokens, chunk
        self.sigma, self.seed = sigma, seed
        self.query_mlp = query_mlp
        self.timing = timing
        self.contexts = None
        if tokens <= 0 or chunk <= 0 or sigma <= 0:
            raise ValueError("tokens, chunk and sigma must be positive")

    @torch.no_grad()
    def prepare(self, sequence=0):
        e = self.e
        d, model = e.capture, e.model
        t, heads = d.length, model.config.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        contexts = []
        for i, block in enumerate(model.transformer.h):
            prefix = f"transformer.h.{i}.attn."
            qraw, kraw, vraw = [d.outputs[prefix+n][sequence:sequence+1].float()
                               for n in ("c_q", "c_k", "c_v")]
            v = vraw.reshape(1, t, heads, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix+"ve_gate"][sequence].float()
                ve = d.outputs[f"value_embeds.{i}"][sequence].float().reshape(t, heads, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2*gate.sigmoid())[None, ..., None]*ve[None]
            values = v.transpose(1, 2)[0]
            qp, kp, a, out = attention_state(
                qraw, kraw, values[None], cos, sin, heads, model.window_sizes[i][0])
            q, k, a, out = qp[0], kp[0], a[0], out[0]
            target = e.measured["targets"][i][sequence].reshape(t, heads, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2))-(target*out).sum(-1, keepdim=True)
            contexts.append(dict(x=d.blocks[i][sequence].float(), q=q, k=k,
                kraw=kraw[0].reshape(t, heads, -1), values=values, a=a, out=out, contrast=contrast,
                target=target, gate=gate, ve=ve, wg=wg,
                wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(),
                window=model.window_sizes[i][0]))
            if self.moments:
                contexts[-1]["moments"] = future_moments(q, target, a, contrast, model.window_sizes[i][0])
            if self.query_mlp and i < self.source_layers:
                contexts[-1]["source_mlp"] = (
                    d.blocks[i][sequence].float()+d.outputs[prefix+"c_proj"][sequence].float(),
                    d.outputs[f"transformer.h.{i}.mlp.c_fc"][sequence].float(),
                    block.mlp.c_fc.weight.float(), block.mlp.c_proj.weight.float())
        self.contexts, self.sequence = contexts, sequence
        self.cos, self.sin = cos[0, :, 0], sin[0, :, 0]
        return self

    def sources(self, families):
        for family in families:
            if family == "wte":
                yield family, -1, "transformer.wte"
            else:
                for layer in range(self.source_layers):
                    suffix = "mlp.c_fc" if family == "hidden" else "attn.c_q"
                    yield family, layer, f"transformer.h.{layer}.{suffix}"

    def experiment(self, family, layer, raw, sl, receivers, start):
        """Bind clean tensors once per token chunk, outside the direction loop."""
        model = self.e.model
        if family == "hidden":
            response = partial(compiled_hidden, raw[sl],
                               weight=model.transformer.h[layer].mlp.c_proj.weight.float())
        elif family == "wte":
            response = partial(compiled_embedding, raw[sl])
        elif self.routed is not None and self.query_mlp:
            ctx = self.contexts[layer]
            _, _, _, _, top_keys, top_values, top_p, rests, _, _ = self.routed
            mid, hidden, w_fc, w_out = ctx["source_mlp"]
            response = partial(routed_query_mlp_moves, raw[sl], cos=self.cos[sl], sin=self.sin[sl],
                               top_keys=top_keys[layer][sl], top_values=top_values[layer][sl],
                               top_p=top_p[layer][sl], rest=rests[layer][sl],
                               weight=model.transformer.h[layer].attn.c_proj.weight.float(),
                               mid=mid[sl], hidden=hidden[sl], w_fc=w_fc, w_out=w_out)
        else:
            ctx = self.contexts[layer]
            kwargs = dict(cos=self.cos[sl], sin=self.sin[sl], keys=ctx["k"], values=ctx["values"],
                          probabilities=ctx["a"][:, sl], base=ctx["out"][:, sl],
                          weight=model.transformer.h[layer].attn.c_proj.weight.float())
            if self.query_mlp:
                mid, hidden, w_fc, w_out = ctx["source_mlp"]
                kwargs.update(mid=mid[sl], hidden=hidden[sl], w_fc=w_fc, w_out=w_out)
            response = partial(query_mlp_moves if self.query_mlp else compiled_query, raw[sl], **kwargs)
        if self.stacked and receivers and all("moments" in self.contexts[j] for j, _ in receivers):
            return LocalExperiment(response, self.stacked_scorer(receivers, sl))
        scorers = []
        for j, coefficient in receivers:
            ctx = self.contexts[j]
            changes = dict(coefficient=coefficient, x=ctx["x"][sl], kraw=ctx["kraw"][sl],
                           gate=None if ctx["gate"] is None else ctx["gate"][sl],
                           ve=None if ctx["ve"] is None else ctx["ve"][sl],
                           wkv=ctx["wkv"], wg=ctx["wg"], cos=self.cos[sl], sin=self.sin[sl])
            if "moments" in ctx:
                mk, mv, mkk, mkv = (m[:, sl] for m in ctx["moments"])
                scorers.append(partial(receiver_moment_scores, **changes, mk=mk, mv=mv, mkk=mkk, mkv=mkv))
            else:
                scorers.append(partial(receiver_scores, **changes, q=ctx["q"], target=ctx["target"],
                                       a=ctx["a"][:, :, sl], contrast=ctx["contrast"][:, :, sl], start=start,
                                       window=ctx["window"]))

        def score(move):
            if not scorers:
                return move.new_zeros(move.shape[:2])
            total = scorers[0](move)
            for receiver in scorers[1:]:
                total = total + receiver(move)
            return total
        return LocalExperiment(response, score)

    def stacked_scorer(self, receivers, sl):
        """stacked_receiver_scores bound to these receivers' clean tensors at tokens sl."""
        ctxs = [self.contexts[j] for j, _ in receivers]
        x = torch.stack([c["x"][sl] for c in ctxs])
        kraw = torch.stack([c["kraw"][sl] for c in ctxs])
        heads, g = kraw.shape[-2], self.e.model.transformer.h[0].attn.ve_gate_channels
        some = lambda name, shape: torch.stack([c[name][sl] if c[name] is not None else
                                                x.new_zeros(shape) for c in ctxs])
        wg = torch.stack([c["wg"] if c["wg"] is not None else x.new_zeros(heads, g) for c in ctxs])
        moments = [torch.stack([c["moments"][k][:, sl] for c in ctxs]) for k in range(4)]
        return partial(stacked_receiver_scores, coeffs=torch.stack([coefficient for _, coefficient in receivers]),
                       x=x, kraw=kraw, wkv=torch.stack([c["wkv"] for c in ctxs]),
                       gate=some("gate", (x.shape[1], heads)), ve=some("ve", kraw.shape[1:]), wg=wg,
                       cos=self.cos[sl], sin=self.sin[sl], mk=moments[0], mv=moments[1], mkk=moments[2],
                       mkv=moments[3])

    @torch.no_grad()
    def estimate(self, budgets=(512,), families=("hidden", "query", "wte")):
        assert self.contexts is not None
        e, model, d = self.e, self.e.model, self.e.capture
        budgets = sorted(set(budgets))
        if not budgets or budgets[0] <= 0:
            raise ValueError("Probe budgets must be positive")
        # Preserve every requested prefix when the execution tile is larger.
        chunk = gcd(self.chunk, *budgets)
        split, part = getattr(e, 'split', 1), getattr(e, 'part', 0)
        if split > 1:  # each GPU of the sequence takes a share of the draws, decoded at the full budget
            if len(budgets) != 1 or budgets[0] % split:
                raise ValueError('Split future probes need one budget divisible by the GPUs of a sequence')
            share = budgets[0] // split
            chunk = gcd(self.chunk, share)
            span = range(part * share, (part + 1) * share, chunk)
        else:
            span = range(0, max(budgets), chunk)
        result = {p: {} for p in budgets}
        timings = {}
        lam, lam0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        for family, layer, name in self.sources(families):
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            raw = d.outputs[name][self.sequence].float()
            t, width = raw.shape
            if budgets[-1] > width:
                raise ValueError(f"{name}: budget {budgets[-1]} exceeds the {width}-direction cycle")
            gen = torch.Generator(raw.device).manual_seed(self.seed+7919*(layer+2)+101*("hidden query wte".split().index(family)))
            # Per-token random signs/offsets prevent a shared low-rank projection
            # from persisting in the batch's parameter gradient.
            frame = hadamard(width, raw.device, torch.float32)
            permutation = torch.randperm(width, device=raw.device, generator=gen)
            offsets = torch.randint(width, (t,), device=raw.device, generator=gen)
            signs = 2*torch.randint(2, (t, width), device=raw.device, generator=gen)-1
            steps = self.sigma*raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            credits = {p: torch.zeros_like(raw) for p in budgets}
            receivers = []
            coeff = torch.ones((), device=raw.device)
            if family == "wte":
                for j in range(len(self.contexts)):
                    coeff = lam[j]*coeff+lam0[j]
                    receivers.append((j, coeff))
            else:
                for j in range(layer+1, len(self.contexts)):
                    coeff = coeff*lam[j]
                    receivers.append((j, coeff))
            for t0 in range(0, t, self.tokens):
                sl = slice(t0, min(t0+self.tokens, t))
                trial = self.experiment(family, layer, raw, sl, receivers, t0)
                accumulator = torch.zeros_like(raw[sl])
                for p0 in span:
                    p1 = p0+chunk
                    index = (offsets[sl, None]+permutation[None, p0:p1]) % width
                    rows = frame[index]*signs[sl, None]
                    accumulator += trial(rows, steps[sl])
                    if split > 1:
                        if p1 == (part + 1) * share:
                            credits[budgets[0]][sl] = accumulator/budgets[0]
                    elif p1 in credits:
                        credits[p1][sl] = accumulator/p1
            for p in budgets:
                result[p][name] = credits[p][None]
            if self.timing and d.x.is_cuda:
                torch.cuda.synchronize()
            timings[name] = time.perf_counter()-start
        self.timings = timings
        return result


@torch.no_grad()
def add_future_gradients(estimator, gradients):
    """Add independent forward responses, releasing dense contexts per sequence.

    Draw a fresh frame seed from the checkpointed rank RNG each update. Hidden
    uses a partial cycle, so fixing that frame across training would bias it
    toward a persistent subspace. Never overwrite the measured hub targets.
    """
    e, s, d = estimator, estimator.s, estimator.capture
    e.maps.clear()
    seed = int(torch.randint(2**31, (), device=e.device,
                             generator=d.shared_generator if e.split > 1 else d.generator).item())
    future = FiniteFuture(e, source_layers=s.future_layers, tokens=s.future_tokens,
                          chunk=s.future_chunk, sigma=s.future_sigma, seed=seed,
                          query_mlp=True, timing=False, moments=s.future_moments,
                          stacked=s.future_stacked)
    for sequence in range(d.batch):
        future.seed = seed + 104729 * sequence
        future.routed = getattr(e, 'routed', {}).get(sequence)
        future.prepare(sequence)
        corrections = future.estimate((s.future_draws,))[s.future_draws]
        for site, credit in corrections.items():
            inputs = d.inputs[site][sequence]
            gradient = gradients[site + '.weight']
            if 'mlp.c_fc' in site:
                raw = d.outputs[site][sequence].float()
                step = s.future_sigma*raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                # Coordinates that are inactive in every signed trial have no
                # response; remove their partial-cycle decoding cross-talk.
                credit = credit * (raw+step > 0)[None]
            if site == 'transformer.wte':
                d.embedding_gradients(gradients, credit[0], sequence)
            else:
                gradient.addmm_(credit[0].T, inputs)
        future.contexts = None
        del corrections


# ------------------------------------------------------------------------------------------------------ optimizer


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


RUN = (1194, 240, 50, 514)


def polar(g):
    """Approximate orthogonalization with six quintic maps."""
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (1.05 * x.norm() + 1e-6)).bfloat16()
    for a, b, c in MAPS:
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360."""

    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t = 0

    def rail_beta(self):
        """Scheduled fast-rail and Nesterov beta."""
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                g = p.grad.float()
                lanes = 1 if p.size(0) >= p.size(1) else 0
                state = self.state[p]
                if not state:
                    state.update(fast=torch.zeros_like(g), slow=torch.zeros_like(g),
                                 energy=torch.zeros_like(g.mean(lanes, keepdim=True)))
                fast, slow, energy = state['fast'], state['slow'], state['energy']
                fast.lerp_(g, 1 - fast_beta)
                slow.lerp_(g, 1 - self.slow_beta)
                update = polar(g.lerp(weight * fast + (1 - weight) * slow, beta))
                energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                gained = update * energy.clamp_min(1e-10).rsqrt()
                update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                w = p.float()
                decay = (slow * w >= 0) * w * (wd * lr)
                p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
        self.t += 1


def newton_schulz(x):
    """The record's six quintic maps on a stack (B, n, n) of square matrices, in bfloat16."""
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    """The same maps on wide matrices (n < m), iterated on the n x n Gram matrix (Gram Newton-Schulz, Dao lab 2026).
    R is rebuilt from Q X before the maps in `restarts` to bound the float16 error."""
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    R, Q = x @ x.transpose(1, 2), eye
    for t, (a, b, c) in enumerate(MAPS):
        if t in restarts:
            x = Q @ x
            R, Q = x @ x.transpose(1, 2), eye
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return Q @ x


def polar_batch(g, margin=1.1):
    """Approximate orthogonalization of a stack (B, r, c) of same-shape matrices."""
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
                 ratio: float, margin: float):
    """One ANVIL II step for a stack of same-shape matrices; updates fast, slow and energy in place."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class BatchedAnvil(Anvil):
    """ANVIL II from modded-nanogpt PR #360, applied to all same-shape matrices at once."""

    def __init__(self, *args, polar_margin=1.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.polar_margin, self.stacks = polar_margin, {}

    def load_state_dict(self, state_dict):
        # Cached stacks must be rebuilt from the freshly restored per-parameter state.
        self.stacks.clear()
        return super().load_state_dict(state_dict)

    def stacked_state(self, shape, params):
        """The moving averages of one shape group as stacked tensors. Each parameter's state holds views into them,
        so the Boost -> Cut handoff, which copies per-parameter state, carries them over."""
        key = tuple(params)
        stack = self.stacks.get(key)
        if stack is not None and all(self.state[p].get('fast') is not None for p in params):
            return stack
        lanes = 1 if shape[0] >= shape[1] else 0
        weights = torch.stack([p.float() for p in params])

        def gathered(key, like):
            held = [self.state[p].get(key) for p in params]
            return torch.stack(held).clone() if all(h is not None for h in held) else torch.zeros_like(like)

        stack = dict(fast=gathered('fast', weights), slow=gathered('slow', weights),
                     energy=gathered('energy', weights.mean(lanes + 1, keepdim=True)))
        for i, p in enumerate(params):
            self.state[p].update(fast=stack['fast'][i], slow=stack['slow'][i], energy=stack['energy'][i])
        self.stacks[key] = stack
        return stack

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        device = self.param_groups[0]['params'][0].device
        value = lambda v: torch.tensor(float(v), device=device)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            by_shape = {}
            for p in group['params']:
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, params in by_shape.items():
                stack = self.stacked_state(shape, params)
                new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                                   stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                                   value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                                   1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5, self.polar_margin)
                torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        # Identical all-reduced gradients must produce identical rank replicas.
        # Scope this to updates so embedding scatter and probes keep their fast paths.
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for o in self.optimizers:
                o.step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_model(device, seed, config, projection_std, head_std=0.001):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # Output projections: uniform with the declared std / sqrt(fan_in).
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = projection_std * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        if head_std != 0.001:
            nn.init.normal_(model.lm_head.weight, std=head_std)
    return model.requires_grad_(False)


def make_optimizer(model, recipe, steps):
    """ANVIL II on block matrices, Adam elsewhere."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name or 'ngram' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
        betas=recipe.betas, eps=1e-8, weight_decay=0.0,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
                         polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    return Optimizers(anvil, adam)


def set_rate(optimizer, step, steps, warmdown, final=0.0):
    """Set the original val linear warmdown rate, with one-based training steps (down to `final` of the rate)."""
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


# ------------------------------------------------------------------------------------------------------- training


def save_training_state(path, model, optimizer, estimator, *, step, train_seconds, recipe, progress=None):
    """Atomically save common training state and each rank's independent RNGs."""
    device = estimator.device
    random = dict(generator=estimator.generator.get_state(),
                  shared_generator=estimator.shared_generator.get_state(),
                  cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    if isinstance(estimator, Boost):
        random['capture_generator'] = estimator.capture.generator.get_state()
    states = [None] * estimator.world
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(states, random)
    else:
        states[0] = random
    if estimator.rank == 0:
        memory = getattr(estimator, 'memory', None)
        checkpoint = dict(step=step, train_seconds=train_seconds, recipe=recipe, world=estimator.world,
            schedule_steps=optimizer.optimizers[0].total,
            model=model.state_dict(), rng=states,
            estimator=type(estimator).__name__, progress=progress,
            optimizers=[dict(state=o.state_dict(), t=getattr(o, 't', None), kind=type(o).__name__)
                        for o in optimizer.optimizers],
            memory=None if memory is None else {name:getattr(memory, name) for name in
                    ('sites', 'beta', 'ridge', 'alpha_beta', 'key_moment', 'cross_moments', 'alpha', 'calibration')})
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def restore_training_state(path, model, optimizer, estimator, *, recipe):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    canonical = lambda x: json.loads(json.dumps(x))
    if canonical(checkpoint['recipe']) != canonical(recipe) or checkpoint['world'] != estimator.world:
        raise ValueError('Resume requires the same recipe and number of ranks')
    if checkpoint['estimator'] != type(estimator).__name__:
        raise ValueError('Resume requires the same estimator phase')
    if checkpoint['schedule_steps'] != optimizer.optimizers[0].total or checkpoint['memory'] is not None:
        raise ValueError('Resume requires the same schedule and no auxiliary memory')
    model.load_state_dict(checkpoint['model'], strict=True)
    for active, saved in zip(optimizer.optimizers, checkpoint['optimizers'], strict=True):
        if saved['kind'] != type(active).__name__:
            raise ValueError('Resume requires the same optimizer implementation')
        active.load_state_dict(saved['state'])
        if saved['t'] is not None:
            active.t = saved['t']
    optimizer.param_groups = [g for active in optimizer.optimizers for g in active.param_groups]
    random = checkpoint['rng'][estimator.rank]
    estimator.generator.set_state(random['generator'])
    estimator.shared_generator.set_state(random['shared_generator'])
    estimator.capture.generator.set_state(random['capture_generator'])
    torch.set_rng_state(random['cpu'])
    if random['cuda'] is not None:
        torch.cuda.set_rng_state(random['cuda'], estimator.device)
    return checkpoint['step'], checkpoint['train_seconds']


def stage_schedule(rows, settings):
    """Disjoint data slices, monotonically numbered updates, and phase-specific scheduler clocks."""
    b1, b2 = settings.boost_batch, settings.cut_batch
    if rows % b2 or b2 % b1:
        raise ValueError('Data rows must divide into complete Cut batches, and Cut batch must divide by Boost batch')
    n1, n2 = rows // b1, rows // b2
    alignment = b2 // b1
    boost = n1 if settings.estimator == 'boost' else round(settings.switch * n1) // alignment * alignment
    batches = [('boost', i, (i-1)*b1, i*b1, n1) for i in range(1, boost+1)]
    if settings.estimator != 'boost':
        batches += [(settings.estimator, i, (i-1)*b2, i*b2, n2) for i in range(boost*b1//b2+1, n2+1)]
    return batches


def handoff_optimizer(old, new, phase_step):
    """Keep ANVIL rails and Adam moments; change only the new phase's ANVIL scheduler clock."""
    for new_o, old_o in zip(new.optimizers, old.optimizers, strict=True):
        new_o.state.update(old_o.state)
    new.optimizers[0].t = phase_step - 1


def holdout_loss(model, rows, rank, world):
    """Mean next-token loss over the held-out training sequences, split across ranks."""
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in rows[rank::world].split(8):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            total[0] += loss.double()
            total[1] += batch[:, 1:].numel()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])  # one matrix of block i (attn.c_q, ..., mlp.c_proj)
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 32  # training sequences that feed no update and score the end merge
MERGE_FROM, MERGE_EVERY = 0.6, 5  # the exponential means follow the weights from 60% of the epoch, every 5 updates
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)  # weight of the newest weights in each mean
MERGE_RHOS = (0.0, 0.5, 1.0)  # how far a candidate moves outside the high-curvature subspace
MERGE_K, MERGE_TOKENS = 64, 512  # input-covariance directions per linear layer; frequent-token rows per table


@torch.no_grad()
def curvature_bases(model, rows, rank, world, k=MERGE_K):
    """Per linear layer, the top-k eigenvectors of its input covariance E[x x^T], and the token order of
    `rows` by frequency (for the embedding tables), from forward passes of the current weights on `rows` (the
    held-out training sequences; no gradient). Computed on rank 0 and broadcast, so every rank merges identically."""
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
              if isinstance(mod, nn.Linear) and 've_gate' not in name]
    bases = {key: torch.zeros(mod.weight.shape[1], k, device=device) for key, mod in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        acc, hooks = {}, []
        for key, mod in layers:
            def hook(module, inputs, output, key=key):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                acc[key] = acc.get(key, 0) + (x.T @ x).double()
            hooks.append(mod.register_forward_hook(hook))
        counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)
        for batch in rows.split(4):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            counts += torch.bincount(batch[:, :-1].reshape(-1).long(), minlength=len(counts)).double()
        for h in hooks:
            h.remove()
        for key, _ in layers:
            bases[key] = torch.linalg.eigh(acc[key])[1].flip(1)[:, :k].float().contiguous()
        order = torch.argsort(counts, descending=True)
    if world > 1:
        for key, _ in layers:
            torch.distributed.broadcast(bases[key], 0)
        torch.distributed.broadcast(order, 0)
    return bases, order


def subspace_move(name, move, bases, order):
    """The part of a weight move inside its high-curvature subspace: the top input-covariance directions of a
    linear layer, the rows of the most frequent tokens of an embedding table; other tensors whole."""
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


MERGE_COEFS = (0.75, 1.0, 1.25)  # multiples of each move in the joint search


@torch.no_grad()
def joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load,
                sweeps=2, shortlist=8):
    """The joint search over keys (rho, j, c) = final + c * move(rho, j) per part (None = the final weights):
    the best single key for every part, then coordinate sweeps (all keys per part, then each part's best `shortlist`)."""
    parts = sorted({merge_part(n) for n, _ in named})
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in MERGE_COEFS]
    cache = {}

    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]

    def state(choice):
        return [value(i, choice[merge_part(n)]) for i, (n, _) in enumerate(named)]
    best_key, best = None, plain
    for key in keys[1:]:
        load([value(i, key) for i in range(len(params))])
        loss = holdout_loss(model, rows, rank, world)
        if loss < best:
            best_key, best = key, loss
    choice = {p: best_key for p in parts}
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            tried = []
            for key in (keys if sweep == 0 else ranked[p]):
                if key == choice[p]:
                    tried.append((best, key))
                    continue
                trial = dict(choice)
                trial[p] = key
                load(state(trial))
                loss = holdout_loss(model, rows, rank, world)
                tried.append((loss, key))
                if loss < best:
                    best, choice = loss, trial
            if sweep == 0:
                ranked[p] = [k for _, k in sorted(tried, key=lambda x: x[0])[:shortlist]]
    load(state(choice))
    return dict(holdout_plain=plain, holdout_merged=best,
                merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]]) for p, k in choice.items()})


@torch.no_grad()
def end_merge(model, means, rows, rank, world):
    """The end merge (module docstring): per part, candidates final + c * (inside + rho * (full - inside)), full = an
    exponential mean minus the final weights and inside its part in the high-curvature subspace (curvature_bases on the
    held-out rows), chosen on the held-out loss by joint_parts. Every rank takes the same decisions (the losses are
    all-reduced)."""
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows, rank, world)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]

    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)

    def load(ws):
        for q, w in zip(params, ws):
            q.copy_(w.to(q.dtype))
    plain = holdout_loss(model, rows, rank, world)
    return joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load)


def train_stages(args, settings, device, rank, world):
    """Boost -> own-query Cut; each row is consumed exactly once, including across resume."""
    if world not in (1, 2, 4, 8):
        raise ValueError('Stage training supports 1, 2, 4 or 8 GPUs')
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout_rows, training = training[-HOLDOUT:], training[:-HOLDOUT]  # these feed no update, only the end merge
    means = None  # the exponential means of the end merge, one per MERGE_DECAYS entry
    schedule = stage_schedule(len(training), settings)
    steps = min(args.steps or len(schedule), len(schedule))
    stop = min(steps, args.stop_after_step or steps)
    split = max(1, world // settings.boost_batch)
    groups = ([torch.distributed.new_group(list(range(i, i+split))) for i in range(0, world, split)]
              if split > 1 else [None])
    group = groups[rank // split] if split > 1 else None
    cut_split = max(1, world // settings.cut_batch)  # GPUs per sequence in Cut
    cut_groups = groups if cut_split == split else (
        [torch.distributed.new_group(list(range(i, i+cut_split))) for i in range(0, world, cut_split)]
        if cut_split > 1 else [None])
    cut_group = cut_groups[rank // cut_split] if cut_split > 1 else None
    recipe_state = asdict(settings)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, seed=SEED, recipe=recipe_state, steps=steps,
            epoch_updates=len(schedule), train_rows=len(training),
            upstream='qlabs-eng/noise@98dbf6339c7768b2a6c2b9022aa25c80eb22a259',
            additions=['linear_path_scalars', 'input_key_probes', 'input_gate_probes',
                       'independent_future_hidden_query_embedding_probes', 'cut_batch4_split2', 'recipe_11',
                       'end_merge'],
            resume=str(args.resume.resolve()) if args.resume else None, torch_version=torch.__version__,
        ), indent=2)+'\n')

    def estimator_for(model, phase):
        if phase == 'boost':
            return Boost(model, settings, SEED, rank, world, split, group)
        return Cut(model, settings, SEED, rank, world, cut_split, cut_group)

    def rank_slice(batch, phase):
        group_size = split if phase == 'boost' else cut_split
        per = len(batch) // (world // group_size)
        first = (rank // group_size) * per
        return batch[first:first+per].to(device)

    # Compile every active phase on random tokens; recreate all model/optimizer/RNG state afterward.
    phases = {phase: horizon for phase, _, _, _, horizon in schedule}
    noise = torch.randint(0, settings.model.vocab_size,
                          (max(settings.boost_batch, settings.cut_batch), settings.model.sequence_len+1),
                          generator=torch.Generator().manual_seed(SEED))
    for phase, horizon in phases.items():
        model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
        optimizer, estimator = make_optimizer(model, settings, horizon), estimator_for(model, phase)
        size = settings.boost_batch if phase == 'boost' else settings.cut_batch
        batch = rank_slice(noise[:size], phase)
        estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        estimator.close()
        del model, optimizer, estimator
    torch.cuda.empty_cache()

    completed, previous_seconds = 0, 0.
    phase = schedule[0][0]
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        completed = checkpoint['step']
        if not 0 < completed < stop:
            raise ValueError('Resume checkpoint must precede the requested stop and follow at least one update')
        saved_phase, phase_step, _, seqs, _ = schedule[completed-1]
        expected = dict(phase=saved_phase, phase_step=phase_step, seqs=seqs, epoch_updates=len(schedule))
        if checkpoint.get('progress') != expected:
            raise ValueError('Resume data position or phase schedule differs')
        phase = saved_phase
        del checkpoint
    model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
    optimizer, estimator = make_optimizer(model, settings, phases[phase]), estimator_for(model, phase)
    if args.resume:
        completed, previous_seconds = restore_training_state(args.resume, model, optimizer, estimator, recipe=recipe_state)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, paused, elapsed = time.perf_counter(), 0., previous_seconds
    try:
        for step in range(completed+1, stop+1):
            current, phase_step, begin, end, horizon = schedule[step-1]
            if current != phase:
                estimator.close()
                next_optimizer = make_optimizer(model, settings, horizon)
                handoff_optimizer(optimizer, next_optimizer, phase_step)
                optimizer, estimator, phase = next_optimizer, estimator_for(model, current), current
            batch = rank_slice(training[begin:end], phase)
            set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.perf_counter()-start-paused
            record = dict(step=step, phase=phase, phase_step=phase_step, seqs=end,
                          tokens=end*settings.model.sequence_len, train_loss=loss, train_seconds=round(elapsed, 2),
                          peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3))
            if args.eval_every and step % args.eval_every == 0 and step < steps:
                pause = time.perf_counter()
                record['val_loss'] = evaluate(model.state_dict(), args.data_dir, model_file=MODEL_FILE)
                paused += time.perf_counter()-pause
            if rank == 0:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+'\n'); log.flush()
            if (step % MERGE_EVERY == 0 or step == stop) and begin / len(training) >= MERGE_FROM:
                with torch.no_grad():  # a resumed run restarts the means at its resumed weights
                    if means is None:
                        means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                    else:
                        for a, mean in zip(MERGE_DECAYS, means):
                            for m, q in zip(mean, model.parameters()):
                                m.lerp_(q.detach().float(), a)
            if (args.checkpoint_every and (step % args.checkpoint_every == 0 or step == stop)) or step == stop < steps:
                pause = time.perf_counter()
                save_training_state(args.output/'checkpoint.pt', model, optimizer, estimator, step=step,
                                    train_seconds=elapsed, recipe=recipe_state,
                                    progress=dict(phase=phase, phase_step=phase_step, seqs=end, epoch_updates=len(schedule)))
                paused += time.perf_counter()-pause
        if stop < steps:
            if rank == 0:
                (args.output/'segment.json').write_text(json.dumps(dict(step=stop, total_steps=steps, seqs=end,
                    train_seconds=elapsed, checkpoint=str((args.output/'checkpoint.pt').resolve())), indent=2)+'\n')
            return
        extra = {}
        if means:  # the end merge, timed with the training (the plain weights' validation is not)
            plain = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            pause = time.perf_counter()
            extra['val_loss_plain'] = evaluate(plain, args.data_dir, model_file=MODEL_FILE)
            paused += time.perf_counter()-pause
            merge_start = time.perf_counter()
            extra.update(end_merge(model, means, holdout_rows, rank, world))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter()-merge_start, 2)
            elapsed = previous_seconds + time.perf_counter()-start-paused
            if rank == 0:
                torch.save(plain, args.output/'final_plain.pt')
            means = None
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        result = dict(val_loss=evaluate(state, args.data_dir, model_file=MODEL_FILE), train_seconds=round(elapsed, 2), steps=stop,
                      seqs=end, tokens=end*settings.model.sequence_len, estimator=settings.estimator,
                      peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3), **extra)
        if rank == 0:
            torch.save(state, args.output/'final.pt')
            (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    finally:
        estimator.close()
        if log is not None:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/arch'))
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--steps', type=int, help='Diagnostic prefix; preserves the full-epoch LR schedule.')
    # Execution chunks only: the default command reproduces the logged algorithm and budgets.
    parser.add_argument('--chain-tokens', type=int, default=512)
    parser.add_argument('--key-chunk', type=int, default=32)
    parser.add_argument('--key-path-tokens', type=int, default=64)
    parser.add_argument('--future-tokens', type=int, default=128)
    parser.add_argument('--future-chunk', type=int, default=512)
    args = parser.parse_args()
    for key in ('steps', 'stop_after_step'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(key + ' must be positive')
    if min(args.checkpoint_every, args.eval_every) < 0:
        parser.error('Logging intervals must be nonnegative')
    settings = Settings(chain_tokens=args.chain_tokens, key_chunk=args.key_chunk,
                        key_path_tokens=args.key_path_tokens, future_tokens=args.future_tokens,
                        future_chunk=args.future_chunk)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    try:
        train_stages(args, settings, device, rank, world)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
