"""Boost -> Cut: forward-only pretraining in two phases (paper code; the benchmark preset is a track-1 entry).

Phase 1, Boost (stagewise). Every sub-block (each attention, each MLP) learns to lower the loss of the shared head
reading the stream right after it. A readout loss is a per-token function of one stream, so each token's credit is
measured exactly: forward differences along a full sign-flipped Hadamard cycle, each the actual soft-capped
cross-entropy of the head at the moved stream. The writers use that credit directly. The MLP hidden layer, queries,
keys, values, gates and value embeddings take one local step from it. The token embedding learns at the first
attention stage along its own path. The mixing scalars use central differences of their stage's loss.

Phase 2, Cut (end-to-end along strong routes). The estimand is the attention-cut gradient: credit reaches a site
through later MLPs, the identity path and the x0 injections, not through later attention. With every attention output
held at its clean value, a move of one token's stream reaches only that token's loss, so every token is its own
experiment, and a full cycle of moves measures its credit exactly. Each move runs forward through the later blocks
with every MLP restricted to the token's top-k units (by output contribution), the other units held at their clean
outputs. This is "Neural Networks Are Already Clean" (Cammarata 2026) read as a measurement design: the function, and
with it the credit, rides on each token's strong activations. Measured this way:
- the raw embedding and every block's mid stream (a full cycle per token, one-sided); block outputs from the same
  moves (a move of block d's output is a move of block d+1's mid stream scaled by its residual scalar);
- the top stream through the head alone; the mixing scalars by their own moves (central differences);
- one local step from those: writers, the MLP hidden layer, queries (a full cycle per head), keys (single-key
  experiments evaluated in closed form, no interference), values, gates and value embeddings (sampled).

Cut's attention externalities, by backward induction over depth (CutBI, the default). Splitting every path from a site
to the loss at its first later attention layer, the full credit is the attention-cut credit plus, at each later
attention layer, the credit of that layer's normed input stream through its own queries, keys, values and gate, read
through the attention-cut chain: what a token's keys and values do to the other tokens' losses (each token an agent;
the externality joins its own objective). That input credit is one local step from the layer's attention-output
credit, so measuring the sites from the top block down makes each layer's credit complete before the chains below need
it, and the same moves measure the full credit: <u, norm(xin + delta) - norm(xin)> joins each move's loss difference
wherever the move reaches a block's input. Values take the exact local step (A^T times the target). On the Cut record's
final weights, the estimate's cosine to the gradient over all block matrices rises from 0.83 to 0.99.

The model, init, optimizer (ANVIL II on block matrices, Adam elsewhere), rates and schedule shape are the Noise
record's. Boost runs at 4 sequences per update (two GPUs per sequence), Cut at 8 (one per GPU); the handoff keeps
the weights, the optimizer state and the data order.

    torchrun --standalone --nproc_per_node=8 train/val/run.py                                        # benchmark model
    torchrun --standalone --nproc_per_node=8 train/val/run.py --preset 243m --output logs/bc243m
"""
import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask  # noqa: E402
from utils.prepare_data import SEQ, TRAIN_ROWS, VAL_ROWS  # noqa: E402

SEED = 0
PRESETS = {'2m': (2, 128), '7m': (4, 256), '38m': (8, 512), '243m': (16, 1024)}  # the Dust paper's ladder
WIDTH, HIDDEN = 512, 2048  # set from the preset in main(), before anything compiles
EPS = torch.finfo(torch.float32).eps
POLAR_MARGIN = 1.1  # exact credit can be near rank 1; the record's 1.05 sits at the edge of the maps' convergence
torch._dynamo.config.cache_size_limit = 64


@dataclass
class Recipe:
    """The record's optimizer settings."""
    lr: float = 0.003       # Adam; embeddings 300x, mixing scalars 30x
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.3   # final fraction of the epoch with linear lr decay
    anvil_lr: float = 0.023
    anvil_wd: float = 2.25


@dataclass
class Settings:
    # Boost
    sigma: float = 0.005        # readout probe step, relative to the probed vector's RMS at that token
    probes: int = 0             # directions per token per readout (0: a full cycle, exact; fewer: a forward-gradient
                                # estimate along that many random orthogonal directions, the local-loss baseline)
    fc_k: int = 0               # MLP units probed per token (0: HIDDEN / 8)
    fc_sigma: float = 0.05
    local: dict = field(default_factory=lambda: dict(q=64, k=320, v=512, gate=64))  # attention-internal draws
    nrep: int = 64
    emb_chunk: int = 16
    scalar_step: float = 0.01
    chunk: int = 2 ** 27
    boost_exact_keys: bool = True
    # Cut
    chain_sigma: float = 0.01
    route_k: int = 0            # MLP units per token per block carrying a move (0: HIDDEN / 16; HIDDEN: no restriction)
    chain_tokens: int = 64
    cut_exact_keys: bool = True
    key_sigma: float = 0.003
    key_chunk: int = 8


# ---------------------------------------------------------------------------------------------------------- helpers

@torch.compile
def coordinate_differences(raw, y, sigma):
    """Finite differences of each token's loss along each of its own logits (the record's coordinate head): every
    moved loss is a full forward evaluation of the soft-capped cross-entropy, written in closed form."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


HADAMARDS = {}


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1."""
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    """The attention each token would see if only its own state moved: every query attends to the other positions'
    clean keys and values and to its own moved ones, by swapping its own clean key and value through the
    log-sum-exp of one attention over the clean keys and values. Tensors are [batch, token, head, dim]."""
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
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / WIDTH + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def row_derivatives(base, C, rows, y, steps, sq, dots, dir_sq):
    """Forward differences along each token's own directions (rows (n, K) of the step's basis): base (n, V); C (P, V);
    dots, dir_sq (n, K). The forward-gradient (local-loss) baseline's probes."""
    target = y.clamp(min=0)
    dirs = C[rows]
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = dirs.gather(2, target[:, None, None].expand(-1, rows.shape[1], 1)).squeeze(-1)
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / WIDTH + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * dirs) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    change = out.float() - baseline.float().unsqueeze(0)
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def value_scores(change, hub):
    return (change.float() * hub.float().permute(0, 2, 1, 3).unsqueeze(3)).sum(-1)


@torch.compile(dynamic=False)
def heads_first(move):
    n, b, t, h, hd = move.shape
    return move.to(torch.bfloat16).permute(1, 3, 2, 0, 4).reshape(b, h, t, n * hd)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return (credited.permute(3, 0, 2, 1).unsqueeze(-1) * a.float()).sum(0)


@torch.compile(dynamic=False)
def hidden_scores(ut, st, Hk, cols, g):
    moved = st.unsqueeze(-1) * Hk
    change = F.relu(ut.unsqueeze(1) + moved).square() - F.relu(ut.unsqueeze(1) - moved).square()
    stream_change = torch.bmm(change.to(torch.bfloat16), cols)
    scored = torch.bmm(stream_change, g.to(torch.bfloat16).unsqueeze(-1)).squeeze(-1).float()
    return scored / (2 * st)


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
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / WIDTH + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch.compile(dynamic=False)
def chain_chunk(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                target_logit_w, y):
    """Forward differences of each token's loss along every move of one site, attention held, MLPs restricted to
    each token's top-k units. site: -1 for the raw embedding, else the block whose mid stream moves. move (m, P, W);
    scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block (m, W); idx, hs per block
    (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block (H, W); weight_t (W, V);
    base (m, V) clean top-stream logits before the norm; target_logit_w (m, W). Returns (m, P)."""
    L = len(mids)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site
    for b in range(first, L):
        if b > first:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        dz = norm_difference(mids[b], delta)
        dh = torch.bmm(dz, w_fc[b][idx[b]].transpose(1, 2))
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm((up - down) * (up + down), w_out_t[b][idx[b]])
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / WIDTH + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def chain_chunk_bi(site, own, move, scale, e, x0, xins, mids, xL, idx, hs, us, lam_r, lam_0, w_fc, w_out_t, weight_t,
                   base, target_logit_w, y):
    """chain_chunk with the attention externalities: wherever a move reaches a block's input stream, that block's
    input credit u (m, W) scores the change of its normed input, <u, norm(xin + delta) - norm(xin)>, next to the
    token's own loss difference. xins per block (m, W) clean input streams; us per block (m, W). own: the move also
    reaches the site block's own input (its mixing scalars)."""
    L = len(mids)
    ext = lambda b, delta: (norm_difference(xins[b], delta) * us[b].unsqueeze(1)).sum(-1)
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
        score = ext(0, delta)
    else:
        dx0, delta, first = None, move, site
        score = ext(site, delta) if own else 0
    for b in range(first, L):
        if b > first:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
            score = score + ext(b, delta)
        dz = norm_difference(mids[b], delta)
        dh = torch.bmm(dz, w_fc[b][idx[b]].transpose(1, 2))
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm((up - down) * (up + down), w_out_t[b][idx[b]])
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / WIDTH + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return ((moved - clean) * (y >= 0).unsqueeze(1) + score) / scale


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


# ---------------------------------------------------------------------------------------------------- the clean pass

class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
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
            with torch.autocast('cuda', dtype=self.dtype):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast('cuda', dtype=self.dtype):
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


# ------------------------------------------------------------------------------------------------------------ Boost

class Boost:
    """Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)
        self.profile, self.times, self.last = False, {}, 0.0

    def close(self):
        self.capture.close()

    def tick(self, name):
        if not self.profile:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + now - self.last
        self.last = now

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
        return hadamard(WIDTH, self.device, torch.float32) * self.signs(WIDTH)

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
        offset = torch.randint(0, WIDTH, (n, 1), device=self.device, generator=self.generator)  # each token's
        # first row (used only by partial cycles; drawn either way so the random stream is the same)
        credit = torch.zeros(n, WIDTH, device=self.device)
        K = self.s.probes
        if K and K < WIDTH:  # the forward-gradient baseline: K consecutive rows per token from a random offset
            lo, hi = self.share(K)
            rows = (offset + torch.arange(lo, hi, device=self.device)) % WIDTH
            per = max(1, self.s.chunk // ((hi - lo) * vocab))
            for t0 in range(0, n, per):
                xs, ys, rs = x[t0:t0 + per], y[t0:t0 + per], rows[t0:t0 + per]
                derivative = row_derivatives(xs @ weight.T, C, rs, ys, steps[t0:t0 + per],
                                             xs.square().sum(-1, keepdim=True), (xs @ stream_dirs).gather(1, rs),
                                             dir_sq[rs])
                coefficients = torch.zeros(xs.shape[0], WIDTH, device=self.device).scatter_(1, rs, derivative)
                credit[t0:t0 + per] = coefficients @ H / K
            return self.group_sum(credit)
        lo, hi = self.share(WIDTH)
        C, stream_dirs, dir_sq, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq)
            credit[t0:t0 + per] = derivative @ Hs / WIDTH
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
        e = e.float().view(b, t, WIDTH)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, WIDTH, device=self.device)
        n = self.s.emb_chunk
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, WIDTH).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, WIDTH), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(WIDTH)
        for j0 in range(lo, hi, n):
            rows = H[j0:j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / WIDTH).reshape(b * t, WIDTH)

    def attention_maps(self, i):
        """Layer i's clean attention A[b, h, s, t] and the matched filter K[b, h, t, s] = A[s, t] / max_t' A[s, t']."""
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
        K = (A / A.amax(-1, keepdim=True)).transpose(-1, -2)
        self.maps[i] = (A.to(torch.bfloat16), K.to(torch.bfloat16).contiguous())
        return self.maps[i]

    def query_key_errors(self, kind, draws, targets, layers=None):
        """Query or key credit by the record's local step: draws move every position's raw query (or key), the
        attention reruns, each query's output change is scored against its target; key scores reach the moved
        positions through the matched filter."""
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = 0.1
        n = chunk = min(self.s.nrep, draws)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_' + kind
            A, K = self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q_raw), processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = attention(qp.to(dt), kp.to(dt), values.to(dt), window)
            rep = lambda z_: z_.to(dt).unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            fixed, vals = rep(kp if kind == 'q' else qp), rep(values)
            raw = q_raw if kind == 'q' else k_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = moved_heads(raw, a, sigma, cos, sin)
                qa, ka = (moved, fixed) if kind == 'q' else (fixed, moved)
                out = attention(qa, ka, vals, window).view(n, b, t, h, hd)
                score = head_scores(out, baseline, target)
                if kind == 'k':
                    score = (K.to(dt) @ score.to(dt)).float()
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors

    def exact_key_errors(self, targets, layers=None):
        """Key credit, one local step from the targets, by single-key experiments (key_scores): exact up to the
        one-sided step, with no interference between positions. The GPUs of a sequence split the directions."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd // 2] * cos + z[..., hd // 2:] * sin,
                                   z[..., :hd // 2] * -sin + z[..., hd // 2:] * cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        eye = torch.eye(hd, device=self.device)
        lo, hi = self.share(hd)
        errors, self.key_credit = {}, {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                heads = lambda name: d.outputs[prefix + name][r].float().view(t, h, hd).transpose(0, 1)
                q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
                if has_ve(i, L):
                    ve = d.outputs[f'value_embeds.{i}'][r].float().view(t, h, hd).transpose(0, 1)
                    gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][r].float().view(t, h)).T.unsqueeze(-1)
                    values = values + gate * ve
                qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
                logits = (qp @ kp.transpose(1, 2)) / hd ** 0.5
                A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
                hub = targets[i][r].view(t, h, hd).transpose(0, 1)
                C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
                steps = s.key_sigma * k_raw.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                for j0 in range(lo, hi, s.key_chunk):
                    j1 = min(j0 + s.key_chunk, hi)
                    credit_all[r, :, :, j0:j1] = key_scores(qp, kp, k_raw, steps, cos, sin, A, C, eye[j0:j1])
            self.key_credit[i] = credit_all
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def value_errors(self, kind, draws, targets):
        """Values ('v') or value-embedding gates ('gate') by the record's local step: every position's value moves
        along its own code, the attention (linear in its values, with the clean probabilities) carries the moves to
        every query's output, scored per head against the target and carried back by the matched filter."""
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        bf = torch.bfloat16
        errors = {}
        n = min(64, draws)
        for i in range(cfg.n_layer):
            if kind == 'gate' and not has_ve(i, cfg.n_layer):
                continue
            A, K = self.attention_maps(i)
            hub = targets[i].view(b, t, h, hd)
            site = f'transformer.h.{i}.attn.c_v' if kind == 'v' else f'transformer.h.{i}.attn.ve_gate'
            width = h if kind == 'gate' else cfg.n_embd
            w = width // h
            estimate = torch.zeros(b, t, h, w, device=self.device)
            if kind == 'gate':
                z = d.outputs[f'transformer.h.{i}.attn.ve_gate'].float()
                ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            for chunk in range(draws // n):
                a = d.orthogonal_noise(site, draws, chunk, n, b, t, width, dtype=torch.float32)
                if kind == 'v':
                    move = a.view(n, b, t, h, hd)
                else:
                    eps = 0.1
                    slope = (torch.sigmoid(z + eps * a) - torch.sigmoid(z - eps * a)) / eps
                    move = slope.unsqueeze(-1) * ve
                change = (A @ heads_first(move)).view(b, h, t, n, hd)
                score = value_scores(change, hub)
                credited = (K @ score.to(bf)).float()
                estimate += code_estimate(credited, a.view(n, b, t, h, w))
            errors[site] = (estimate / draws).reshape(b, t, width)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """MLP hidden credit, one local step from the measured credit g at the block output: each token's fc_k
        largest units move along a sign-flipped Hadamard cycle, through relu^2 and the output projection."""
        k = self.s.fc_k
        Hk = hadamard(k, self.device, torch.float32) * self.signs(k)
        top = u.topk(k, dim=-1, sorted=False).indices
        uk = u.gather(1, top)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        columns = out_weight.to(torch.bfloat16).T.contiguous()
        credit = torch.zeros_like(u)
        lo, hi = self.share(k)
        Hs = Hk[lo:hi]
        per = max(1, 2 ** 26 // (k * WIDTH))
        for t0 in range(0, u.shape[0], per):
            ut, st, tt, gt = uk[t0:t0 + per], steps[t0:t0 + per], top[t0:t0 + per], g[t0:t0 + per]
            cols = F.embedding(tt, columns)
            derivative = hidden_scores(ut, st, Hs, cols, gt)
            credit[t0:t0 + per].scatter_(1, tt, derivative @ Hs / k)
        return credit

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def internals(self, gradients, targets, b, t, exact_keys):
        """Queries, keys, values, gates and value embeddings, one local step from the targets."""
        d, s, L = self.capture, self.s, self.model.config.n_layer
        self.maps = {}
        for kind, count in s.local.items():
            if not count:
                continue
            if kind == 'k' and exact_keys:
                local = self.exact_key_errors(targets)  # each GPU of a sequence fills its share of directions
            else:
                draws = count // self.split
                local = self.value_errors(kind, draws, targets) if kind in ('v', 'gate') else \
                    self.query_key_errors(kind, draws, targets)
                local = {site: error / self.split for site, error in local.items()}  # the group averages
            for site, error in local.items():
                gradients[site + '.weight'] = d.reassemble(site, error)
                if kind == 'v' and has_ve(int(site.split('.')[2]), L):
                    i = int(site.split('.')[2])
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
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
            hub = self.probe(xm, yf, mix=block.attn.c_proj.weight, probed=att_in)
            targets[i] = hub.view(b, t, WIDTH)
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
        self.internals(gradients, targets, b, t, s.boost_exact_keys)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, WIDTH))
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
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()


# -------------------------------------------------------------------------------------------------------------- Cut

class Cut(Boost):
    """Phase 2: attention-cut credit measured end to end along each token's strong routes; one sequence per GPU."""

    def __init__(self, model, settings, seed, rank, world):
        super().__init__(model, settings, seed, rank, world, split=1, group=None)

    def chain_credits(self, e, x0, mids, outs, routes, y):
        """Attention-cut credit at the raw embedding ('emb') and every block's mid stream, and each block's
        mixing-scalar credits ('lam', i), measured: full cycles of moves per token, plus +-h moves of each scalar."""
        model, s = self.model, self.s
        L = model.config.n_layer
        vocab = model.config.vocab_size
        weight = model.lm_head.weight.float()[:vocab]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.float() for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.float().T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        credits = {}
        for site in [-1] + list(range(L)):
            H = self.basis()
            probed = e if site < 0 else mids[site]
            steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            credit = torch.empty(n, WIDTH, device=self.device)
            lam = torch.zeros(2, device=self.device)
            for t0 in range(0, n, m):
                sl = slice(t0, t0 + m)
                move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                scale = steps[sl].expand(-1, WIDTH)
                if site >= 0:  # the block's mixing scalars, central differences (a move along the stream itself
                    # barely changes the normed stream, so one-sided ones would be dominated by curvature)
                    prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                    move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1), -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                    scale = torch.cat([scale, torch.full_like(scale[:, :2], h), torch.full_like(scale[:, :2], -h)], 1)
                derivative = chain_chunk(
                    site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                    [r[1][sl] for r in routes], lam_r, lam_0, w_fc, w_out_t, weight_t, base[sl], target_w[sl], y[sl])
                credit[sl] = derivative[:, :WIDTH] @ H / WIDTH
                if site >= 0:
                    lam += (derivative[:, WIDTH:WIDTH + 2] + derivative[:, WIDTH + 2:]).sum(0) / 2
            credits['emb' if site < 0 else site] = credit
            if site >= 0:
                credits['lam', site] = lam
        return credits

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        if self.profile:
            torch.cuda.synchronize()
            self.last = time.perf_counter()
        d.capture(x, y)
        self.tick('forward')
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
            top = (r * mlp.c_proj.weight.float().norm(dim=0)).topk(s.route_k, dim=-1).indices
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        self.tick('routes')
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        self.tick('top')
        chain = self.chain_credits(e, x0, mids, outs, routes, yf)
        self.tick('chain')
        lam_r = model.resid_lambdas.float()
        targets = {}
        scalars = torch.zeros(2, L, device=self.device)
        for i in range(L):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            d_out = lam_r[i + 1] * chain[i + 1] if i < L - 1 else g_top
            gradients[p + 'attn.c_proj.weight'] = d_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = d_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            # the internals' local step through the attention module, scored as c_proj sees the mid-stream credit
            targets[i] = (d_mid @ block.attn.c_proj.weight.float()).view(b, t, WIDTH)
            scalars[:, i] = chain['lam', i]
        self.measured = dict(chain=chain, g_top=g_top, targets=targets)
        self.tick('writers_hidden')
        self.internals(gradients, targets, b, t, s.cut_exact_keys)
        self.tick('local')
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, WIDTH))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients


class CutBI(Cut):
    """Cut with the attention externalities, by backward induction over depth. Split every path from a site to the
    loss at its first later attention layer: the site's credit is Cut's attention-cut credit plus, at each later
    attention layer, the credit of that layer's normed input stream through its queries, keys, values and gate (one
    local step from the layer's attention-output credit; it reaches other positions' losses), read through the
    attention-cut chain. Measured from the top block down, each layer's input credit u exists before the chains below
    need it, so the same moves measure the full credit: <u, norm(xin + delta) - norm(xin)> joins each move's loss
    difference wherever the move reaches a block's input. Values take the exact local step (A^T times the target)."""

    def exact_values(self, i, target):
        """Layer i's value and gate errors, the local step in closed form: a value move at t reaches query s's output
        as A[s, t] times the move, so the value credit is sum_s A[s, t] target_s; a gate move scales the value
        embedding by the gate's slope (the record's central difference, odd in the move)."""
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        A, _ = self.attention_maps(i)  # A[b, h, s, t]
        credit = (A.float().transpose(-1, -2) @ target.view(b, t, h, hd).transpose(1, 2)).transpose(1, 2)
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors

    def layer_internals(self, i, target, gradients):
        """Layer i's query, key, value, gate and value-embedding gradients, one local step from its attention-output
        credit target (b, t, W). Returns the credit u (n, W) of the layer's normed input stream: each error through its
        projection (the error dotted with the projection of a move, for every move at once)."""
        d, s = self.capture, self.s
        b, t = d.batch, d.length
        targets = {i: target}
        errors = self.query_key_errors('q', s.local['q'], targets, layers=[i])
        errors.update(self.exact_key_errors(targets, layers=[i]))
        errors.update(self.exact_values(i, target))
        u = torch.zeros(b * t, WIDTH, device=self.device)
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            w = d.modules[site].weight.float()  # (out, in): the gate reads the first channels of the normed stream
            u[:, :w.shape[1]] += error.reshape(b * t, -1) @ w
        prefix = f'transformer.h.{i}.attn.'
        if has_ve(i, self.model.config.n_layer):  # a value move a is a value-embedding move a / gate
            gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'].float())
            ve_error = (errors[prefix + 'c_v'].view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
            gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
        return u

    def input_credit(self, xin, u):
        """Each token's credit at a block's input stream through the block's attention alone, measured: a full cycle
        of moves of the stream, each scored <u, norm(xin + move) - norm(xin)>."""
        H = self.basis()
        steps = self.s.chain_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        credit = torch.empty_like(xin)
        for t0 in range(0, len(xin), self.s.chain_tokens):
            sl = slice(t0, t0 + self.s.chain_tokens)
            move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
            credit[sl] = (norm_difference(xin[sl], move) * u[sl].unsqueeze(1)).sum(-1) / steps[sl] @ H / WIDTH
        return credit

    def site_chain(self, site, c, us, scalars=False):
        """Credit at one site (-1: the raw embedding; else block site's mid stream), measured as Cut's chain with the
        externalities of the blocks above; scalars: the block's mixing scalars instead (central differences of moves
        that reach the block's own input too)."""
        s = self.s
        n, m, h = c['e'].shape[0], s.chain_tokens, s.scalar_step
        H = None if scalars else self.basis()
        probed = c['e'] if site < 0 else c['mids'][site]
        steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        out = torch.zeros(2, device=self.device) if scalars else torch.empty(n, WIDTH, device=self.device)
        for t0 in range(0, n, m):
            sl = slice(t0, t0 + m)
            if scalars:
                prev = (c['x0'] if site == 0 else c['outs'][site - 1])[sl].unsqueeze(1)
                x0s = c['x0'][sl].unsqueeze(1)
                move = torch.cat([h * prev, h * x0s, -h * prev, -h * x0s], 1)
                scale = torch.cat([torch.full_like(steps[sl].expand(-1, 2), h),
                                   torch.full_like(steps[sl].expand(-1, 2), -h)], 1)
            else:
                move, scale = steps[sl].unsqueeze(-1) * H.unsqueeze(0), steps[sl].expand(-1, WIDTH)
            derivative = chain_chunk_bi(
                site, scalars, move, scale, c['e'][sl], c['x0'][sl], [z[sl] for z in c['xins']],
                [z[sl] for z in c['mids']], c['xL'][sl], [r[0][sl] for r in c['routes']],
                [r[1][sl] for r in c['routes']], [u[sl] for u in us], c['lam_r'], c['lam_0'], c['w_fc'], c['w_out_t'], c['weight_t'], c['base'][sl],
                c['target_w'][sl], c['y'][sl])
            if scalars:
                out += (derivative[:, :2] + derivative[:, 2:]).sum(0) / 2
            else:
                out[sl] = derivative @ H / WIDTH
        return out

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        xins, mids, outs, routes = [], [], [], []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i].float().flatten(0, 1)
            mid = xin + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            top = (r * mlp.c_proj.weight.float().norm(dim=0)).topk(s.route_k, dim=-1).indices
            xins.append(xin), mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        c = dict(e=e, x0=norm(e), xins=xins, mids=mids, outs=outs, routes=routes, xL=outs[-1], y=yf,
                 lam_r=model.resid_lambdas.float(), lam_0=model.x0_lambdas.float(),
                 w_fc=[blk.mlp.c_fc.weight.float() for blk in model.transformer.h],
                 w_out_t=[blk.mlp.c_proj.weight.float().T.contiguous() for blk in model.transformer.h],
                 weight_t=weight.T.to(torch.bfloat16).contiguous(), base=outs[-1] @ weight.T,
                 target_w=weight[yf.clamp(min=0)])
        us = [torch.zeros_like(e) for _ in range(L)]
        self.maps = {}
        inputs, scalars = {}, torch.zeros(2, L, device=self.device)
        for i in reversed(range(L)):  # backward induction: every block above is complete
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = self.site_chain(i, c, us)
            gradients[p + 'attn.c_proj.weight'] = d_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            us[i] = self.layer_internals(i, (d_mid @ block.attn.c_proj.weight.float()).view(b, t, WIDTH), gradients)
            scalars[:, i] = self.site_chain(i, c, us, scalars=True)
            if i:  # the block's input stream: the identity path to its mid stream, and its own attention
                inputs[i] = d_mid + self.input_credit(xins[i], us[i])
            d_out = c['lam_r'][i + 1] * inputs[i + 1] if i < L - 1 else g_top
            gradients[p + 'mlp.c_proj.weight'] = d_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
        emb = self.site_chain(-1, c, us)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, WIDTH))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients


class DFA(Cut):
    """Baseline (direct feedback alignment): Cut with every site's credit replaced by a fixed random projection of the
    exact output error (the coordinate head's logit credit), e @ B_site, B_site ~ N(0, 1 / V) drawn once from a fixed
    seed. Everything one local step from those credits is Cut's."""

    def __init__(self, model, settings, seed, rank, world):
        super().__init__(model, settings, seed, rank, world)
        V, L = model.config.vocab_size, model.config.n_layer
        g = torch.Generator(self.device).manual_seed(seed + 4242)
        self.feedback = {site: torch.randn(V, WIDTH, device=self.device, generator=g) / V ** 0.5
                         for site in ['emb'] + list(range(L)) + ['top']}

    def chain_credits(self, e, x0, mids, outs, routes, y):
        d, L = self.capture, self.model.config.n_layer
        error = d.head_error()[..., :self.model.config.vocab_size].flatten(0, 1).float()  # (n, V), exact
        credits = {site: error @ self.feedback[site] for site in ['emb'] + list(range(L))}
        for i in range(L):  # the mixing scalars by one local step from the feedback credit
            prev = x0 if i == 0 else outs[i - 1]
            credits['lam', i] = torch.stack([(credits[i] * prev).sum(), (credits[i] * x0).sum()])
        return credits

    def probe(self, x, y, mix=None, probed=None):  # the top stream's credit: feedback too
        error = self.capture.head_error()[..., :self.model.config.vocab_size].flatten(0, 1).float()
        return error @ self.feedback['top']


# -------------------------------------------------------------------------------------------- model and optimizer

def make_model(device, seed, config):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():  # output projections: uniform with std 1.5 / sqrt(fan_in) instead of zeros (the record's)
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)  # PR #360's total, warmup, cooldown and engage steps; scaled to the run length


def polar(g):
    """Approximate orthogonalization with six quintic maps."""
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (POLAR_MARGIN * x.norm() + 1e-6)).bfloat16()
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


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


def make_optimizer(model, recipe, steps):
    """ANVIL II on block matrices, Adam elsewhere."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW([{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
                             betas=recipe.betas, eps=1e-8, weight_decay=0.0)
    return Optimizers(Anvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps), adam)


@torch.no_grad()
def evaluate(model, rows, batch_size=16):
    """Mean next-token loss over the rows, split across ranks: utils/eval.py's computation, for any model size."""
    rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) \
        if torch.distributed.is_initialized() else (0, 1)
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in rows[rank::world].split(batch_size):
        batch = batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    if world > 1:
        torch.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def load_split(path, rows=None):
    data = torch.load(path, map_location='cpu', weights_only=True).long()
    assert data.shape[1] == SEQ and (rows is None or data.shape[0] >= rows), (path, tuple(data.shape))
    return data if rows is None else data[:rows]


# ------------------------------------------------------------------------------------------------------------ train

def main():
    global WIDTH, HIDDEN
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/val'))
    parser.add_argument('--preset', choices=sorted(PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--switch', type=float, default=0.3, help='Fraction of the epoch trained by Boost.')
    parser.add_argument('--route-k', type=int, default=0, help='Cut: MLP units per token per block (0: hidden/16).')
    parser.add_argument('--estimator', choices=('cut', 'cutbi', 'dfa'), default='cutbi', help='Phase 2 (cutbi: Cut '
                        'with the attention externalities by backward induction; dfa: the direct feedback alignment '
                        'baseline).')
    parser.add_argument('--keys', choices=('exact', 'sampled'), default='exact', help='Cut\'s keys.')
    parser.add_argument('--boost-keys', choices=('exact', 'sampled'), default='exact', help='Boost\'s keys.')
    parser.add_argument('--train-file', default='train.pt')
    parser.add_argument('--train-rows', type=int, default=TRAIN_ROWS)
    parser.add_argument('--test-file', default='', help='Also score the final weights on this split (e.g. test.pt).')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many sequences.')
    parser.add_argument('--boost-probes', type=int, default=0, help='Boost: directions per token per readout '
                        '(0: a full cycle; fewer: the forward-gradient local-loss baseline).')
    parser.add_argument('--chain-tokens', type=int, default=0)
    parser.add_argument('--key-chunk', type=int, default=0)
    parser.add_argument('--steps', type=int, default=0, help='Smoke test: stop after this many updates.')
    parser.add_argument('--save-every', type=int, default=0, help='Save the weights every this many sequences '
                        '(analysis: credit structure along training).')
    args = parser.parse_args()

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    n_layer, WIDTH = PRESETS[args.preset]
    HIDDEN = 4 * WIDTH
    config = GPTConfig(n_layer=n_layer, n_head=WIDTH // 64, n_kv_head=WIDTH // 64, n_embd=WIDTH)
    recipe = Recipe()
    settings = Settings(fc_k=HIDDEN // 8, route_k=args.route_k or HIDDEN // 16,
                        cut_exact_keys=args.keys == 'exact', boost_exact_keys=args.boost_keys == 'exact')
    if args.boost_probes:
        settings.probes = args.boost_probes
    if args.chain_tokens:
        settings.chain_tokens = args.chain_tokens
    if args.key_chunk:
        settings.key_chunk = args.key_chunk
    batch1, batch2 = 4, 8
    if world % batch1 and batch1 % world or batch2 % world:
        raise ValueError('Run on 1, 2, 4 or 8 GPUs.')
    split = world // batch1 if batch1 < world else 1
    per1, per2 = max(1, batch1 // world), batch2 // world
    groups = [None]
    if split > 1:
        groups = [torch.distributed.new_group(list(range(g * split, (g + 1) * split))) for g in range(world // split)]
    group = groups[rank // split] if split > 1 else None

    training = load_split(args.data_dir / args.train_file, args.train_rows)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    val_rows = load_split(args.data_dir / 'val.pt', VAL_ROWS)
    steps1, steps2 = len(training) // batch1, len(training) // batch2
    n1 = round(args.switch * steps1) // 2 * 2  # Boost updates, ending on a phase-2 batch boundary
    j0 = n1 * batch1 // batch2
    if args.steps:  # smoke test: a few updates of each phase
        n1, last = min(n1, args.steps), j0 + args.steps
    else:
        last = steps2
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, preset=args.preset, model=asdict(config), seed=args.seed,
            switch=args.switch, boost_steps=n1, cut_steps=last - j0, boost_batch=batch1, cut_batch=batch2,
            train_file=args.train_file, train_rows=len(training), recipe=asdict(recipe), settings=asdict(settings),
            polar_margin=POLAR_MARGIN, torch_version=torch.__version__,
        ), indent=2, default=str) + '\n')

    def boost_init():
        model = make_model(device, args.seed, config)
        return model, make_optimizer(model, recipe, steps1), Boost(model, settings, args.seed, rank, world, split, group)

    # untimed warmup on random tokens (compilation of both phases), then training from a fresh model
    noise = torch.randint(0, config.vocab_size, (batch2, SEQ), generator=torch.Generator().manual_seed(SEED)).to(device)
    model, optimizer, estimator = boost_init()
    first = (rank // split) * per1
    estimator.step(noise[first:first + per1, :-1].contiguous(), noise[first:first + per1, 1:].contiguous(), optimizer)
    estimator.close()
    model = make_model(device, args.seed, config)
    optimizer = make_optimizer(model, recipe, steps2)
    Phase2 = dict(cut=Cut, cutbi=CutBI, dfa=DFA)[args.estimator]
    estimator = Phase2(model, settings, args.seed, rank, world)
    mine = slice(rank * per2, (rank + 1) * per2)
    estimator.step(noise[mine, :-1].contiguous(), noise[mine, 1:].contiguous(), optimizer)
    estimator.close()
    del model, optimizer, estimator
    torch.cuda.empty_cache()
    model, optimizer, estimator = boost_init()

    if rank == 0:
        print(f'boostcut {args.preset}: Boost {n1} steps at batch {batch1}, Cut {last - j0} steps at batch {batch2}',
              flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0

    def set_rate(step, steps):
        left = (steps - step + 1) / steps
        for g in optimizer.param_groups:
            g.setdefault('initial_lr', g['lr'])
            g['lr'] = g['initial_lr'] * min(1.0, left / recipe.warmdown)

    def after(phase, step, seqs, loss):
        nonlocal paused, elapsed
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(phase=phase, step=step, seqs=seqs, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and seqs % args.eval_every == 0 and seqs < len(training):
            pause = time.perf_counter()
            record['val_loss'] = evaluate(model, val_rows)
            paused += time.perf_counter() - pause
        if args.save_every and seqs % args.save_every == 0:  # untimed, like validation
            pause = time.perf_counter()
            if rank == 0:
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / f'ckpt_{seqs}.pt')
            if world > 1:
                torch.distributed.barrier()
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()

    for step in range(1, n1 + 1):
        first = (step - 1) * batch1 + (rank // split) * per1
        batch = training[first:first + per1].to(device)
        set_rate(step, steps1)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        after('boost', step, step * batch1, loss)

    # the handoff: the same model, and the optimizer state under Cut's schedule at the same fraction of the epoch
    estimator.close()
    old = optimizer
    optimizer = make_optimizer(model, recipe, steps2)
    for new_o, old_o in zip(optimizer.optimizers, old.optimizers):
        for p, state in old_o.state.items():
            new_o.state[p] = state
    optimizer.optimizers[0].t = j0
    del old
    estimator = Phase2(model, settings, args.seed, rank, world)
    for step in range(j0 + 1, last + 1):
        batch = training[(step - 1) * batch2:step * batch2][mine].to(device)
        set_rate(step, steps2)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        after('cut', step, step * batch2, loss)

    estimator.close()
    result = dict(val_loss=evaluate(model, val_rows), train_seconds=round(elapsed, 2), boost_steps=n1,
                  cut_steps=last - j0, tokens=last * batch2 * (SEQ - 1))
    if args.test_file:
        result['test_loss'] = evaluate(model, load_split(args.data_dir / args.test_file))
    if rank == 0:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'final.pt')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
