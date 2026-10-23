"""Track 3's model: the 37.7M-parameter GPT of the Dust paper (8 blocks, width 512) with full-bandwidth feedback
and an n-gram memory.

Feedback follows Wang et al., "Full-bandwidth transformer" (arXiv 2608.08888), in the additive form of their
App. O.1: a first pass runs the plain model, then a second pass adds feedback(h[t-1]) to position t's token
embedding, h being the first pass's normalized top state. feedback starts at zero, so training starts from the
plain model.

The n-gram memory follows DeepSeek's Engram ("Conditional Memory via Scalable Lookup"), without its gate and
convolution: hashed rows for the last 2 and 3 tokens, added to the token embedding. It also starts at zero.
eval.py scores this module with --model train/arch/model.py.
"""
from dataclasses import dataclass
from functools import cache
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex = torch.compile(flex_attention)


@cache
def window_mask(window, length, device):
    return create_block_mask(
        lambda b, h, q, k: (q >= k) & (q - k <= window), None, None, length, length, device=device,
    )


def attention(q, k, v, window):
    """Causal attention with a left window; tensors have shape [batch, token, head, dim].

    A query at position s sees keys at s - window to s. Flex attention skips the blocks outside the
    window, several times faster than a dense masked kernel on the half-window blocks.
    """
    mask = window_mask(window, q.shape[1], q.device)
    return flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask).transpose(1, 2)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    # Per block, repeating: S = half-sequence window, L = full. The last block is always L.
    window_pattern: str = 'SSSL'
    # n-gram memory: n = 2..ngram_max, each hashed into ngram_heads tables of about ngram_rows rows (Engram's
    # 5x the vocabulary); ngram_max < 2 turns it off.
    ngram_max: int = 3
    ngram_heads: int = 8
    ngram_rows: int = 5 * 4096


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value embeddings on every other block, counting back from the last."""
    return layer_idx % 2 == (n_layer - 1) % 2


def next_prime(n):
    n += 1
    while any(n % p == 0 for p in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


class NgramMemory(nn.Module):
    """Hashed n-gram embeddings. For each n, every head hashes the last n tokens (XOR of token times an odd
    multiplier, mod its own prime) into its own table rows of width n_embd / heads; a position's memory is the
    sum over n of its heads' rows side by side. The hash is fixed, so the rows are a plain lookup table."""

    def __init__(self, config, seed=0):
        super().__init__()
        self.orders, self.heads = config.ngram_max - 1, config.ngram_heads
        self.pad = config.vocab_size  # stands for the tokens before the sequence start
        rng = random.Random(seed)
        self.multipliers = [2 * rng.randrange(2 ** 30) + 1 for _ in range(config.ngram_max)]
        self.primes, self.offsets, p, rows = [], [], config.ngram_rows, 0
        for _ in range(self.orders * self.heads):
            p = next_prime(p)
            self.primes.append(p)
            self.offsets.append(rows)
            rows += p
        self.table = nn.Embedding(rows, config.n_embd // self.heads)

    def ids(self, idx):
        """Table rows of each position, [batch, token, orders * heads], n = 2 first."""
        mix, out = idx * self.multipliers[0], []
        for k in range(1, self.orders + 1):
            mix = mix ^ (F.pad(idx[:, :-k], (k, 0), value=self.pad) * self.multipliers[k])
            for h in range(self.heads):
                j = (k - 1) * self.heads + h
                out.append(torch.remainder(mix, self.primes[j]) + self.offsets[j])
        return torch.stack(out, -1)

    def forward(self, idx):
        B, T = idx.shape
        return self.table(self.ids(idx)).view(B, T, self.orders, -1).sum(2)


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # The value embedding is scaled per head by 2 * sigmoid(gate), read from the block input's first 32
        # channels.
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = (norm(q), norm(k))
        y = attention(q, k, v, window_size[0])
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # Pre-norm block. attn.c_proj and mlp.c_proj are the "writers" into the residual stream.
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)  # untied from the embedding
        self.feedback = nn.Linear(config.n_embd, config.n_embd, bias=False)  # h[t-1] into position t's embedding
        self.ngram = NgramMemory(config) if config.ngram_max >= 2 else None
        # The stream entering block i is resid_lambdas[i] * x + x0_lambdas[i] * x0, a learned mix of the
        # previous block's output and the normalized token embedding.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.feedback.weight)
        if self.ngram is not None:
            torch.nn.init.zeros_(self.ngram.table.weight)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        # Token and value embeddings are stored in bfloat16 on the GPU
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            if self.ngram is not None:
                self.ngram.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def stack(self, e, idx, cos_sin):
        """The blocks on embedding e (the token embedding, plus feedback on the second pass), ending at the final norm."""
        x = norm(e)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        return norm(x)

    def embed(self, idx, previous=None):
        """The token embedding plus the n-gram memory, and on the second pass the feedback of h[t-1]."""
        e = self.transformer.wte(idx)
        if self.ngram is not None:
            e = e + self.ngram(idx).to(e.dtype)
        return e if previous is None else e + self.feedback(previous).to(e.dtype)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        h = self.stack(self.embed(idx), idx, cos_sin)
        previous = F.pad(h[:, :-1], (0, 0, 1, 0))  # position 0 has no previous token
        x = self.stack(self.embed(idx, previous), idx, cos_sin)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # soft cap
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
