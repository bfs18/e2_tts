from torch import nn
from dataclasses import dataclass
from typing import Optional
from rfwave.attention import (
    Attention, FeedForward, RMSNorm, apply_rotary_emb, get_pos_embed_indices,
    score_mask, precompute_freqs_cis, score_mask_from_bool_mask, modulate, _get_start)
from rfwave.dataset import get_num_tokens
from rfwave.modules import ConvNeXtV2Block

import torch
import math


class InputAdaptor(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, *args):
        return x


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-6
    max_seq_len: int = 8192
    dropout: float = 0.0
    qk_norm: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(dim=args.dim, num_heads=args.n_heads, qkv_bias=False, qk_norm=args.qk_norm,
                                   attn_drop=args.dropout, proj_drop=args.dropout, norm_layer=RMSNorm)
        # self.feed_forward = MLP(dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout,
        #                         act_layer=lambda: nn.GELU(approximate="tanh"))
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        # self.feed_forward = ConvFeedForward(
        #     dim=args.dim, hidden_dim=args.hidden_dim, drop=args.dropout, multiple_of=args.multiple_of)
        # self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class DurInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, n_attn_layers=4, dropout=0.):
        super().__init__()
        params = ModelArgs(dim=embedding_dim, n_layers=n_attn_layers, n_heads=8, dropout=dropout)
        self.dim = embedding_dim

        self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        # self.attn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.attn_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.attn_output = nn.Linear(params.dim, params.dim, bias=False)
        self.pad_token = 0
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("attn_freqs_cis", freqs_cis, persistent=False)
        freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, theta_rescale_factor=8.)
        self.register_buffer("attn_freqs_cis_eval", freqs_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('fc2.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_embed(self, start, length, scale=1.):
        # attn_freqs_cis = self.attn_freqs_cis if self.training else self.attn_freqs_cis_eval
        attn_freqs_cis = self.attn_freqs_cis
        pos = get_pos_embed_indices(start, length, max_pos=attn_freqs_cis.size(0), scale=scale)
        return attn_freqs_cis[pos]

    def forward(self, tokens: torch.Tensor):
        _bsz, num_phones = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        phone_start = torch.zeros([_bsz], dtype=torch.long, device=h.device)

        freqs_cis = self.get_pos_embed(phone_start, num_phones)
        # phone_mask = score_mask_from_bool_mask(tokens == self.pad_token)
        phone_mask = score_mask(get_num_tokens(tokens, self.pad_token))

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=phone_mask)
        h = self.attn_norm(h)
        h = self.attn_output(h)
        return h.transpose(1, 2)


class E2ECtxCharInputAdaptor(InputAdaptor):
    def __init__(self, embedding_dim, vocab_size, ctx_dim, num_layers=4):
        super().__init__()
        self.dim = embedding_dim
        self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ctx_proj = nn.Conv1d(ctx_dim, embedding_dim, kernel_size=1)
        self.tok_blocks = (nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])
                           if num_layers >= 1 else nn.Identity())
        self.ctx_blocks = (nn.Sequential(*[ConvNeXtV2Block(embedding_dim, embedding_dim*3) for _ in range(num_layers)])
                           if num_layers >= 1 else nn.Identity())
        self.register_buffer("freqs_cis", precompute_freqs_cis(embedding_dim, 1024), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, tokens, ctx):
        te = self.tok_embeddings(tokens)
        s = torch.zeros([tokens.size(0)], dtype=torch.long, device=tokens.device)
        pi = get_pos_embed_indices(s, tokens.size(1), max_pos=self.freqs_cis.size(0))
        pe = self.freqs_cis[pi]
        # te = apply_rotary_emb(te, pe)
        # use as absolute positional embedding.
        te = te + pe
        ce = self.ctx_proj(ctx)
        te = self.tok_blocks(te.transpose(1, 2))
        ce = self.ctx_blocks(ce)
        return torch.cat([te, ce], dim=2)
