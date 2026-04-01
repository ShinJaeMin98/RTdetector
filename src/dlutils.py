import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


def _rt_delta_bias_self(delta, seq_len):
    """(B, 1, S) or (B, S) -> additive bias (B, 1, S, S) for self-attention logits."""
    if delta.dim() == 3:
        d = delta.squeeze(1)
    else:
        d = delta
    if d.size(1) != seq_len:
        d = d[:, :seq_len]
    bias = d.unsqueeze(2) + d.unsqueeze(1)
    return bias.unsqueeze(1)


def _rt_delta_bias_cross(delta, mem_len):
    """Broadcast delta along memory positions: (B, 1, 1, Lm)."""
    if delta.dim() == 3:
        d = delta.squeeze(1)
    else:
        d = delta
    if d.size(1) < mem_len:
        d = F.pad(d, (0, mem_len - d.size(1)))
    elif d.size(1) > mem_len:
        d = d[:, :mem_len]
    return d.unsqueeze(1).unsqueeze(1)


def _head_dim(attn):
    return getattr(attn, "head_dim", attn.embed_dim // attn.num_heads)


def _proj_qkv_same(x, attn):
    """x: (L, B, E) -> q, k, v each (B, H, L, D)."""
    L, B, E = x.shape
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    nh = attn.num_heads
    hd = _head_dim(attn)
    q = q.view(L, B, nh, hd).permute(1, 2, 0, 3)
    k = k.view(L, B, nh, hd).permute(1, 2, 0, 3)
    v = v.view(L, B, nh, hd).permute(1, 2, 0, 3)
    return q, k, v


def _proj_qkv_cross(tgt, memory, attn):
    """Cross-attn: Q from tgt, K/V from memory. Matches nn.MultiheadAttention packed layout:
    w splits [E, 2E] — Q uses first E rows; K,V share the remaining 2E rows (see _in_projection_packed).
    """
    Lq, B, E = tgt.shape
    Lm = memory.size(0)
    nh = attn.num_heads
    hd = _head_dim(attn)
    w = attn.in_proj_weight
    b = attn.in_proj_bias
    w_q, w_kv = w.split([E, E * 2], dim=0)
    if b is None:
        b_q = b_kv = None
    else:
        b_q, b_kv = b.split([E, E * 2])
    q = F.linear(tgt, w_q, b_q)
    kv_proj = F.linear(memory, w_kv, b_kv)
    kv_proj = kv_proj.view(Lm, B, 2, E)
    k = kv_proj[:, :, 0, :].contiguous()
    v = kv_proj[:, :, 1, :].contiguous()
    q = q.view(Lq, B, nh, hd).permute(1, 2, 0, 3)
    k = k.view(Lm, B, nh, hd).permute(1, 2, 0, 3)
    v = v.view(Lm, B, nh, hd).permute(1, 2, 0, 3)
    return q, k, v


def _merge_heads(out, L, B, E, attn):
    """out: (B, H, L, D) -> (L, B, E)"""
    nh = attn.num_heads
    hd = _head_dim(attn)
    out = out.permute(2, 0, 1, 3).contiguous().view(L, B, E)
    return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


def _rt_attention_scores(q, k, attn_module, rt_tau, delta_bias, attn_mask_add=None):
    """logits = tau * (QK^T / sqrt(d)) + delta_bias (+ optional additive mask)."""
    d_k = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * d_k
    B = scores.size(0)
    tau = rt_tau
    if tau.dim() == 2:
        tau = tau.unsqueeze(-1)
    while tau.dim() < scores.dim():
        tau = tau.unsqueeze(-1)
    if tau.shape[0] != B:
        raise RuntimeError(
            f"RT tau batch {tau.shape[0]} does not match attention batch {B}"
        )
    scores = tau * scores
    if delta_bias is not None:
        scores = scores + delta_bias
    if attn_mask_add is not None:
        scores = scores + attn_mask_add
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=attn_module.dropout, training=attn_module.training)
    return attn


def rt_self_attention_forward(x, attn_module, rt_tau, rt_delta, causal=False):
    """RT-Attention on self-attention (Eq.13 style): softmax(tau * QK^T/sqrt(dk) + Delta) V."""
    L, B, E = x.shape
    q, k, v = _proj_qkv_same(x, attn_module)
    delta_bias = _rt_delta_bias_self(rt_delta, L)
    attn_mask_add = None
    if causal and L > 1:
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
        attn_mask_add = mask.view(1, 1, L, L)
    w = _rt_attention_scores(q, k, attn_module, rt_tau, delta_bias, attn_mask_add)
    out = torch.matmul(w, v)
    return _merge_heads(out, L, B, E, attn_module)


def rt_cross_attention_forward(tgt, memory, attn_module, rt_tau, rt_delta):
    """Cross-attention with per-memory-position bias from delta."""
    Lq, B, E = tgt.shape
    Lm = memory.size(0)
    q, k, v = _proj_qkv_cross(tgt, memory, attn_module)
    delta_bias = _rt_delta_bias_cross(rt_delta, Lm)
    w = _rt_attention_scores(q, k, attn_module, rt_tau, delta_bias, None)
    out = torch.matmul(w, v)
    return _merge_heads(out, Lq, B, E, attn_module)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, **kwargs):
        rt_tau = getattr(self, '_rt_tau', None)
        rt_delta = getattr(self, '_rt_delta', None)
        if rt_tau is not None and rt_delta is not None:
            src2 = rt_self_attention_forward(src, self.self_attn, rt_tau, rt_delta, causal=False)
        else:
            src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False, **kwargs):
        rt_tau = getattr(self, '_rt_tau', None)
        rt_delta_self = getattr(self, '_rt_delta_self', None)
        rt_delta_cross = getattr(self, '_rt_delta_cross', None)
        rt_delta = getattr(self, '_rt_delta', None)
        if rt_delta_self is None:
            rt_delta_self = rt_delta
        if rt_delta_cross is None:
            rt_delta_cross = rt_delta
        if rt_tau is not None and rt_delta_self is not None:
            Lq = tgt.size(0)
            causal = Lq > 1
            tgt2 = rt_self_attention_forward(tgt, self.self_attn, rt_tau, rt_delta_self, causal=causal)
        else:
            tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        if rt_tau is not None and rt_delta_cross is not None:
            tgt2 = rt_cross_attention_forward(tgt, memory, self.multihead_attn, rt_tau, rt_delta_cross)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
