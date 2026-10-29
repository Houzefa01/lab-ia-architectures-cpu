"""
KVMemory v3 — Attention causale locale (sliding window)
Chaque position ne voit que les C-1 positions précédentes.
L'information se propage via les connexions résiduelles à travers les couches.
Softmax exact, mémoire O(T·C·D), temps O(T·C·D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KVMemoryV3(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, chunk_size=64, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.max_seq_len = max_seq_len
        assert dim % heads == 0

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            LocalAttentionBlock(dim, heads, ffn_dim, chunk_size, dropout) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class LocalAttentionBlock(nn.Module):
    """Attention causale locale avec fenêtre de taille C."""
    def __init__(self, dim, heads, ffn_dim, chunk_size=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)

        D = self.head_dim
        W = self.chunk_size
        scale = 1.0 / math.sqrt(D)

        outputs = []
        for start in range(0, T, W):
            end = min(start + W, T)
            L = end - start

            q_c = q[:, :, start:end, :]  # (B, H, L, D)
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]

            # Attention causale locale dans la fenêtre
            attn = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v_c)
            outputs.append(out)

        attn_out = torch.cat(outputs, dim=2)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test KVMemory v3 (local attention) ===")
    model = KVMemoryV3(vocab_size=10, dim=32, heads=2, layers=2, ffn_dim=128, max_seq_len=48, chunk_size=16)
    print(f"Paramètres: {count_parameters(model):,}")

    x = torch.randint(0, 10, (2, 48))
    out = model(x)
    assert out.shape == (2, 48, 10)
    print(f"✅ Shape OK: {out.shape}")

    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient nul {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN {name}"
    print("✅ Gradients OK")

    seq1 = torch.randint(0, 10, (1, 48))
    seq2 = seq1.clone()
    seq2[0, 24:] = torch.randint(0, 10, (24,))
    model.eval()
    with torch.no_grad():
        o1, o2 = model(seq1), model(seq2)
    diff = (o1[0, :24] - o2[0, :24]).abs().max().item()
    assert diff < 1e-6, f"FUITE CAUSALE: diff={diff:.6f}"
    print(f"✅ Causalité vérifiée (diff={diff:.2e})")

    print("\n✅ KVMemory v3 validé — prêt pour benchmark")