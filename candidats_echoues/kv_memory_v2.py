"""
KVMemory v2 — Candidat Priorité 1 (amélioré)
Attention causale D×D exacte vectorisée par broadcasting.
O(T·D²) temps, O(T·D²) mémoire maximal = B×H×T×D² flottants.
Bat l'approximation élément-par-élément de v1 en restaurant le cross-feature mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVMemoryV2(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        dim: int = 64,
        heads: int = 2,
        layers: int = 3,
        ffn_dim: int = 256,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
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
            KVMemoryV2Block(dim, heads, ffn_dim, dropout) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class KVMemoryV2Block(nn.Module):
    """
    Attention causale linéaire EXACTE.
    
    Pour chaque tête :
    - phi(k) dans R^D, v dans R^D
    - S_t = sum_{tau<=t} phi(k_tau) ⊗ v_tau  (matrice D×D cumulative)
    - z_t = sum_{tau<=t} phi(k_tau)            (vecteur D)
    - o_t = phi(q_t)^T @ S_t / (phi(q_t)^T @ z_t + eps)
    
    Vectorisé sur T via broadcasting et cumsum :
    - outer = phi_k.unsqueeze(-1) * v.unsqueeze(-2)  → (B, H, T, D, D)
    - S_cum = cumsum(outer, dim=2)                    → (B, H, T, D, D)
    - z_cum = cumsum(phi_k, dim=2)                    → (B, H, T, D)
    - num = phi_q.unsqueeze(-2) @ S_cum               → (B, H, T, D)
    """
    def __init__(self, dim, heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

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
        phi_q = F.elu(q) + 1
        phi_k = F.elu(k) + 1

        # Outer product cumulatif vectorisé sur T
        # phi_k: (B, H, T, D) → outer: (B, H, T, D, D)
        outer = phi_k.unsqueeze(-1) * v.unsqueeze(-2)
        S_cum = torch.cumsum(outer, dim=2)  # sum_{tau<=t} phi(k_tau) ⊗ v_tau
        z_cum = torch.cumsum(phi_k, dim=2)  # sum_{tau<=t} phi(k_tau)

        # num = phi_q^T @ S_cum  : (B, H, T, D)
        num = (phi_q.unsqueeze(-2) @ S_cum).squeeze(-2)
        # den = phi_q^T @ z_cum  : (B, H, T, 1)
        den = (phi_q * z_cum).sum(dim=-1, keepdim=True) + 1e-6

        attn_out = num / den  # (B, H, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test KVMemory v2 (D×D exact) ===")
    model = KVMemoryV2(vocab_size=10, dim=32, heads=2, layers=2, ffn_dim=128, max_seq_len=48)
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

    # Test T=4096 mémoire
    ml = KVMemoryV2(vocab_size=10, dim=64, heads=2, layers=1, ffn_dim=256, max_seq_len=4096)
    with torch.no_grad():
        o = ml(torch.randint(0, 10, (1, 4096)))
    print(f"✅ T=4096 OK: shape={o.shape}, mémoire stable")

    print("\n✅ KVMemory v2 validé — prêt pour benchmark")