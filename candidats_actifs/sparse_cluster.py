"""
Sparse Clustered Attention (SCA) — version soft, fully vectorisée
Pas de boucle Python sur T.
Chaque cluster = cumulatif de k⊗v pondéré par probabilité de routage.
Complexité O(T·hd²·K) — K=4 clusters.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class SparseClusteredBlock(nn.Module):
    def __init__(self, dim, heads=2, num_clusters=4, eps=1e-6):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.K = num_clusters; self.eps = eps
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.routing = nn.Linear(dim, num_clusters)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4, bias=False), nn.GELU(),
            nn.Linear(dim*4, dim, bias=False))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        residual = x; x = self.norm1(x)

        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)

        H, hd, K, eps = self.heads, self.hd, self.K, self.eps

        # Routage soft : (B, T, K)
        route = F.softmax(self.routing(x), dim=-1)

        # kv : (B, H, T, hd, hd)
        kv = torch.einsum('bhtd,bhte->bhtde', k, v)

        # État final : somme des états par cluster pondérés par le routage
        # Pour chaque cluster c, état cumulatif S_c[t] = sum_{s≤t} route[s,c]·kv[s]
        # Sortie : S[t] = sum_c route[t,c]·S_c[t]
        # → Équivalent à une GLA avec gate = route[t] (mais K états indépendants)

        S_out = torch.zeros(B, H, T, hd, hd, device=x.device)
        z_out = torch.zeros(B, H, T, hd, device=x.device)

        for c in range(K):
            w = route[:, :, c:c+1]       # (B, T, 1)
            w4 = w.unsqueeze(1)           # (B, 1, T, 1)
            w5 = w.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1, 1)

            kv_c = kv * w5
            k_c = k * w4

            S_c = torch.cumsum(kv_c, dim=2)
            z_c = torch.cumsum(k_c, dim=2)

            S_out = S_out + w5 * S_c
            z_out = z_out + w4 * z_c

        # Sortie
        numerator = torch.einsum('bhtd,bhtde->bhte', q, S_out)
        denominator = torch.einsum('bhtd,bhtd->bht', q, z_out).unsqueeze(-1)
        out = numerator / (denominator + eps)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class SparseClusteredTransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, num_clusters=4):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            SparseClusteredBlock(dim, heads, num_clusters) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        B, T = x.shape; assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x)


if __name__ == "__main__":
    import time
    print("=== Test SCA soft ===")
    m = SparseClusteredTransformer(vocab_size=10, dim=32, heads=2, layers=2,
                                    ffn_dim=128, max_seq_len=48, num_clusters=4)
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    x = torch.randint(0, 10, (2, 48)); out = m(x)
    assert out.shape == (2, 48, 10); print("✅ Shape OK")
    out.mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    print("✅ Gradients OK")
    s1 = torch.randint(0, 10, (1, 48)); s2 = s1.clone(); s2[0, 24:] = torch.randint(0, 10, (24,))
    m.eval()
    with torch.no_grad():
        o1, o2 = m(s1), m(s2)
    diff = (o1[0, :24] - o2[0, :24]).abs().max().item()
    assert diff < 1e-6; print(f"✅ Causalité OK (diff={diff:.2e})")
    print("✅ SCA soft validé")
