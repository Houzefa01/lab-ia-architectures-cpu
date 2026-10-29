"""Candidat #16-intégré — Raisonneur Block-Local + Couche Mémoire parcimonieuse (savoir).
Une couche du milieu remplace son FFN par une MemoryLayer (product-key, N slots, K lus/token).
Le raisonnement reste dans block-local (RAM) ; le savoir vit dans la mémoire (grande capacité,
valeurs déportables sur disque, accès parcimonieux). Position-wise -> strictement causal.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candidats_actifs.block_local import _build_block_mask, BlockLocalAttention, FeedForward


class MemoryLayer(nn.Module):
    """Product-Key Memory position-wise : N=H*H slots, K lus par token (grad sparse)."""
    def __init__(self, dim, n_keys=256, k=16):
        super().__init__()
        self.dim, self.H, self.N, self.K = dim, n_keys, n_keys * n_keys, k
        self.KH = max(4, int(math.isqrt(k)) * 2)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k1 = nn.Parameter(torch.randn(n_keys, dim // 2) * 0.1)
        self.k2 = nn.Parameter(torch.randn(n_keys, dim // 2) * 0.1)
        self.values = nn.Embedding(self.N, dim, sparse=True)
        nn.init.normal_(self.values.weight, 0.0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).reshape(B * T, D)
        q1, q2 = q[:, :D // 2], q[:, D // 2:]
        s1, i1 = (q1 @ self.k1.t()).topk(self.KH, dim=1)
        s2, i2 = (q2 @ self.k2.t()).topk(self.KH, dim=1)
        cand = (s1[:, :, None] + s2[:, None, :]).view(B * T, -1)
        idx = (i1[:, :, None] * self.H + i2[:, None, :]).view(B * T, -1)
        sc, sel = cand.topk(self.K, dim=1)
        idx = torch.gather(idx, 1, sel)
        w = F.softmax(sc, dim=1)
        vals = self.values(idx)                              # (B*T, K, D) — seuls K slots lus
        return (w.unsqueeze(-1) * vals).sum(1).reshape(B, T, D)

    def bytes_read_per_token(self):
        return self.K * self.dim * 4


class MemBlock(nn.Module):
    """Bloc dont la 'FFN' est remplacée par la mémoire parcimonieuse."""
    def __init__(self, dim, heads, block_size, n_keys, k):
        super().__init__()
        self.attn = BlockLocalAttention(dim, heads, block_size)
        self.mem = MemoryLayer(dim, n_keys, k)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mem(self.norm2(x))
        return x


class StdBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, block_size):
        super().__init__()
        self.attn = BlockLocalAttention(dim, heads, block_size)
        self.ffn = FeedForward(dim, ffn_dim)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BlockLocalMemTransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, block_size=64, n_keys=256, k=16, mem_layer=None):
        super().__init__()
        self.vocab_size, self.dim, self.max_seq_len, self.block_size = vocab_size, dim, max_seq_len, block_size
        mem_layer = layers // 2 if mem_layer is None else mem_layer
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            MemBlock(dim, heads, block_size, n_keys, k) if i == mem_layer
            else StdBlock(dim, heads, ffn_dim, block_size) for i in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding) and m.weight.shape[0] == self.vocab_size:
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        mask = _build_block_mask((T + self.block_size - 1) // self.block_size, self.block_size, x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x))


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = BlockLocalMemTransformer(vocab_size=65, dim=64, heads=2, layers=3, ffn_dim=256,
                                 max_seq_len=256, block_size=64, n_keys=256, k=16)
    memL = [b for b in m.blocks if isinstance(b, MemBlock)][0].mem
    print(f"Params total: {count_parameters(m):,}  | slots mémoire (savoir): {memL.N:,}")
    print(f"Octets lus/token par la mémoire: {memL.bytes_read_per_token()/1024:.1f} Ko")

    print("\n1. Shapes");
    for T in [48, 100, 200]:
        assert m(torch.randint(0, 65, (2, T))).shape == (2, T, 65); print(f"  ✅ T={T}")

    print("2. Gradients")
    out = m(torch.randint(0, 65, (2, 128))); out.mean().backward()
    ok = all(p.grad is not None for n, p in m.named_parameters())
    print(f"  ✅ gradients présents (dont mémoire sparse): {ok}"); m.zero_grad()

    print("3. Audit causal (la mémoire ne doit pas casser la causalité)")
    m.eval()
    for T in [48, 100, 200]:
        s1 = torch.randint(0, 65, (1, T)); s2 = s1.clone(); s2[0, T//2:] = torch.randint(0, 65, (T-T//2,))
        with torch.no_grad():
            d = (m(s1)[0, :T//2] - m(s2)[0, :T//2]).abs().max().item()
        assert d < 1e-6, f"FUITE T={T}: {d}"; print(f"  ✅ causal T={T}: diff={d:.1e}")
    print("\n✅ Intégration raisonneur+mémoire validée structurellement")
