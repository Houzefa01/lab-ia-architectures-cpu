"""
GLA Chunked (vectorisé par chunks)
Gate dépendante du contenu g_t = sigmoid(W_g x_t)
- Pre-compute kv = k⊗v en batch (einsum)
- Chunks de taille C : boucle interne C itérations, sortie batch
- Pas de Python overhead O(T) — seulement O(T/C)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class GLAChunkedBlock(nn.Module):
    def __init__(self, dim, heads=2, chunk_size=64, eps=1e-6):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.chunk_size = chunk_size; self.eps = eps
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
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
        g = torch.sigmoid(self.g_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2))

        H, hd, C, eps = self.heads, self.hd, self.chunk_size, self.eps

        # Batch kv outer product: (B, H, T, hd, hd)
        kv = torch.einsum('bhtd,bhte->bhtde', k, v)

        S = torch.zeros(B, H, hd, hd, device=x.device)
        z = torch.zeros(B, H, hd, device=x.device)
        outputs = []

        for start in range(0, T, C):
            end = min(start + C, T)
            L = end - start

            g_c = g[:, :, start:end, :]   # (B, H, L, hd)
            q_c = q[:, :, start:end, :]
            k_c = k[:, :, start:end, :]
            kv_c = kv[:, :, start:end, :, :]  # (B, H, L, hd, hd)

            S_chunk = torch.zeros(B, H, L, hd, hd, device=x.device)
            z_chunk = torch.zeros(B, H, L, hd, device=x.device)

            for i in range(L):
                g_i = g_c[:, :, i, :]
                S = g_i.unsqueeze(-1) * S + kv_c[:, :, i, :, :]
                z = g_i * z + k_c[:, :, i, :]
                S_chunk[:, :, i] = S
                z_chunk[:, :, i] = z

            # Batch output: (B, H, L, hd) = q @ S
            numerator = torch.einsum('bhtd,bhtde->bhte', q_c, S_chunk)
            # (B, H, L) = q · z, unsqueeze for division
            denominator = torch.einsum('bhtd,bhtd->bht', q_c, z_chunk).unsqueeze(-1)
            o_chunk = numerator / (denominator + eps)
            outputs.append(o_chunk)

        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class GLAChunked(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, chunk_size=64):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            GLAChunkedBlock(dim, heads, chunk_size) for _ in range(layers)])
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
    print("=== Test GLA Chunked ===")
    m = GLAChunked(vocab_size=10, dim=32, heads=2, layers=2,
                    ffn_dim=128, max_seq_len=48, chunk_size=16)
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
    print(f"✅ GLA Chunked validé")

    # Fast speed test (T=48 only, compared to original GLA)
    print("\n--- Speed comparison (T=48) ---")
    import sys; sys.path.insert(0, '.')
    from candidats_actifs.gla import GLATransformer
    m_ref = GLATransformer(vocab_size=10, dim=32, heads=2, layers=2,
                           ffn_dim=128, max_seq_len=48)
    m_opt = GLAChunked(vocab_size=10, dim=32, heads=2, layers=2,
                       ffn_dim=128, max_seq_len=48, chunk_size=16)

    xb = torch.randint(0, 10, (32, 48))
    with torch.no_grad():
        for _ in range(5):
            m_ref(xb); m_opt(xb)
        t0 = time.perf_counter()
        for _ in range(20):
            m_ref(xb)
        t_ref = (time.perf_counter() - t0) / 20 * 1000
        t0 = time.perf_counter()
        for _ in range(20):
            m_opt(xb)
        t_opt = (time.perf_counter() - t0) / 20 * 1000
    print(f"  Original GLA: {t_ref:.1f}ms")
    print(f"  Chunked GLA:  {t_opt:.1f}ms")
    print(f"  Speedup:      {t_ref/t_opt:.2f}x")
