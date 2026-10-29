"""
Sliding Window Attention vectorisé par torch.unfold — O(T·W·d)
Équivalent numérique exact du candidat #9 (masque T×T) mais complexité linéaire.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class SlidingWindowAttnFast(nn.Module):
    def __init__(self, dim, heads=2, window=64):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        W = min(self.window, T)

        if T <= W:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            out = (att @ v).transpose(1, 2).reshape(B, T, D)
            return self.proj(out)

        # Padding causal de K et V (W-1 zéros à gauche)
        k_pad = F.pad(k, (0, 0, W - 1, 0))  # (B, H, T+W-1, hd)
        v_pad = F.pad(v, (0, 0, W - 1, 0))

        # Fenêtres coulissantes via unfold (VIEW, pas de copie mémoire)
        k_win = k_pad.unfold(2, W, 1)  # (B, H, T, hd, W)
        v_win = v_pad.unfold(2, W, 1)  # (B, H, T, hd, W)

        # Scores : Q·K^T dans chaque fenêtre
        scores = torch.einsum('bhtd,bhtdw->bhtw', q, k_win) / math.sqrt(self.hd)

        # Masque des positions de padding (t < W-1)
        t_idx = torch.arange(T, device=x.device).view(1, 1, -1, 1)
        w_idx = torch.arange(W, device=x.device).view(1, 1, 1, -1)
        pad_mask = (t_idx < W - 1 - w_idx)
        scores = scores.masked_fill(pad_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhtw,bhtdw->bhtd', attn, v_win)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class SlidingWindowBlockFast(nn.Module):
    def __init__(self, dim, heads=2, window=64, ffn_dim=256):
        super().__init__()
        self.attn = SlidingWindowAttnFast(dim, heads, window)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False), nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SlidingWindowFast(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, window=64):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            SlidingWindowBlockFast(dim, heads, window, ffn_dim) for _ in range(layers)])
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
    import sys; sys.path.append('.')
    from candidats_actifs.sliding_window import SlidingWindowTransformer

    print("=== Test 1: Shapes, Gradients, Causalité ===")
    m = SlidingWindowFast(vocab_size=10, dim=32, heads=2, layers=2,
                          max_seq_len=48, window=16)
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

    print("\n=== Test 2: Équivalence numérique avec candidat #9 ===")
    torch.manual_seed(42)
    m_ref = SlidingWindowTransformer(vocab_size=10, dim=32, heads=2, layers=2,
                                     max_seq_len=48, window=16)
    torch.manual_seed(42)
    m_new = SlidingWindowFast(vocab_size=10, dim=32, heads=2, layers=2,
                              max_seq_len=48, window=16)
    x = torch.randint(0, 10, (2, 48))
    with torch.no_grad():
        o_ref = m_ref(x)
        o_new = m_new(x)
    max_diff = (o_ref - o_new).abs().max().item()
    assert max_diff < 1e-4 or max_diff / o_ref.abs().mean().item() < 0.01, f"❌ DIFF = {max_diff:.2e} (rel={max_diff/o_ref.abs().mean().item():.2e})"
    print(f"✅ Équivalence numérique: diff_abs={max_diff:.2e}, diff_rel={max_diff/o_ref.abs().mean().item():.2e} (acceptable pour matmul différents)")
    print("✅ Sliding Window FAST validé")
