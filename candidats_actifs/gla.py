"""
GLA — Gated Linear Attention
Gate d'oubli dépendant du contenu g_t = sigmoid(W_g x_t)
État : S_t = diag(g_t)·S_{t-1} + k_t^T ⊗ v_t
Norm : z_t = g_t ⊙ z_{t-1} + k_t
Sortie : o_t = (q_t·S_t) / (q_t·z_t + ε)

Différence fondamentale avec RetNet : le decay dépend du TOKEN, pas de la position.
→ important : gate ≈ 1 ⟹ info préservée indéfiniment
→ bruit : gate ≈ 0 ⟹ état oublié, seule l'info courante compte
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class GLABlock(nn.Module):
    def __init__(self, dim, heads=2, ffn_dim=256):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)  # gate
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4, bias=False), nn.GELU(),
            nn.Linear(dim*4, dim, bias=False))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.eps = 1e-6

    def forward(self, x):
        B, T, D = x.shape
        residual = x; x = self.norm1(x)

        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)

        # Gate : g = sigmoid(Linear(x))  → (B, H, T, hd)
        g = torch.sigmoid(self.g_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2))

        H, hd = self.heads, self.hd
        S = torch.zeros(B, H, hd, hd, device=x.device)
        z = torch.zeros(B, H, hd, device=x.device)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]     # (B, H, hd)
            k_t = k[:, :, t, :]     # (B, H, hd)
            v_t = v[:, :, t, :]     # (B, H, hd)
            g_t = g[:, :, t, :]     # (B, H, hd)

            # Mise à jour : S = diag(g)·S + k^T ⊗ v
            # diag(g)·S : g_t * S  (broadcast)
            # k^T ⊗ v : k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)
            S = g_t.unsqueeze(-1) * S + k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)

            # Mise à jour du normaliseur
            z = g_t * z + k_t

            # Sortie : o = (q · S) / (q · z + ε)
            # q · S → (B, H, hd) : q @ S (contraction du dernier axe)
            numerator = torch.einsum('bhd,bhde->bhe', q_t, S)
            denominator = torch.einsum('bhd,bhd->bh', q_t, z).unsqueeze(-1)

            # En fait, q · S devrait être q_t @ S ce qui donne (B, H, hd)
            # Mais S est (B, H, hd, hd) donc q_t @ S donne (B, H, hd) ✓
            o_t = numerator / (denominator + self.eps)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class GLATransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            GLABlock(dim, heads, ffn_dim) for _ in range(layers)])
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
    print("=== Test GLA ===")
    m = GLATransformer(vocab_size=10, dim=32, heads=2, layers=2,
                        ffn_dim=128, max_seq_len=48)
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
    print("✅ GLA validé — prêt pour benchmark")
