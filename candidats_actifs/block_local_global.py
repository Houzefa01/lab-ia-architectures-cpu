"""
Candidat #15 — Block-Local-Global (attention à DOUBLE PORTÉE), méthode maison.
Objectif : mémoire bornée O(1) en longueur MAIS portée illimitée (n'importe quel contexte),
sans GPU, sans quitter les matmuls MKL.

Idée (inspirée d'Infini-attention Google 2024, mais avec partition propre + cumsum vectorisé) :
  o(t) = g ⊙ o_local(t)  +  (1-g) ⊙ o_global(t)
  - o_local  : softmax EXACT sur [bloc i-1, bloc i] (2C clés) — précision de près (= block-local #14).
  - o_global : attention LINÉAIRE compressive sur les blocs [0 .. i-2] — portée illimitée,
               via une mémoire compressive M (taille h×h FIXE par tête, indépendante de T).
  Partition sans chevauchement : local = [ (i-1)C , t ], global = [ 0 , (i-1)C ). Union = [0, t].

Coût : local O(T·C) (bmm) ; global O(T·h²) (bmm) + un cumsum sur l'axe des blocs (n=T/C ≤ 64,
PAS sur T -> trivial). Aucun unfold/einsum, aucune boucle Python sur T. Mémoire d'inférence
incrémentale : état M (h×h) + fenêtre 2C = O(1) en longueur de contexte.

Feature map phi(x)=elu(x)+1 (positif) pour l'attention linéaire (Katharopoulos 2020).
Gate g = sigmoid(beta) par tête, init beta=+2 -> ~88% local au départ (préserve l'accuracy #14).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candidats_actifs.block_local import _build_block_mask, FeedForward


class BlockLocalGlobalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 2, block_size: int = 64):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.head_dim, self.block_size = dim, heads, dim // heads, block_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.beta = nn.Parameter(torch.full((heads,), 2.0))  # gate local/global par tête
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H, h, C = self.heads, self.head_dim, self.block_size
        n = (T + C - 1) // C
        T_pad = n * C
        pad = T_pad - T

        qkv = self.qkv(x).reshape(B, T, 3, H, h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
        q_blk = q.reshape(B, H, n, C, h); k_blk = k.reshape(B, H, n, C, h); v_blk = v.reshape(B, H, n, C, h)

        # ---------- BRANCHE LOCALE (softmax exact, fenêtre [i-1, i]) ----------
        zero = torch.zeros(B, H, 1, C, h, dtype=k_blk.dtype, device=k_blk.device)
        k_prev = torch.cat([zero, k_blk[:, :, :n - 1]], dim=2)
        v_prev = torch.cat([zero, v_blk[:, :, :n - 1]], dim=2)
        k_ctx = torch.cat([k_prev, k_blk], dim=3)      # (B,H,n,2C,h)
        v_ctx = torch.cat([v_prev, v_blk], dim=3)
        scale = 1.0 / math.sqrt(h)
        qf = (q_blk * scale).reshape(B * H * n, C, h)
        scores = torch.bmm(qf, k_ctx.reshape(B * H * n, 2 * C, h).transpose(1, 2)).view(B, H, n, C, 2 * C)
        if mask is None:
            mask = _build_block_mask(n, C, x.device)
        scores = scores.masked_fill(mask.view(1, 1, n, C, 2 * C), float('-inf'))
        attn = F.softmax(scores, dim=-1).reshape(B * H * n, C, 2 * C)
        o_local = torch.bmm(attn, v_ctx.reshape(B * H * n, 2 * C, h)).reshape(B, H, T_pad, h)

        # ---------- BRANCHE GLOBALE (mémoire compressive, blocs [0, i-2]) ----------
        phi_k = F.elu(k_blk) + 1.0                          # (B,H,n,C,h) positif
        phi_q = F.elu(q_blk) + 1.0
        # contribution par bloc : dM_j = phi_k_j^T @ v_j (h×h), dz_j = sum_t phi_k_j
        dM = torch.bmm(phi_k.reshape(B * H * n, C, h).transpose(1, 2),
                       v_blk.reshape(B * H * n, C, h)).view(B, H, n, h, h)  # (B,H,n,h,h)
        dz = phi_k.sum(dim=3)                                # (B,H,n,h)
        # cumul jusqu'au bloc j inclus (cumsum sur l'axe des blocs, n petit)
        M_cum = torch.cumsum(dM, dim=2)                      # (B,H,n,h,h)
        z_cum = torch.cumsum(dz, dim=2)                      # (B,H,n,h)
        # mémoire pour le bloc i = cumul jusqu'à i-2 (décalage 2 ; i<2 -> 0) => partition propre
        z2 = torch.zeros(B, H, 2, h, h, dtype=M_cum.dtype, device=M_cum.device)
        M_used = torch.cat([z2, M_cum[:, :, :n - 2]], dim=2) if n > 2 else torch.zeros_like(M_cum)
        z2b = torch.zeros(B, H, 2, h, dtype=z_cum.dtype, device=z_cum.device)
        z_used = torch.cat([z2b, z_cum[:, :, :n - 2]], dim=2) if n > 2 else torch.zeros_like(z_cum)
        # sortie globale : (phi_q @ M) / (phi_q · z)
        pqf = phi_q.reshape(B * H * n, C, h)
        num = torch.bmm(pqf, M_used.reshape(B * H * n, h, h))                       # (B*H*n,C,h)
        den = torch.bmm(pqf, z_used.reshape(B * H * n, h, 1))                       # (B*H*n,C,1)
        o_global = (num / (den + self.eps)).reshape(B, H, T_pad, h)

        # ---------- COMBINAISON (gate par tête) ----------
        g = torch.sigmoid(self.beta).view(1, H, 1, 1)
        o = g * o_local + (1.0 - g) * o_global
        o = o[:, :, :T, :].transpose(1, 2).reshape(B, T, D)
        return self.proj(o)


class BlockLocalGlobalBlock(nn.Module):
    def __init__(self, dim, heads=2, ffn_dim=256, block_size=64):
        super().__init__()
        self.attn = BlockLocalGlobalAttention(dim, heads, block_size)
        self.ffn = FeedForward(dim, ffn_dim)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BlockLocalGlobalTransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, block_size=64):
        super().__init__()
        self.vocab_size, self.dim, self.max_seq_len, self.block_size = vocab_size, dim, max_seq_len, block_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([BlockLocalGlobalBlock(dim, heads, ffn_dim, block_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        C = self.block_size; n = (T + C - 1) // C
        mask = _build_block_mask(n, C, x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x))


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test Block-Local-Global (#15) ===")
    torch.manual_seed(0)
    m = BlockLocalGlobalTransformer(vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                                    max_seq_len=4096, block_size=64)
    print(f"Params: {count_parameters(m):,}")

    print("\n--- 1. Shapes ---")
    for T in [48, 100, 192, 300]:
        assert m(torch.randint(0, 10, (2, T))).shape == (2, T, 10); print(f"✅ T={T}")

    print("\n--- 2. Gradients ---")
    out = m(torch.randint(0, 10, (2, 200))); out.mean().backward()
    for nme, p in m.named_parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any(), f"grad KO {nme}"
    print("✅ gradients OK (dont beta gate)"); m.zero_grad()

    print("\n--- 3. Audit causal (CRITIQUE, global inclus) ---")
    m.eval()
    for T in [48, 100, 192, 256, 400]:
        s1 = torch.randint(0, 10, (1, T)); s2 = s1.clone(); s2[0, T//2:] = torch.randint(0, 10, (T - T//2,))
        with torch.no_grad():
            d = (m(s1)[0, :T//2] - m(s2)[0, :T//2]).abs().max().item()
        assert d < 1e-6, f"FUITE CAUSALE T={T}: {d}"; print(f"✅ causal T={T}: diff={d:.2e}")

    print("\n--- 4. PORTÉE GLOBALE : perturbation LOINTAINE (hors champ réceptif local) a un effet ---")
    # champ réceptif local (3 couches, C=64) ~ 192. On perturbe la position 5 (bloc 0) et on
    # regarde la position 380 (bloc 5). Via le GLOBAL (mémoire des blocs 0..3), effet attendu > 0.
    T = 448
    sb = torch.randint(0, 10, (1, T)); sp = sb.clone(); sp[0, 5] = (sb[0, 5] + 1) % 10
    with torch.no_grad():
        d_far = (m(sb)[0, 380] - m(sp)[0, 380]).abs().max().item()
    print(f"  effet d'une perturbation à pos=5 sur pos=380 (gap=375 >> RF local 192): diff={d_far:.2e}")
    assert d_far > 1e-6, "❌ la branche globale ne propage pas l'info lointaine"
    print("✅ portée globale ACTIVE (info au-delà du champ réceptif local est bien captée)")

    print("\n✅ Block-Local-Global (#15) validé structurellement")
