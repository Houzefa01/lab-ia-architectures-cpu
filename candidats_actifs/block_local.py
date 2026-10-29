"""
Block-Local Attention — attention causale par blocs contigus avec chevauchement
Chaque bloc de C=64 positions attend sur [bloc précédent ⧺ bloc courant] = 2C clés/valeurs,
via DEUX bmm batchés (torch.bmm) + un masque causal (n, C, 2C) construit une fois par forward
et partagé entre les couches. Aucun unfold/einsum/cumsum, aucune boucle Python sur T ou les blocs.
Voir docs/block_local_spec.md (candidat #14) pour la dérivation complète.

Champ réceptif : après L couches, position t dépend de [(floor(t/C) - L)*C, t] (clippé à 0),
soit un lookback garanti L*C = 192 >= gap copy (186) pour L=3, C=64.
À T <= 2C, block-local == attention causale pleine exacte (bloc 1 voit tout le bloc 0 + son
triangle causal).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


def _build_block_mask(n: int, C: int, device) -> torch.Tensor:
    """Masque booléen (n, C, 2C) — True = position masquée (-inf). Construit une fois
    par forward, partagé entre les couches (cf. spec §3, Annexe A point 3).

    Colonne j in [0, C) = bloc précédent (position globale (i-1)*C + j) ;
    colonne j in [C, 2C) = bloc courant (position globale i*C + (j-C)).
    Ligne r = position intra-bloc de la requête (position globale i*C + r).

    Bloc i >= 1 : gauche entièrement visible (toujours passé) ; droite triangulaire (j-C <= r).
    Bloc 0      : pas de prédécesseur -> gauche entièrement masquée ; droite triangulaire idem.
    """
    r = torch.arange(C, device=device).view(1, C, 1)
    j = torch.arange(2 * C, device=device).view(1, 1, 2 * C)

    vis_base = (j < C) | ((j - C) <= r)   # (1, C, 2C) — blocs i >= 1
    vis_0 = (j >= C) & ((j - C) <= r)     # (1, C, 2C) — bloc 0 (pas de bloc précédent)

    mask_0 = ~vis_0                                        # (1, C, 2C)
    mask_base = (~vis_base).expand(max(n - 1, 0), C, 2 * C)  # (n-1, C, 2C), vide si n==1

    return torch.cat([mask_0, mask_base], dim=0)  # (n, C, 2C) bool


class BlockLocalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 2, block_size: int = 64):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H, h, C = self.heads, self.head_dim, self.block_size
        n = (T + C - 1) // C
        T_pad = n * C
        pad_len = T_pad - T

        # 1-2. qkv + reshape/permute -> (B, H, T, h) x 3
        qkv = self.qkv(x).reshape(B, T, 3, H, h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. pad droit T -> T_pad (zéros), puis vue en blocs (B, H, n, C, h)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))

        q_blk = q.reshape(B, H, n, C, h)
        k_blk = k.reshape(B, H, n, C, h)
        v_blk = v.reshape(B, H, n, C, h)

        # 4. shift bloc précédent : un unique cat décalé sur l'axe des blocs
        zero_blk = torch.zeros(B, H, 1, C, h, dtype=k_blk.dtype, device=k_blk.device)
        k_prev = torch.cat([zero_blk, k_blk[:, :, : n - 1]], dim=2)  # (B, H, n, C, h)
        v_prev = torch.cat([zero_blk, v_blk[:, :, : n - 1]], dim=2)

        # 5. contexte = bloc précédent ++ bloc courant -> 2C clés/valeurs
        k_ctx = torch.cat([k_prev, k_blk], dim=3)  # (B, H, n, 2C, h)
        v_ctx = torch.cat([v_prev, v_blk], dim=3)

        # 6. aplatir pour bmm
        scale = 1.0 / math.sqrt(h)
        q_flat = (q_blk * scale).reshape(B * H * n, C, h)
        k_flat = k_ctx.reshape(B * H * n, 2 * C, h)
        v_flat = v_ctx.reshape(B * H * n, 2 * C, h)

        # 7. scores = (Q * scale) @ K^T -- sgemm batché
        scores = torch.bmm(q_flat, k_flat.transpose(1, 2))  # (B*H*n, C, 2C)
        scores = scores.view(B, H, n, C, 2 * C)

        # 8. masque causal (n, C, 2C), construit une fois par forward et partagé si fourni
        if mask is None:
            mask = _build_block_mask(n, C, x.device)
        scores = scores.masked_fill(mask.view(1, 1, n, C, 2 * C), float('-inf'))

        # 9. softmax
        attn = F.softmax(scores, dim=-1)
        attn_flat = attn.reshape(B * H * n, C, 2 * C)

        # 10. out = P @ V -- sgemm batché
        out = torch.bmm(attn_flat, v_flat)  # (B*H*n, C, h)

        # 11. dé-aplatir, transpose, tronquer au T réel
        out = out.reshape(B, H, T_pad, h)
        out = out.transpose(1, 2).reshape(B, T_pad, D)
        out = out[:, :T, :]

        # 12. proj
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockLocalBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 2, ffn_dim: int = 256, block_size: int = 64):
        super().__init__()
        self.attn = BlockLocalAttention(dim, heads, block_size)
        self.ffn = FeedForward(dim, ffn_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BlockLocalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        dim: int = 64,
        heads: int = 2,
        layers: int = 3,
        ffn_dim: int = 256,
        max_seq_len: int = 4096,
        block_size: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            BlockLocalBlock(dim, heads, ffn_dim, block_size) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        C = self.block_size
        n = (T + C - 1) // C
        mask = _build_block_mask(n, C, x.device)  # une fois par forward, partagé entre couches

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test Block-Local Attention ===")
    torch.manual_seed(0)

    model = BlockLocalTransformer(
        vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256, max_seq_len=4096, block_size=64
    )
    print(f"Paramètres (dim=64, heads=2, layers=3, ffn=256, max_seq_len=4096): {count_parameters(model):,}")

    # ------------------------------------------------------------------
    # 1. Shapes : (2,48)->(2,48,vocab), T non multiple de C (100), T<=C (48), T>2C (192)
    # ------------------------------------------------------------------
    print("\n--- Test 1 : Shapes ---")
    for T in [48, 100, 192]:
        x = torch.randint(0, 10, (2, T))
        out = model(x)
        assert out.shape == (2, T, 10), f"Shape incorrecte pour T={T}: {out.shape}"
        print(f"✅ Shape OK T={T}: {tuple(out.shape)}")

    # ------------------------------------------------------------------
    # 2. Gradients
    # ------------------------------------------------------------------
    print("\n--- Test 2 : Gradients ---")
    x = torch.randint(0, 10, (2, 48))
    out = model(x)
    out.mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient nul pour {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN pour {name}"
    print("✅ Gradients OK (tous les paramètres, non-None, non-NaN)")
    model.zero_grad()

    # ------------------------------------------------------------------
    # 3. Audit causal : diff < 1e-6 pour T in {48, 100, 192, 256}
    # ------------------------------------------------------------------
    print("\n--- Test 3 : Audit causal (CRITIQUE) ---")
    model.eval()
    for T in [48, 100, 192, 256]:
        seq1 = torch.randint(0, 10, (1, T))
        seq2 = seq1.clone()
        seq2[0, T // 2:] = torch.randint(0, 10, (T - T // 2,))
        with torch.no_grad():
            o1 = model(seq1)
            o2 = model(seq2)
        diff = (o1[0, : T // 2] - o2[0, : T // 2]).abs().max().item()
        assert diff < 1e-6, f"FUITE CAUSALE DÉTECTÉE T={T}: diff={diff:.6f}"
        print(f"✅ Causalité T={T}: diff={diff:.2e}")

    # ------------------------------------------------------------------
    # 4. Équivalence : à T <= 2C, block-local == attention causale pleine (mêmes poids)
    # ------------------------------------------------------------------
    print("\n--- Test 4 : Équivalence bloc-local ≡ attention causale pleine (T<=2C) ---")

    def full_causal_reference(m: BlockLocalTransformer, x: torch.Tensor) -> torch.Tensor:
        """Recalcule la sortie de m avec une attention causale pleine T×T standard,
        en réutilisant EXACTEMENT les mêmes poids (qkv/proj/ffn/norms/head) que m.
        Sert à prouver que le chemin bmm par blocs est numériquement correct."""
        B, T = x.shape
        pos = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0)
        h = m.token_emb(x) + m.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        for block in m.blocks:
            normed = block.norm1(h)
            Bn, Tn, D = normed.shape
            H, hd = block.attn.heads, block.attn.head_dim
            qkv = block.attn.qkv(normed).reshape(Bn, Tn, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            att = att.masked_fill(causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            attn_out = (att @ v).transpose(1, 2).reshape(Bn, Tn, D)
            attn_out = block.attn.proj(attn_out)
            h = h + attn_out
            h = h + block.ffn(block.norm2(h))

        h = m.norm(h)
        return m.head(h)

    T_eq = 100  # <= 2*block_size (128) et non multiple de C : régime "bloc 1 voit tout bloc 0"
    x_eq = torch.randint(0, 10, (2, T_eq))
    model.eval()
    with torch.no_grad():
        out_bl = model(x_eq)
        out_full = full_causal_reference(model, x_eq)
    diff_eq = (out_bl - out_full).abs().max().item()
    assert diff_eq < 1e-4, f"❌ Équivalence échouée à T={T_eq}: diff={diff_eq:.6f}"
    print(f"✅ Équivalence numérique à T={T_eq} (<=2C=128): diff_abs={diff_eq:.2e}")

    # ------------------------------------------------------------------
    # 5. Sanity fenêtrage actif à grand T : hors du champ réceptif -> aucun effet ;
    #    dans le champ réceptif -> effet mesurable (le mécanisme n'est pas mort/dégénéré).
    #    RF après L=3 couches : t dépend de [(floor(t/C)-3)*C, t] (clippé à 0).
    # ------------------------------------------------------------------
    print("\n--- Test 5 : Fenêtrage actif à grand T (T=512) ---")
    T_big = 512
    C = model.block_size
    t0 = 400  # bloc i0=6, r0=16 -> b(t0, 3 couches) = (6-3)*64 = 192
    pos_outside = 100   # < 192 : garanti HORS champ réceptif -> aucun effet
    pos_inside = 300    # dans [192, 400] : DANS le champ réceptif -> effet attendu

    seq_base = torch.randint(0, 10, (1, T_big))
    seq_outside = seq_base.clone()
    seq_outside[0, pos_outside] = (seq_base[0, pos_outside] + 1) % 10
    seq_inside = seq_base.clone()
    seq_inside[0, pos_inside] = (seq_base[0, pos_inside] + 1) % 10

    model.eval()
    with torch.no_grad():
        o_base = model(seq_base)
        o_outside = model(seq_outside)
        o_inside = model(seq_inside)

    diff_outside = (o_base[0, t0] - o_outside[0, t0]).abs().max().item()
    diff_inside = (o_base[0, t0] - o_inside[0, t0]).abs().max().item()

    assert diff_outside < 1e-6, f"❌ Fuite au-delà du champ réceptif garanti: diff={diff_outside:.6f}"
    assert diff_inside > 1e-6, f"❌ Fenêtrage semble inactif/dégénéré: diff dans le RF={diff_inside:.6f}"
    print(f"✅ Hors champ réceptif (pos={pos_outside} -> t={t0}): diff={diff_outside:.2e} (≈0, attendu)")
    print(f"✅ Dans le champ réceptif (pos={pos_inside} -> t={t0}): diff={diff_inside:.2e} (>0, attendu)")

    print("\n✅ Block-Local Attention validé — prêt pour comparaisons")
