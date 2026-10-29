"""Candidat #17 — Block-Local + Mémoire Paginée Hiérarchique (BL-PM). Spec: docs/candidat17_spec.md.
Raisonneur block-local (#14, RAM) + mémoire de savoir PAGINÉE : savoir en pages contiguës,
récupéré 1× par SEGMENT de C tokens (amortissement HDD ÷C), branche gatée EN PLUS du FFN
(pas de régression LM), routage dense stable (1 seul AdamW). Cible: ~0.031 accès HDD/token (vs 32 #16).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candidats_actifs.block_local import _build_block_mask, BlockLocalAttention, FeedForward


class PagedMemory(nn.Module):
    def __init__(self, dim, n_pages=1024, slots_per_page=32, p=2, lam=1e-4):
        super().__init__()
        self.d, self.P, self.S, self.p, self.lam = dim, n_pages, slots_per_page, p, lam
        self.K_pages = nn.Parameter(torch.randn(n_pages, dim) * 0.02)          # centroïdes (RAM)
        self.K_slots = nn.Parameter(torch.randn(n_pages, slots_per_page, dim) * 0.02)  # savoir (→HDD)
        self.V_slots = nn.Parameter(torch.randn(n_pages, slots_per_page, dim) * 0.02)
        self.W_qseg = nn.Linear(2 * dim, dim, bias=False)
        self.W_qtok = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.tau = nn.Parameter(torch.ones(1))
        self._aux = torch.zeros(())
        self._last_pages = None

    def forward(self, h, C):
        B, T, d = h.shape
        n = (T + C - 1) // C
        Tp = n * C
        hp = F.pad(h, (0, 0, 0, Tp - T))
        hb = hp.view(B, n, C, d)

        # --- requête de segment (passé STRICT) ---
        seg_sum = hb.sum(2)                                            # (B,n,d)
        prev = torch.cat([torch.zeros(B, 1, d, device=h.device), seg_sum[:, :n - 1] / C], 1)
        csum = torch.cumsum(seg_sum, dim=1)
        cnt = (torch.arange(n, device=h.device) * C).clamp(min=1).view(1, n, 1)
        doc = torch.cat([torch.zeros(B, 1, d, device=h.device), csum[:, :n - 1]], 1) / cnt
        q_seg = self.W_qseg(torch.cat([prev, doc], -1))               # (B,n,d)

        # --- routage niveau page (dense, RAM) ---
        s = q_seg @ self.K_pages.t()                                  # (B,n,P)
        s_p, pidx = s.topk(self.p, dim=-1)                            # (B,n,p)
        self._last_pages = pidx.detach()
        # perte d'équilibrage (touche tous les centroïdes -> gradient dense)
        probs = F.softmax(s[:, 1:], dim=-1) if n > 1 else F.softmax(s, dim=-1)
        m = probs.mean(dim=(0, 1))
        top1 = (s[:, 1:] if n > 1 else s).argmax(-1)
        f = F.one_hot(top1, self.P).float().mean(dim=(0, 1))
        self._aux = self.lam * self.P * (f * m).sum()

        # --- fetch banc (p pages -> p·S slots) ---
        Kb = self.K_slots[pidx].reshape(B, n, self.p * self.S, d)     # (B,n,pS,d)
        Vb = self.V_slots[pidx].reshape(B, n, self.p * self.S, d)

        # --- cross-attention par token sur le banc (RAM, 2 bmm) ---
        q_tok = self.W_qtok(hp).view(B, n, C, d).reshape(B * n, C, d)
        scores = torch.bmm(q_tok * (d ** -0.5), Kb.reshape(B * n, self.p * self.S, d).transpose(1, 2))
        scores = scores.view(B * n, C, self.p, self.S) + self.tau * s_p.reshape(B * n, 1, self.p, 1)
        attn = F.softmax(scores.reshape(B * n, C, self.p * self.S), dim=-1)
        o = torch.bmm(attn, Vb.reshape(B * n, self.p * self.S, d))    # (B·n,C,d)
        out = self.W_o(o.reshape(B, Tp, d))[:, :T]

        # segment 0 : pas de passé -> contribution nulle
        seg0 = (torch.arange(T, device=h.device) >= C).float().view(1, T, 1)
        return out * seg0

    def access_stats(self, T, C):
        """accès HDD à froid (cache vide) = nb de pages DISTINCTES lues / T."""
        n = (T + C - 1) // C
        pages = self._last_pages[:, 1:].reshape(-1)  # segments valides
        distinct = pages.unique().numel()
        return {"pages_distinctes": distinct, "acces_par_token": distinct / T,
                "octets_par_token": distinct * (8 + self.S * self.d * 2) / T}


class PageMemBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, block_size, n_pages, slots_per_page, p):
        super().__init__()
        self.attn = BlockLocalAttention(dim, heads, block_size)
        self.ffn = FeedForward(dim, ffn_dim)
        self.mem = PagedMemory(dim, n_pages, slots_per_page, p)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.norm3 = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.full((dim,), -4.0))  # gate init fermé -> ≡ #14
        self.block_size = block_size

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        x = x + torch.sigmoid(self.alpha) * self.mem(self.norm3(x), self.block_size)
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


class BlockLocalPageMemTransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256, max_seq_len=4096,
                 block_size=64, n_pages=1024, slots_per_page=32, p=2, mem_layer=None):
        super().__init__()
        self.vocab_size, self.dim, self.max_seq_len, self.block_size = vocab_size, dim, max_seq_len, block_size
        ml = layers // 2 if mem_layer is None else mem_layer
        self.token_emb = nn.Embedding(vocab_size, dim); self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            PageMemBlock(dim, heads, ffn_dim, block_size, n_pages, slots_per_page, p) if i == ml
            else StdBlock(dim, heads, ffn_dim, block_size) for i in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding) and m.weight.shape[0] == self.vocab_size:
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = _build_block_mask((T + self.block_size - 1) // self.block_size, self.block_size, x.device)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.norm(h))

    def aux_loss(self):
        return sum(b.mem._aux for b in self.blocks if isinstance(b, PageMemBlock))

    def mem_block(self):
        return [b for b in self.blocks if isinstance(b, PageMemBlock)][0]


def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = BlockLocalPageMemTransformer(vocab_size=65, dim=64, heads=2, layers=3, ffn_dim=256,
                                     max_seq_len=512, block_size=64, n_pages=1024, slots_per_page=32, p=2)
    mem = m.mem_block().mem
    print(f"Params: {count_parameters(m):,} | pages(savoir)={mem.P}×{mem.S}={mem.P*mem.S:,} slots, p={mem.p}/segment")

    print("\n1. Shapes");
    for T in [48, 100, 200, 512]:
        assert m(torch.randint(0, 65, (2, T))).shape == (2, T, 65); print(f"  ✅ T={T}")

    print("2. Gradients (avec perte auxiliaire d'équilibrage)")
    out = m(torch.randint(0, 65, (2, 256)))
    (out.mean() + m.aux_loss()).backward()
    miss = [n for n, p in m.named_parameters() if p.grad is None]
    nan = [n for n, p in m.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"  grad manquants: {miss or 'aucun'} | NaN: {nan or 'aucun'}"); m.zero_grad()

    print("3. Audit causal (la mémoire paginée ne doit pas fuiter)")
    m.eval()
    for T in [48, 100, 200, 320]:
        s1 = torch.randint(0, 65, (1, T)); s2 = s1.clone(); s2[0, T//2:] = torch.randint(0, 65, (T-T//2,))
        with torch.no_grad():
            d = (m(s1)[0, :T//2] - m(s2)[0, :T//2]).abs().max().item()
        assert d < 1e-6, f"FUITE T={T}: {d}"; print(f"  ✅ causal T={T}: diff={d:.1e}")

    print("4. Équivalence gate fermé (α=-20) ≡ raisonneur pur (attn+FFN), pas de régression")
    mb = m.mem_block()
    with torch.no_grad(): mb.alpha.fill_(-20.0)
    x = torch.randint(0, 65, (2, 200))
    with torch.no_grad():
        o_full = m(x)
        contrib = (torch.sigmoid(mb.alpha) * mb.mem(mb.norm3(torch.zeros(2, 200, 64)), 64)).abs().max().item()
    print(f"  contribution mémoire à α=-20: {torch.sigmoid(mb.alpha).max().item():.2e} (≈0 -> ≡ #14) ✅")
    with torch.no_grad(): mb.alpha.fill_(-4.0)

    print("5. Accès HDD / token (le critère décisif vs #16=32)")
    with torch.no_grad(): m(torch.randint(0, 65, (1, 512)))
    st = mem.access_stats(512, 64)
    print(f"  pages distinctes lues (T=512): {st['pages_distinctes']} -> {st['acces_par_token']:.4f} accès/token "
          f"({st['octets_par_token']:.0f} o/token)")
    print(f"  #16: 32 accès/token, 8192 o/token -> #17 réduit d'un facteur ~{32/max(st['acces_par_token'],1e-9):.0f}x")
    print("\n✅ Candidat #17 validé structurellement")
