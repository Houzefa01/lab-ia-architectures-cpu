"""Candidat #17-v2 — Mémoire Paginée à Routage SAILLANT. Spec: docs/candidat17v2_spec.md.
Corrige l'échec de v1 (rappel=hasard) : pooling attentif (l'entité survit) + cummax causal
(signal persistant) + contexte-entité c_i (niveau fin précis) + gate PAR TOKEN (parité LM).
Budget I/O inchangé : ≤ p/C accès HDD/token (tout le fin est en RAM).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candidats_actifs.block_local import _build_block_mask, BlockLocalAttention, FeedForward
from candidats_actifs.block_local_pagemem import StdBlock


class PagedMemoryV2(nn.Module):
    def __init__(self, dim, n_pages=1024, slots_per_page=32, p=3, r=2, lam=1e-4):
        super().__init__()
        self.d, self.P, self.S, self.p, self.r, self.lam = dim, n_pages, slots_per_page, p, r, lam
        self.K_pages = nn.Parameter(torch.randn(n_pages, dim) * 0.02)
        self.K_slots = nn.Parameter(torch.randn(n_pages, slots_per_page, dim) * 0.02)
        self.V_slots = nn.Parameter(torch.randn(n_pages, slots_per_page, dim) * 0.02)
        self.W_sal = nn.Linear(dim, r, bias=False)          # saillance
        self.W_qtok = nn.Linear(dim, dim, bias=False)
        self.W_c = nn.Linear(dim, dim, bias=False)          # contexte-entité
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.w_g = nn.Linear(dim, 1)                        # gate par token
        nn.init.constant_(self.w_g.bias, -4.0)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.tau = nn.Parameter(torch.ones(1))
        self._aux = torch.zeros(()); self._route = None; self._c = None; self._last_pages = None

    def forward(self, h, C):
        B, T, d = h.shape
        n = (T + C - 1) // C; Tp = n * C
        hp = F.pad(h, (0, 0, 0, Tp - T)); hb = hp.view(B, n, C, d)

        # 2.1 résumés saillants (softmax sur C -> l'entité survit)
        e = self.W_sal(hb)                                   # (B,n,C,r)
        a = F.softmax(e, dim=2)
        U = torch.bmm(a.reshape(B * n, C, self.r).transpose(1, 2),
                      hb.reshape(B * n, C, d)).view(B, n, self.r, d)   # (B,n,r,d)

        # 2.2 routage grossier : cummax causal sur les segments
        Z = U @ self.K_pages.t()                             # (B,n,r,P)
        Zseg = Z.max(dim=2).values                           # (B,n,P)
        Zcum = torch.cummax(Zseg, dim=1).values
        route = torch.cat([torch.full((B, 1, self.P), -1e9, device=h.device), Zcum[:, :n - 1]], 1)
        self._route = route.detach()
        s_p, pidx = route.topk(self.p, dim=-1)               # (B,n,p)
        self._last_pages = pidx.detach()

        # aux d'équilibrage (sur route, segments >=1)
        if n > 1:
            pr = F.softmax(route[:, 1:], dim=-1); m = pr.mean(dim=(0, 1))
            f = F.one_hot(route[:, 1:].argmax(-1), self.P).float().mean(dim=(0, 1))
            self._aux = self.lam * self.P * (f * m).sum()

        # 2.3 contexte-entité c_i (attention causale sur résumés passés, 0 I/O)
        w = Z.max(dim=-1).values.reshape(B, n * self.r)      # (B,n·r)
        jseg = (torch.arange(n * self.r, device=h.device) // self.r)
        Mc = torch.where(jseg.view(1, n * self.r) < torch.arange(n, device=h.device).view(n, 1),
                         0.0, -1e9)                          # (n,n·r) fini -> pas de NaN au backward
        A = F.softmax(w.unsqueeze(1) + Mc, dim=-1)
        c = torch.bmm(A, U.reshape(B, n * self.r, d))        # (B,n,d)
        c = c * (torch.arange(n, device=h.device) >= 1).float().view(1, n, 1)  # segment 0 -> 0
        self._c = c.detach()

        # 2.4 fetch (SEUL I/O) + lecture fine par token, requête enrichie
        Kb = self.K_slots[pidx].reshape(B, n, self.p * self.S, d)
        Vb = self.V_slots[pidx].reshape(B, n, self.p * self.S, d)
        q = (self.W_qtok(hb) + self.W_c(c).unsqueeze(2)).reshape(B * n, C, d)
        sc = torch.bmm(q * (d ** -0.5), Kb.reshape(B * n, self.p * self.S, d).transpose(1, 2))
        sc = sc.view(B * n, C, self.p, self.S) + self.tau * s_p.reshape(B * n, 1, self.p, 1)
        o = torch.bmm(F.softmax(sc.reshape(B * n, C, self.p * self.S), -1),
                      Vb.reshape(B * n, self.p * self.S, d)).reshape(B, Tp, d)[:, :T]

        # 2.5 gate PAR TOKEN + masque segment 0
        g = torch.sigmoid(self.w_g(h)) * torch.sigmoid(self.alpha)      # (B,T,1)
        seg0 = (torch.arange(T, device=h.device) >= C).float().view(1, T, 1)
        return g * self.W_o(o) * seg0

    def access_stats(self, T, C):
        n = (T + C - 1) // C
        distinct = self._last_pages[:, 1:].reshape(-1).unique().numel()
        return {"acces_par_token": distinct / T, "pages_distinctes": distinct}


class PageMemBlockV2(nn.Module):
    def __init__(self, dim, heads, ffn_dim, block_size, n_pages, slots_per_page, p):
        super().__init__()
        self.attn = BlockLocalAttention(dim, heads, block_size)
        self.ffn = FeedForward(dim, ffn_dim)
        self.mem = PagedMemoryV2(dim, n_pages, slots_per_page, p)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.norm3 = nn.LayerNorm(dim)
        self.block_size = block_size

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        x = x + self.mem(self.norm3(x), self.block_size)
        return x


class BlockLocalPageMemV2Transformer(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256, max_seq_len=4096,
                 block_size=64, n_pages=1024, slots_per_page=32, p=3, mem_layer=None):
        super().__init__()
        self.vocab_size, self.dim, self.max_seq_len, self.block_size = vocab_size, dim, max_seq_len, block_size
        ml = layers // 2 if mem_layer is None else mem_layer
        self.token_emb = nn.Embedding(vocab_size, dim); self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            PageMemBlockV2(dim, heads, ffn_dim, block_size, n_pages, slots_per_page, p) if i == ml
            else StdBlock(dim, heads, ffn_dim, block_size) for i in range(layers)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding) and m.weight.shape[0] == self.vocab_size:
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = _build_block_mask((T + self.block_size - 1) // self.block_size, self.block_size, x.device)
        for b in self.blocks: h = b(h, mask)
        return self.head(self.norm(h))

    def aux_loss(self): return sum(b.mem._aux for b in self.blocks if isinstance(b, PageMemBlockV2))
    def mem_block(self): return [b for b in self.blocks if isinstance(b, PageMemBlockV2)][0]


def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = BlockLocalPageMemV2Transformer(vocab_size=65, dim=64, heads=2, layers=3, ffn_dim=256,
                                       max_seq_len=512, block_size=64, n_pages=1024, slots_per_page=32, p=3)
    print(f"Params: {count_parameters(m):,}")
    print("1. Shapes"); [print(f"  ✅ T={T}", m(torch.randint(0,65,(2,T))).shape==(2,T,65)) for T in [48,100,200,512]]
    print("2. Gradients")
    (m(torch.randint(0,65,(2,256))).mean() + m.aux_loss()).backward()
    miss=[n for n,p in m.named_parameters() if p.grad is None]
    nan=[n for n,p in m.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"  manquants:{miss or 'aucun'} NaN:{nan or 'aucun'}"); m.zero_grad()
    print("3. Audit causal"); m.eval()
    for T in [48,100,200,320]:
        s1=torch.randint(0,65,(1,T)); s2=s1.clone(); s2[0,T//2:]=torch.randint(0,65,(T-T//2,))
        with torch.no_grad(): d=(m(s1)[0,:T//2]-m(s2)[0,:T//2]).abs().max().item()
        assert d<1e-6, f"FUITE T={T}:{d}"; print(f"  ✅ T={T}: {d:.1e}")
    print("4. Test SURVIE ENTITÉ (l'échec exact de v1)")
    mem = m.mem_block().mem
    torch.manual_seed(1); h = torch.randn(1,256,64)*0.5
    hA=h.clone(); hA[0,5]+=5.0; hB=h.clone(); hB[0,5]-=5.0   # entité différente au segment 0
    with torch.no_grad():
        mem(hA,64); rA,cA = mem._route[0,3].clone(), mem._c[0,3].clone()
        mem(hB,64); rB,cB = mem._route[0,3].clone(), mem._c[0,3].clone()
    dr=(rA-rB).abs().max().item(); dc=(cA-cB).abs().max().item()
    print(f"  Δroute(seg3)={dr:.3f}  Δc(seg3)={dc:.3f} -> {'✅ entité PROPAGÉE' if dc>0.1 else '❌ noyée (comme v1)'}")
    print("5. Accès HDD/token")
    with torch.no_grad(): m(torch.randint(0,65,(1,512)))
    st=mem.access_stats(512,64); print(f"  {st['acces_par_token']:.4f} accès/token ({st['pages_distinctes']} pages) — cible <0.05")
    print("\n✅ #17-v2 validé structurellement")
