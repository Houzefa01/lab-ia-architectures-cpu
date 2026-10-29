"""
Mamba SSM Sélectif — version épurée pour CPU
h_t = exp(Δ_t·A)·h_{t-1} + Δ_t·B_t
y_t = C_t·h_t + D·x_t
Structure : SSM (N états → sortie scalaire) + projection linéaire + GLU
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class SelectiveSSM(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim; self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.zeros(dim))
        self.dt_proj = nn.Linear(dim, 1, bias=False)
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(dim, d_state, bias=False)
        self.expand = nn.Linear(1, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape; N = self.d_state
        dt = F.softplus(self.dt_proj(x))            # (B,T,1)
        B_in = self.B_proj(x)                       # (B,T,N)
        C_in = self.C_proj(x)                       # (B,T,N)
        A = -self.A_log.exp()                       # (N,)

        h = torch.zeros(B, N, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, 0:1]   # (B,1)
            b_t = B_in[:, t]       # (B,N)
            c_t = C_in[:, t]       # (B,N)

            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B,N)
            h = A_bar * h + dt_t * b_t                 # (B,N)
            y = (c_t * h).sum(-1, keepdim=True)        # (B,1)
            ys.append(y)

        ssm_out = torch.cat(ys, dim=-1)               # (B,T)
        ssm_out = self.expand(ssm_out.unsqueeze(-1))  # (B,T,D)
        return ssm_out + self.D.unsqueeze(0).unsqueeze(0) * x


class MambaSelectiveBlock(nn.Module):
    def __init__(self, dim, d_state=16, ffn_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        r = x; x = self.norm(x)
        s = self.ssm(x)
        g = F.silu(self.gate(x))
        u = self.up(x)
        x = g * u * s.mean(-1, keepdim=True).expand(-1, -1, u.size(-1))
        x = self.down(x)
        return r + x


class MambaSelective(nn.Module):
    def __init__(self, vocab_size=10, dim=64, layers=3, ffn_dim=256,
                 max_seq_len=4096, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            MambaSelectiveBlock(dim, d_state, ffn_dim) for _ in range(layers)])
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
    print("=== Test Mamba SSM Sélectif ===")
    m = MambaSelective(vocab_size=10, dim=32, layers=2, max_seq_len=48, d_state=8)
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
    print("✅ Mamba SSM Sélectif validé")
