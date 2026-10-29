"""
RetNet Multi-Gamma — retention causale avec gamma différent par tête
Têtes gamma-court (0.9) : contexte local, précision LM
Têtes gamma-long (0.999) : contexte global, rappel associatif
Apprentissage : gamma = sigma(theta) avec theta libre
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class RetNetMultiGammaBlock(nn.Module):
    def __init__(self, dim, heads=2, chunk_size=64, gamma_list=None):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4, bias=False), nn.GELU(),
            nn.Linear(dim*4, dim, bias=False))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if gamma_list is None:
            if heads == 2:
                gamma_list = [0.9, 0.999]
            elif heads == 4:
                gamma_list = [0.9, 0.95, 0.99, 0.999]
            else:
                gamma_list = [0.9 + 0.099 * i / (heads - 1) for i in range(heads)]
        assert len(gamma_list) == heads
        self.log_gamma = nn.Parameter(torch.tensor([math.log(g) for g in gamma_list]))

    def forward(self, x):
        B, T, C = x.shape
        residual = x; x = self.norm1(x)
        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)

        gamma = self.log_gamma.exp().clamp(max=0.9999)
        CS = self.chunk_size
        outputs = []
        S = torch.zeros(B, self.heads, self.hd, self.hd, device=x.device)

        for start in range(0, T, CS):
            end = min(start + CS, T)
            L = end - start

            q_c = q[:, :, start:end, :]
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]

            gamma_h = gamma.view(1, -1, 1, 1)
            i = torch.arange(L, device=x.device).view(1, 1, L, 1)
            j = torch.arange(L, device=x.device).view(1, 1, 1, L)
            decay = (gamma_h ** (i - j)) * (i >= j).float()

            scores = torch.matmul(q_c, k_c.transpose(-2, -1))
            scores = scores * decay
            local_out = torch.matmul(scores, v_c)

            cross_out = torch.matmul(q_c, S)

            chunk_out = local_out / math.sqrt(self.hd) + cross_out
            outputs.append(chunk_out)

            contrib = torch.einsum(
                'bhld,bhle->bhde',
                k_c * (gamma_h ** (L - 1 - torch.arange(L, device=x.device).view(1, 1, L, 1))).squeeze(-1).unsqueeze(-1),
                v_c
            )
            S = (gamma ** L).view(1, -1, 1, 1) * S + contrib

        attn = torch.cat(outputs, dim=2).transpose(1, 2).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = residual + attn
        x = x + self.ffn(self.norm2(x))
        return x


class RetNetMultiGamma(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256,
                 max_seq_len=4096, chunk_size=64, gamma_list=None):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            RetNetMultiGammaBlock(dim, heads, chunk_size, gamma_list) for _ in range(layers)
        ])
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
    print("=== Test RetNet Multi-Gamma ===")
    m = RetNetMultiGamma(vocab_size=10, dim=32, heads=2, layers=2, max_seq_len=48, chunk_size=16)
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    
    for name, p in m.named_parameters():
        if 'gamma' in name:
            print(f"  {name}: init_gamma={p.exp().detach().tolist()}")

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
    assert diff < 1e-6, f"FUITE CAUSALE: diff={diff:.6f}"
    print(f"✅ Causalité OK (diff={diff:.2e})")
    print("✅ RetNet Multi-Gamma validé — prêt pour benchmark")
