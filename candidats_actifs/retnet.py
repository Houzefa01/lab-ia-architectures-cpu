"""
Retention causale (RetNet-style)
- o[t] = q[t] @ S[t]  où S[t] = gamma * S[t-1] + k[t]^T @ v[t]
- Produit scalaire QK exact (pas d'approximation phi)
- Métrique de position: gamma^{t-tau}
- O(T·D²) temps récurrent, O(D²) mémoire
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class RetentionBlock(nn.Module):
    def __init__(self, dim, heads=2, gamma_init=0.999):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4, bias=False), nn.GELU(),
            nn.Linear(dim*4, dim, bias=False))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # gamma appris par tête
        self.log_gamma = nn.Parameter(torch.full((heads,), math.log(gamma_init)))
    
    def forward(self, x):
        B, T, C = x.shape
        residual = x; x = self.norm1(x)
        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        
        # Récurrence par tête: S_t = gamma * S_{t-1} + k_t ⊗ v_t
        gamma = self.log_gamma.exp().view(1, self.heads, 1, 1)
        S = torch.zeros(B, self.heads, self.hd, self.hd, device=x.device)
        outputs = []
        for t in range(T):
            k_t = k[:, :, t, :].unsqueeze(-1)  # (B,H,D,1)
            v_t = v[:, :, t, :].unsqueeze(-2)  # (B,H,1,D)
            S = gamma * S + k_t @ v_t
            q_t = q[:, :, t, :].unsqueeze(-2)  # (B,H,1,D)
            o_t = (q_t @ S).squeeze(-2)  # (B,H,D)
            outputs.append(o_t)
        
        attn = torch.stack(outputs, dim=2).transpose(1, 2).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = residual + attn
        x = x + self.ffn(self.norm2(x))
        return x


class RetNet(nn.Module):
    def __init__(self, vocab_size=10, dim=64, heads=2, layers=3, ffn_dim=256, max_seq_len=4096):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim; self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([RetentionBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B,T = x.shape; assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for b in self.blocks: x = b(x)
        x = self.norm(x); return self.head(x)

if __name__ == "__main__":
    print("=== Test RetNet ===")
    m = RetNet(vocab_size=10, dim=32, heads=2, layers=2, max_seq_len=48)
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
    x = torch.randint(0,10,(2,48)); out = m(x)
    assert out.shape==(2,48,10); print("✅ Shape OK")
    out.mean().backward()
    for n,p in m.named_parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    print("✅ Gradients OK")
    s1=torch.randint(0,10,(1,48)); s2=s1.clone(); s2[0,24:]=torch.randint(0,10,(24,))
    m.eval()
    with torch.no_grad(): o1,o2=m(s1),m(s2)
    diff=(o1[0,:24]-o2[0,:24]).abs().max().item()
    assert diff<1e-6; print(f"✅ Causalité OK (diff={diff:.2e})")
    print("✅ RetNet validé")