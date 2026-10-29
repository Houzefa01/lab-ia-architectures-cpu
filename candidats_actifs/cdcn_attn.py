"""
CDCN-Attention — Hybride : 7 couches conv dilatées + 1 couche attention causale
- Convs : O(T·d), champ réceptif 127
- Attention : 1 couche finale pour le rappel associatif exact
- Complexité totale : O(T·d + T²·d) = O(T²·d) pire cas, mais T² est pour UNE SEULE couche
  Pour T=4096, d=64 : convs = 4096×64×7×3 = 5.5M, attention = 4096²×64/2 = 537M
  L'attention domine à grand T, mais c'est 1 couche au lieu de 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size,
                              padding=self.pad, dilation=dilation, bias=False)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        out = out[:, :, :x.size(2)]
        out = out.transpose(1, 2)
        gate, h = out.chunk(2, dim=-1)
        return torch.sigmoid(gate) * torch.tanh(h)


class ConvBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv = CausalConv1d(dim, 3, dilation)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        r = x; x = self.norm(x); x = self.conv(x)
        x = self.proj(x); return r + x


class CausalAttention(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim; self.heads = heads; self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class CDCN_Attn(nn.Module):
    def __init__(self, vocab_size=10, dim=64, conv_layers=7, ffn_dim=256,
                 attn_heads=2, max_seq_len=4096, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size; self.dim = dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Convs dilatées
        self.conv_blocks = nn.ModuleList([
            ConvBlock(dim, dilation=2**l) for l in range(conv_layers)
        ])
        
        # Attention finale
        self.attn_norm = nn.LayerNorm(dim)
        self.attention = CausalAttention(dim, attn_heads)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropout(x)
        for block in self.conv_blocks:
            x = block(x)
        x = x + self.attention(self.attn_norm(x))
        x = self.norm(x)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test CDCN-Attention ===")
    model = CDCN_Attn(vocab_size=10, dim=32, conv_layers=4, attn_heads=2, max_seq_len=48)
    print(f"Paramètres: {count_parameters(model):,}")
    
    x = torch.randint(0, 10, (2, 48)); out = model(x)
    assert out.shape == (2, 48, 10); print(f"✅ Shape OK: {out.shape}")
    
    loss = out.mean(); loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, f"Gradient nul {n}"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN {n}"
    print("✅ Gradients OK")
    
    seq1 = torch.randint(0, 10, (1, 48)); seq2 = seq1.clone()
    seq2[0, 24:] = torch.randint(0, 10, (24,))
    model.eval()
    with torch.no_grad():
        o1, o2 = model(seq1), model(seq2)
    diff = (o1[0, :24] - o2[0, :24]).abs().max().item()
    assert diff < 1e-6, f"FUITE CAUSALE: diff={diff:.6f}"
    print(f"✅ Causalité vérifiée (diff={diff:.2e})")
    print("\n✅ CDCN-Attention validé")