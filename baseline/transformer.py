"""
Transformer causal de référence — BASELINE INVARIABLE
dim=64, heads=2, layers=3, ffn=256
Masque causal strict, AdamW lr=3e-4, cosine decay, warmup=100 steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Masque causal strict
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
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


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, heads)
        self.ffn = FeedForward(dim, ffn_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        dim: int = 64,
        heads: int = 2,
        layers: int = 3,
        ffn_dim: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ffn_dim) for _ in range(layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Init weights
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
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rapide : dimensions, gradient, causalité
    print("=== Test Baseline Transformer ===")
    
    model = BaselineTransformer(vocab_size=10, dim=32, heads=2, layers=2, ffn_dim=128, max_seq_len=48)
    print(f"Paramètres: {count_parameters(model):,}")
    
    # 2a. Vérification des dimensions
    x = torch.randint(0, 10, (2, 48))
    out = model(x)
    assert out.shape == (2, 48, 10), f"Shape incorrecte: {out.shape}"
    print(f"✅ Shape OK: {out.shape}")
    
    # 2b. Test de gradient
    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient nul pour {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN pour {name}"
    print("✅ Gradients OK")
    
    # 2c. AUDIT CAUSAL OBLIGATOIRE
    seq1 = torch.randint(0, 10, (1, 48))
    seq2 = seq1.clone()
    seq2[0, 24:] = torch.randint(0, 10, (24,))
    
    model.eval()
    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)
    
    diff = (out1[0, :24] - out2[0, :24]).abs().max().item()
    assert diff < 1e-6, f"FUITE CAUSALE DÉTECTÉE: diff={diff:.6f}"
    print(f"✅ Causalité vérifiée — zéro fuite (diff={diff:.2e})")
    
    print("\n✅ Baseline Transformer validé — prêt pour comparaisons")