"""
CDCN — Causal Dilated Convolution Network (WaveNet-like)
- Convolutions causales dilatées, champ réceptif exponentiel 2^L-1
- Gating : sigmoid(z) * tanh(h)
- O(B·T·d) temps, O(B·T·d) mémoire
- Causal par construction (pas de fuite possible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    """Convolution 1D causale avec padding = dilation."""
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # padding = dilation * (kernel_size - 1) pour causalité
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size, 
                              padding=self.pad, dilation=dilation, bias=False)
        # dim*2: gate || output (concaténés)
    
    def forward(self, x):
        # x: (B, T, C) → conv1d attend (B, C, T)
        x = x.transpose(1, 2)  # (B, C, T)
        out = self.conv(x)     # (B, 2*C, T + 2*pad)
        # Enlever le padding FUTUR (garder seulement les positions causales)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # enlève les positions futures
        # Couper pour correspondre à T
        out = out[:, :, :x.size(2)]
        out = out.transpose(1, 2)  # (B, T, 2*C)
        gate, h = out.chunk(2, dim=-1)
        return torch.sigmoid(gate) * torch.tanh(h)


class CDCNBlock(nn.Module):
    """Un bloc de convolution causale dilatée avec résiduelle."""
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.conv = CausalConv1d(dim, kernel_size, dilation)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.proj(x)
        return residual + x  # résiduelle


class CDCN(nn.Module):
    """
    Causal Dilated Convolution Network.
    
    L couches avec dilations exponentielles 2^0, 2^1, ..., 2^{L-1}.
    Champ réceptif total = sum_{l=0}^{L-1} (kernel-1) * 2^l + 1 = 2^L (pour kernel=3).
    
    Pour L=8, champ = 255, ce qui couvre la copy task (gap=188).
    """
    def __init__(
        self,
        vocab_size: int = 10,
        dim: int = 64,
        layers: int = 8,
        ffn_dim: int = 256,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Champ réceptif = 2^layers (avec kernel=3)
        self.receptive_field = 2 ** layers
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Couches dilatées : dilations 1, 2, 4, 8, ...
        self.blocks = nn.ModuleList([
            CDCNBlock(dim, dilation=2**l) for l in range(layers)
        ])
        
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
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test CDCN ===")
    model = CDCN(vocab_size=10, dim=32, layers=4, ffn_dim=128, max_seq_len=48)
    print(f"Paramètres: {count_parameters(model):,}")
    print(f"Champ réceptif: {model.receptive_field}")

    # 2a. Vérification des dimensions
    x = torch.randint(0, 10, (2, 48))
    out = model(x)
    assert out.shape == (2, 48, 10), f"Shape incorrecte: {out.shape}"
    print(f"✅ Shape OK: {out.shape}")

    # 2b. Test de gradient
    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient nul {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN {name}"
    print("✅ Gradients OK")

    # 2c. AUDIT CAUSAL OBLIGATOIRE
    seq1 = torch.randint(0, 10, (1, 48))
    seq2 = seq1.clone()
    seq2[0, 24:] = torch.randint(0, 10, (24,))
    
    model.eval()
    with torch.no_grad():
        o1, o2 = model(seq1), model(seq2)
    
    diff = (o1[0, :24] - o2[0, :24]).abs().max().item()
    assert diff < 1e-6, f"FUITE CAUSALE DÉTECTÉE: diff={diff:.6f}"
    print(f"✅ Causalité vérifiée — zéro fuite (diff={diff:.2e})")

    print("\n✅ CDCN validé — prêt pour benchmark")