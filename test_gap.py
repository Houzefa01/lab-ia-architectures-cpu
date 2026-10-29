"""Copy task avec gap variable — debug : est-ce que le modèle apprend ?"""
import torch, torch.nn.functional as F, math, sys
sys.path.append('.')
from baseline.transformer import BaselineTransformer

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
BATCH = 32

def gen_copy(num, gap, seed=42):
    torch.manual_seed(seed)
    seq_len = 3 + gap + 3  # SOS k1 k2 [gap] END k2 k1
    inputs, targets = [], []
    for _ in range(num):
        k1 = int(torch.randint(2, 9, (1,))[0])
        k2 = int(torch.randint(2, 9, (1,))[0])
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0] = 0
        seq[1] = k1
        seq[2] = k2
        seq[3:3+gap] = torch.randint(2, 9, (gap,))
        end = 3 + gap
        seq[end] = 1
        seq[end+1] = k2
        seq[end+2] = k1
        tgt = torch.zeros(seq_len, dtype=torch.long)
        tgt[:-1] = seq[1:]
        tgt[-1] = 0
        inputs.append(seq); targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets), seq_len

def train_eval(model, train_x, train_y, test_x, test_y, end_pos, steps=2000):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    def lr_fn(s):
        if s < 200: return s/200
        return 0.5*(1+math.cos(math.pi*(s-200)/(steps-200)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), BATCH, shuffle=True)
    
    for step, (x, y) in enumerate(loader):
        if step >= steps: break
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)
    
    model.eval()
    with torch.no_grad():
        p = model(test_x)[:, end_pos:end_pos+2, :].argmax(-1)
        t = test_y[:, end_pos:end_pos+2]
        cor = (p == t).all(dim=1).sum().item()
        tot = test_x.size(0)
    print(f"    Accuracy: {cor/tot*100:.1f}%", flush=True)
    return cor/tot*100

for gap in [10, 50, 186]:
    print(f"\n=== Gap={gap} ===", flush=True)
    train_x, train_y, sl = gen_copy(8000, gap, 42)
    test_x, test_y, _ = gen_copy(2000, gap, 43)
    end_pos = 3 + gap
    print(f"  Seq len: {sl}, end_pos: {end_pos}", flush=True)
    torch.manual_seed(42)
    model = BaselineTransformer(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=sl+1)
    acc = train_eval(model, train_x, train_y, test_x, test_y, end_pos, steps=2000)