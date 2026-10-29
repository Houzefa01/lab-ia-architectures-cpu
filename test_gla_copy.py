"""GLA chunked : copy task next-token + LM perplexity"""
import sys, os, math
sys.path.insert(0, os.path.abspath('.'))
from baseline.transformer import BaselineTransformer
from candidats_actifs.gla_chunked import GLAChunked
import torch, torch.nn.functional as F
import numpy as np

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
STEPS = 3000; BATCH = 32

def log(m): print(m, flush=True)

def gen_copy(num, seed=42, seq_len=192, gap=186):
    torch.manual_seed(seed)
    inputs, targets = [], []
    for _ in range(num):
        k1 = int(torch.randint(2, 9, (1,))[0])
        k2 = int(torch.randint(2, 9, (1,))[0])
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0] = 0; seq[1] = k1; seq[2] = k2
        seq[3:3+gap] = torch.randint(2, 9, (gap,))
        end = 3 + gap
        seq[end] = 1; seq[end+1] = k2; seq[end+2] = k1
        tgt = torch.zeros(seq_len, dtype=torch.long)
        tgt[:-1] = seq[1:]; tgt[-1] = 0
        inputs.append(seq); targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets)

train_x, train_y = gen_copy(8000, 42)
test_x, test_y = gen_copy(2000, 43)
end_pos = 3 + 186

for name, Klass, kw in [
    ('Baseline', BaselineTransformer, dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)),
    ('GLA_chunked', GLAChunked, dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096, chunk_size=64)),
]:
    log(f"\n=== {name} ===")
    torch.manual_seed(42)
    model = Klass(**kw)
    log(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/100 if s<100 else 0.5*(1+math.cos(math.pi*(s-100)/(STEPS-100))))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), BATCH, shuffle=True)

    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= STEPS: break
        loss = F.cross_entropy(model(x).view(-1, 10), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        preds = model(test_x)
        p = preds[:, end_pos:end_pos+2, :].argmax(-1)
        t = test_y[:, end_pos:end_pos+2]
        cor = (p == t).all(dim=1).sum().item()
        tot = test_x.size(0)
    acc = cor / tot * 100
    log(f"Copy accuracy: {acc:.1f}% ({cor}/{tot})")

    # Quick LM if copy passes
    if acc >= 95:
        log(f"→ LM test")
        dpath = 'data/tinyshakespeare.txt'
        if not os.path.exists(dpath):
            os.makedirs('data', exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dpath)
        with open(dpath) as f: txt = f.read()
        chars = sorted(set(txt)); vs = len(chars)
        stoi = {c:i for i,c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
        n = int(0.9*len(data))
        train, test = data[:n], data[n:]
        def batch(d, bs, sl=128):
            ix = torch.randint(len(d)-sl, (bs,))
            return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])
        model_lm = Klass(vocab_size=vs, **kw)
        opt_lm = torch.optim.AdamW(model_lm.parameters(), lr=3e-4, weight_decay=0.01)
        sched_lm = torch.optim.lr_scheduler.LambdaLR(opt_lm, lambda s: s/100 if s<100 else 0.5*(1+math.cos(math.pi*(s-100)/(STEPS-100))))
        model_lm.train()
        for step in range(STEPS):
            x, y = batch(train, BATCH, 128)
            loss = F.cross_entropy(model_lm(x).view(-1, vs), y.view(-1))
            opt_lm.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_lm.parameters(), 1.0)
            opt_lm.step(); sched_lm.step()
        model_lm.eval()
        tl = 0.0
        with torch.no_grad():
            for _ in range(100):
                x, y = batch(test, 32, 128)
                tl += F.cross_entropy(model_lm(x).view(-1, vs), y.view(-1)).item()
        ppl = math.exp(tl/100)
        log(f"Perplexity: {ppl:.2f}")
    else:
        log(f"→ GLA chunked échoue copy ({acc:.1f}% < 95%) → Sparse clustering")
