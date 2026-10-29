"""
Benchmark GLA Chunked vs Baseline vs Original GLA
1) Vitesse inférence (7 longueurs, batch=32, 10 runs)
2) Copy task (200 steps, 2 seeds)
3) LM perplexity (200 steps, 2 seeds)
"""
import sys, os, json, math, time, psutil, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.gla import GLATransformer
from candidats_actifs.gla_chunked import GLAChunked

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
DEVICE = torch.device('cpu')
SEEDS = [42, 43]
TOTAL_STEPS = 200
BATCH = 32

def log(m): print(m, flush=True)

def gen_copy(num, sl=192, gap=186, seed=42):
    torch.manual_seed(seed)
    data, tgt = [], []
    for _ in range(num):
        k1, k2 = [int(torch.randint(2, 9, (1,))[0]) for _ in range(2)]
        s = torch.zeros(sl, dtype=torch.long)
        s[0] = 0; s[1] = k1; s[2] = k2
        s[3:3+gap] = torch.randint(2, 9, (gap,))
        ep = 3 + gap; s[ep] = 1; s[ep+1] = k2; s[ep+2] = k1
        data.append(s); tgt.append(s)
    return torch.stack(data), torch.stack(tgt)

def speed_bench(model, lengths=[48,128,256,512,1024,2048,4096], n=5, bs=32):
    model.eval()
    res = {}
    for L in lengths:
        x = torch.randint(0, 10, (bs, L))
        with torch.no_grad():
            for _ in range(3): model(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad(): model(x)
            ts.append((time.perf_counter() - t0) * 1000)
        res[str(L)] = sorted(ts)[n//2]
    return res

def train_on_copy(name, Klass, kw, seeds, steps=200, batch=32):
    log(f"\n--- {name} ---")
    accs, mems, times = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Klass(**kw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        def lr_fn(s):
            if s < 100: return s/100
            return 0.5*(1+math.cos(math.pi*(s-100)/(steps-100)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        tx, ty = gen_copy(8000, 192, 186, 42)
        test_x, test_y = gen_copy(2000, 192, 186, 43)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tx, ty), batch, shuffle=True)
        t0 = time.time()
        peak = 0
        model.train()
        for step, (x, y) in enumerate(loader):
            if step >= steps: break
            l = F.cross_entropy(model(x).view(-1, 10), y.view(-1))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            peak = max(peak, psutil.Process(os.getpid()).memory_info().rss / 1e6)
        dt = time.time() - t0
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for x, y in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_x, test_y), batch):
                p = model(x)[:, -2:].argmax(-1)
                cor += (p == y[:, -2:]).all(1).sum().item()
                tot += x.size(0)
        acc = cor/tot*100
        accs.append(acc); mems.append(peak); times.append(dt)
        log(f"  seed={seed}: acc={acc:.1f}% mem={peak:.0f}MB t={dt:.0f}s")
    return {'accuracies': accs, 'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)), 'peak_mem': float(np.mean(mems)),
            'times': times}

def train_lm(name, Klass, kw, seeds, steps=200, batch=32):
    dpath = 'data/tinyshakespeare.txt'
    if not os.path.exists(dpath):
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dpath)
    with open(dpath) as f: text = f.read()
    chars = sorted(set(text)); vs = len(chars)
    stoi = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); train, test = data[:n], data[n:]
    def batch(d, bs, sl=128):
        ix = torch.randint(len(d)-sl, (bs,))
        return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])
    log(f"\n--- {name} LM (vocab={vs}) ---")
    ppls, mems = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Klass(vocab_size=vs, **kw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/100 if s<100 else 0.5*(1+math.cos(math.pi*(s-100)/(steps-100))))
        peak = 0
        model.train()
        for s in range(steps):
            x, y = batch(train, batch, 128)
            l = F.cross_entropy(model(x).view(-1, vs), y.view(-1))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            peak = max(peak, psutil.Process(os.getpid()).memory_info().rss / 1e6)
        model.eval()
        tl = 0.0
        with torch.no_grad():
            for _ in range(50):
                x, y = batch(test, 32, 128)
                tl += F.cross_entropy(model(x).view(-1, vs), y.view(-1)).item()
        ppl = math.exp(tl/50)
        ppls.append(ppl); mems.append(peak)
        log(f"  seed={seed}: ppl={ppl:.2f} mem={peak:.0f}MB")
    return {'ppls': ppls, 'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)), 'peak_mem': float(np.mean(mems))}

if __name__ == '__main__':
    log("="*60)
    log("BENCHMARK GLA CHUNKED vs BASELINE vs GLA ORIGINAL")
    log("="*60)

    results = {}
    
    # Models to compare
    models = [
        ('Baseline', BaselineTransformer, dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)),
        ('GLA_orig', GLATransformer, dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)),
        ('GLA_chunked', GLAChunked, dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096, chunk_size=64)),
    ]

    # SPEED BENCH (batch=32 inference)
    log("\n>>> SPEED BENCH (batch=32, 10 runs)")
    for name, Klass, kw in models:
        log(f"\n--- {name} ---")
        m = Klass(**kw)
        speeds = speed_bench(m)
        results[f'{name.lower()}_speed'] = speeds
        for L, ms in speeds.items():
            log(f"  T={L:>5}: {ms:.2f}ms")

    # COPY TASK (2 seeds, 200 steps)
    log("\n>>> COPY TASK (2 seeds, 200 steps)")
    for name, Klass, kw in models:
        train_res = train_on_copy(name, Klass, kw, SEEDS, TOTAL_STEPS, BATCH)
        results[f'{name.lower()}_copy'] = train_res

    # RESULTS
    log(f"\n{'='*60}\nRÉSULTATS\n{'='*60}")

    # Speed table
    log(f"\n=== VITESSE (ms) ===")
    log(f"{'T':>6} | {'Baseline':>10} | {'GLA_orig':>10} | {'GLA_chunk':>10} | {'chunk/orig':>10} | {'chunk/base':>10}")
    log("-"*62)
    for L in [48,128,256,512,1024,2048,4096]:
        b = results['baseline_speed'][str(L)]
        go = results['gla_orig_speed'][str(L)]
        gc = results['gla_chunked_speed'][str(L)]
        r_go = go/gc if gc > 0 else float('inf')
        r_gb = b/gc if gc > 0 else float('inf')
        log(f"{L:>6} | {b:>10.2f} | {go:>10.2f} | {gc:>10.2f} | {r_go:>10.2f}x | {r_gb:>10.2f}x")

    # Copy accuracy
    log(f"\n=== COPY TASK ===")
    for name in ['baseline', 'gla_orig', 'gla_chunked']:
        r = results[f'{name}_copy']
        log(f"  {name}: {r['mean_acc']:.2f}% ± {r['std_acc']:.2f}% mem={r['peak_mem']:.0f}MB")

    # Save
    with open('resultats/benchmark_gla_chunked.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n✅ Résultats sauvegardés dans resultats/benchmark_gla_chunked.json")
