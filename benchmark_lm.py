"""
Mini LM benchmark : perplexité sur TinyShakespeare (200 steps)
"""
import sys, os, json, math, time, psutil, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory import KVMemoryVectorized

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
SEEDS = [42, 43, 44]
TOTAL_STEPS = 300
BATCH = 32
DEVICE = torch.device('cpu')

def log(m): print(m, flush=True)

def get_lm():
    dpath = 'data/tinyshakespeare.txt'
    if not os.path.exists(dpath):
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            dpath)
    with open(dpath) as f: text = f.read()
    chars = sorted(set(text))
    stoi = {c:i for i,c in enumerate(chars)}
    vs = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vs

def batch(d, bs, sl=128):
    ix = torch.randint(len(d)-sl, (bs,))
    return torch.stack([d[i:i+sl] for i in ix]), \
           torch.stack([d[i+1:i+sl+1] for i in ix])

def train_lm(name, Klass, vs, seeds, steps=300, bs=32):
    log(f"\n--- {name} (vocab={vs}) ---")
    train_data, test_data, _ = get_lm()
    ppls, mems, times = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Klass(vocab_size=vs, dim=DIM, heads=HEADS,
                      layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
        
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        def lr_fn(s):
            if s < 100: return s / 100
            return 0.5 * (1 + math.cos(math.pi * (s-100) / (steps-100)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        
        t0 = time.time()
        peak = 0
        model.train()
        for step in range(steps):
            x, y = batch(train_data, bs, 128)
            l = F.cross_entropy(model(x).view(-1, vs), y.view(-1))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            peak = max(peak, psutil.Process(os.getpid()).memory_info().rss / 1e6)
        dt = time.time() - t0
        
        model.eval()
        tl = 0.0
        with torch.no_grad():
            for _ in range(50):
                x, y = batch(test_data, bs, 128)
                tl += F.cross_entropy(model(x).view(-1, vs), y.view(-1)).item()
        ppl = math.exp(tl / 50)
        ppls.append(ppl); mems.append(peak); times.append(dt)
        log(f"  seed={seed}: ppl={ppl:.2f} mem={peak:.0f}MB t={dt:.0f}s")
    return {'ppls': ppls, 'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)), 'peak_mem': float(np.mean(mems))}

if __name__ == '__main__':
    log("="*60)
    log("BENCHMARK LM : Baseline vs KVMemory")
    log("="*60)
    
    # Get vocab size first
    train_data, test_data, vs = get_lm()
    log(f"Vocabulaire LM: {vs} caractères")
    
    results = {}
    for name, Klass in [('Baseline', BaselineTransformer), ('KVMemory', KVMemoryVectorized)]:
        res = train_lm(name, Klass, vs, SEEDS, TOTAL_STEPS, BATCH)
        results[f'{name.lower()}_lm'] = res
    
    log(f"\n{'='*60}")
    log("RÉSULTATS LM")
    log(f"{'='*60}")
    b = results['baseline_lm']
    k = results['kvmem_lm']
    log(f"Baseline: ppl={b['mean_ppl']:.2f} ± {b['std_ppl']:.2f}, mem={b['peak_mem']:.0f}MB")
    log(f"KVMemory: ppl={k['mean_ppl']:.2f} ± {k['std_ppl']:.2f}, mem={k['peak_mem']:.0f}MB")
    log(f"Gagnant LM: {'KVMemory' if k['mean_ppl'] < b['mean_ppl'] else 'Baseline'}")
    
    with open('resultats/benchmark_lm.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log("\nRésultats sauvegardés.")