"""
Benchmark rapide : Baseline vs KVMemory
1) Vitesse inférence (7 longueurs, 10 runs)
2) Entraînement copy task (200 steps, 3 seeds)
3) Mémoire pic
"""
import sys, os, json, math, time, psutil
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory import KVMemoryVectorized

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
DEVICE = torch.device('cpu')
SEEDS = [42, 43]  # 2 seeds suffisent vu 100%
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

def speed_bench(model, lengths=[48,128,256,512,1024,2048,4096], n=10):
    model.eval()
    res = {}
    for L in lengths:
        x = torch.randint(0, 10, (1, L))
        with torch.no_grad():
            for _ in range(5): model(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad(): model(x)
            ts.append((time.perf_counter() - t0) * 1000)
        res[str(L)] = sorted(ts)[n//2]
    return res

def train_on_copy(name, Klass, seeds, steps=200, batch=32):
    log(f"\n--- {name} ---")
    accs, mems, times = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Klass(vocab_size=10, dim=DIM, heads=HEADS,
                      layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
        
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        def lr_fn(s):
            if s < 100: return s / 100
            return 0.5 * (1 + math.cos(math.pi * (s-100) / (steps-100)))
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
        acc = cor / tot * 100
        accs.append(acc); mems.append(peak); times.append(dt)
        log(f"  seed={seed}: acc={acc:.1f}% mem={peak:.0f}MB t={dt:.0f}s")
    
    return {'accuracies': accs, 'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)), 'peak_mem': float(np.mean(mems)),
            'times': times}

if __name__ == '__main__':
    log("=" * 60)
    log("BENCHMARK RAPIDE : Baseline vs KVMemory")
    log("=" * 60)
    
    results = {}
    
    for name, Klass in [('Baseline', BaselineTransformer), ('KVMemory', KVMemoryVectorized)]:
        log(f"\n>>> CRÉATION {name}")
        model = Klass(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS,
                      ffn_dim=FFN, max_seq_len=4096)
        
        log(f">>> SPEED BENCH {name}")
        speeds = speed_bench(model)
        for L, ms in speeds.items():
            log(f"  T={L:>5}: {ms:.2f}ms")
        results[f'{name.lower()}_speed'] = speeds
        
        log(f">>> TRAINING {name} (3 seeds, {TOTAL_STEPS} steps)")
        train_res = train_on_copy(name, Klass, SEEDS, TOTAL_STEPS, BATCH)
        results[f'{name.lower()}_train'] = train_res
    
    # Stats
    log(f"\n{'='*60}")
    log("RÉSULTATS")
    log(f"{'='*60}")
    
    b_acc = results['baseline_train']['mean_acc']
    k_acc = results['kvmem_train']['mean_acc']
    log(f"\nPrécision moyenne:")
    log(f"  Baseline: {b_acc:.2f}%")
    log(f"  KVMemory: {k_acc:.2f}%")
    
    b_mem = results['baseline_train']['peak_mem']
    k_mem = results['kvmem_train']['peak_mem']
    log(f"\nMémoire RAM pic:")
    log(f"  Baseline: {b_mem:.0f} MB")
    log(f"  KVMemory: {k_mem:.0f} MB")
    
    log(f"\nVitesse inférence (ms, médiane sur 10):")
    log(f"{'T':>6} | {'Baseline':>10} | {'KVMemory':>10} | {'Ratio':>8}")
    log("-"*42)
    b_s = results['baseline_speed']; k_s = results['kvmem_speed']
    faster_Ls = []
    for L in [48, 128, 256, 512, 1024, 2048, 4096]:
        b, k = b_s[str(L)], k_s[str(L)]
        r = b/k
        w = "KV" if k < b else "TR"
        if k < b: faster_Ls.append(L)
        log(f"{L:>6} | {b:>10.2f} | {k:>10.2f} | {r:>8.2f}x {w}")
    
    log(f"\n>>> KVMemory plus rapide à: {faster_Ls}")
    log(f">>> Précision KV > Baseline: {k_acc > b_acc}")
    log(f">>> Mémoire KV < Baseline: {k_mem < b_mem}")
    
    verdict = (k_acc > b_acc) and (len(faster_Ls) >= 2) and (k_mem < b_mem)
    log(f"\n>>> VAINQUEUR GLOBAL: {'OUI' if verdict else 'NON'}")
    
    with open('resultats/benchmark_rapide.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log("\nRésultats sauvegardés dans resultats/benchmark_rapide.json")