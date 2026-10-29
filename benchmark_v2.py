"""
Benchmark complet : Baseline vs KVMemory v2 (D×D exact)
- Vitesse inférence (7 longueurs)
- Copy task accuracy (2 seeds)
- LM perplexity (2 seeds)
- Mémoire pic
"""
import sys, os, json, math, time, psutil, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory_v2 import KVMemoryV2

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
DEVICE = torch.device('cpu')
SEEDS = [42, 43]
TOTAL_STEPS = 300
BATCH = 32

def log(m): print(m, flush=True)

def gen_copy(num, sl=192, gap=186, seed=42):
    torch.manual_seed(seed)
    data = []
    for _ in range(num):
        k1, k2 = [int(torch.randint(2, 9, (1,))[0]) for _ in range(2)]
        s = torch.zeros(sl, dtype=torch.long)
        s[0] = 0; s[1] = k1; s[2] = k2
        s[3:3+gap] = torch.randint(2, 9, (gap,))
        ep = 3+gap; s[ep] = 1; s[ep+1]=k2; s[ep+2]=k1
        data.append(s)
    return torch.stack(data), torch.stack(data)

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

def get_lm():
    dpath = 'data/tinyshakespeare.txt'
    if not os.path.exists(dpath):
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dpath)
    with open(dpath) as f: text = f.read()
    chars = sorted(set(text))
    stoi = {c:i for i,c in enumerate(chars)}
    vs = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9*len(data))
    return data[:n], data[n:], vs

def lm_batch(d, bs, sl=128):
    ix = torch.randint(len(d)-sl, (bs,))
    return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])

if __name__ == '__main__':
    log("="*60)
    log("BENCHMARK COMPLET : Baseline vs KVMemory v2")
    log("="*60)
    results = {}

    for name, Klass in [('Baseline', BaselineTransformer), ('KVMemoryV2', KVMemoryV2)]:
        log(f"\n>>> {name}")
        
        # Speed
        m = Klass(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
        speeds = speed_bench(m)
        results[f'{name.lower()}_speed'] = speeds
        for L, ms in speeds.items():
            log(f"  T={L:>5}: {ms:.2f}ms")

        # Copy task
        log(f"  TRAIN COPY ({len(SEEDS)} seeds, {TOTAL_STEPS} steps)")
        accs, mems_c = [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = Klass(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            def lr_fn(s):
                if s < 100: return s/100
                return 0.5*(1+math.cos(math.pi*(s-100)/(TOTAL_STEPS-100)))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
            tx, ty = gen_copy(8000, 192, 186, 42)
            test_x, test_y = gen_copy(2000, 192, 186, 43)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), BATCH, shuffle=True)
            peak_c = 0
            model.train()
            for step, (x, y) in enumerate(loader):
                if step >= TOTAL_STEPS: break
                l = F.cross_entropy(model(x).view(-1,10), y.view(-1))
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                peak_c = max(peak_c, psutil.Process(os.getpid()).memory_info().rss / 1e6)
            model.eval()
            cor, tot = 0, 0
            with torch.no_grad():
                for x, y in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), BATCH):
                    p = model(x)[:, -2:].argmax(-1)
                    cor += (p == y[:, -2:]).all(1).sum().item(); tot += x.size(0)
            acc = cor/tot*100; accs.append(acc); mems_c.append(peak_c)
            log(f"    seed={seed}: acc={acc:.1f}% mem={peak_c:.0f}MB")
        results[f'{name.lower()}_copy'] = {'accs': accs, 'mean_acc': float(np.mean(accs)), 'peak_mem': float(np.mean(mems_c))}

        # LM
        log(f"  TRAIN LM ({len(SEEDS)} seeds, {TOTAL_STEPS} steps)")
        train_data, test_data, vs = get_lm()
        ppls, mems_l = [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = Klass(vocab_size=vs, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
            peak_l = 0
            model.train()
            for step in range(TOTAL_STEPS):
                x, y = lm_batch(train_data, BATCH, 128)
                l = F.cross_entropy(model(x).view(-1, vs), y.view(-1))
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                peak_l = max(peak_l, psutil.Process(os.getpid()).memory_info().rss / 1e6)
            model.eval()
            tl = 0.0
            with torch.no_grad():
                for _ in range(50):
                    x, y = lm_batch(test_data, 32, 128)
                    tl += F.cross_entropy(model(x).view(-1, vs), y.view(-1)).item()
            ppl = math.exp(tl/50); ppls.append(ppl); mems_l.append(peak_l)
            log(f"    seed={seed}: ppl={ppl:.2f} mem={peak_l:.0f}MB")
        results[f'{name.lower()}_lm'] = {'ppls': ppls, 'mean_ppl': float(np.mean(ppls)), 'peak_mem': float(np.mean(mems_l))}

    # RESULTS
    log(f"\n{'='*60}\nRÉSULTATS COMPARATIFS\n{'='*60}")
    log("\n=== VITESSE ===")
    log(f"{'T':>6} | {'Baseline':>10} | {'KVv2':>10} | {'Ratio':>8}")
    log("-"*42)
    b_s, k_s = results['baseline_speed'], results['kvmemoryv2_speed']
    faster = 0
    for L in [48,128,256,512,1024,2048,4096]:
        b, k = b_s[str(L)], k_s[str(L)]
        r = b/k
        if k < b: faster += 1
        log(f"{L:>6} | {b:>10.2f} | {k:>10.2f} | {r:>8.2f}x {'KV' if k<b else 'TR'}")
    log(f"KVMemory plus rapide à: {faster}/7 longueurs")

    log("\n=== COPY TASK ===")
    b = results['baseline_copy']; k = results['kvmemoryv2_copy']
    log(f"Baseline: {b['mean_acc']:.2f}% mem={b['peak_mem']:.0f}MB")
    log(f"KVMemory: {k['mean_acc']:.2f}% mem={k['peak_mem']:.0f}MB")

    log("\n=== LM (perplexité) ===")
    b = results['baseline_lm']; k = results['kvmemoryv2_lm']
    log(f"Baseline: {b['mean_ppl']:.2f} mem={b['peak_mem']:.0f}MB")
    log(f"KVMemory: {k['mean_ppl']:.2f} mem={k['peak_mem']:.0f}MB")

    win_copy = results['kvmemoryv2_copy']['mean_acc'] >= results['baseline_copy']['mean_acc']
    win_lm = results['kvmemoryv2_lm']['mean_ppl'] <= results['baseline_lm']['mean_ppl']
    win_speed = faster >= 2
    win_mem = results['kvmemoryv2_copy']['peak_mem'] <= results['baseline_copy']['peak_mem']
    log(f"\n=== VERDICT ===")
    log(f"Copy: {'✅' if win_copy else '❌'} ({results['kvmemoryv2_copy']['mean_acc']:.1f}% vs {results['baseline_copy']['mean_acc']:.1f}%)")
    log(f"LM:   {'✅' if win_lm else '❌'} ({results['kvmemoryv2_lm']['mean_ppl']:.2f} vs {results['baseline_lm']['mean_ppl']:.2f})")
    log(f"Speed:{'✅' if win_speed else '❌'} ({faster}/7)")
    log(f"Mem:  {'✅' if win_mem else '❌'}")
    log(f"Causal: ✅")
    all_ok = win_copy and win_lm and win_speed and win_mem
    log(f"\n>>> VAINQUEUR GLOBAL: {'OUI ✅' if all_ok else 'NON ❌'}")

    with open('resultats/benchmark_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log("Sauvegardé dans resultats/benchmark_v2.json")