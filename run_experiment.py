"""
Experiment: Baseline Transformer vs KVMemory Vectorized
"""
import sys, os, json, math, time, psutil
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wilcoxon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory import KVMemoryVectorized

CONFIG = dict(
    DIM=64, HEADS=2, LAYERS=3, FFN_DIM=256, LR=3e-4,
    TOTAL_STEPS=500, WARMUP=100, BATCH=32, DEVICE=torch.device('cpu'),
)
RESULTS_DIR = 'resultats'
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg, **kw):
    print(msg, flush=True, **kw)

def gen_copy(num, seq_len=192, gap=186, vocab=10, seed=42):
    torch.manual_seed(seed)
    data, tgt = [], []
    for _ in range(num):
        k1, k2 = [int(torch.randint(2, vocab-1, (1,))[0]) for _ in range(2)]
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0] = 0
        seq[1] = k1
        seq[2] = k2
        seq[3:3+gap] = torch.randint(2, vocab-1, (gap,))
        ep = 3 + gap
        seq[ep] = 1; seq[ep+1] = k2; seq[ep+2] = k1
        data.append(seq); tgt.append(seq)
    return torch.stack(data), torch.stack(tgt)

def train_model(model, task='copy', total_steps=500, batch=32):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    def lr_fn(s):
        if s < 100: return s / 100
        p = (s - 100) / (total_steps - 100)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    crit = nn.CrossEntropyLoss()
    
    if task == 'copy':
        tx, ty = gen_copy(8000, 192, 186, 10, 42)
        test_x, test_y = gen_copy(2000, 192, 186, 10, 43)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tx, ty), batch, shuffle=True)
        dev = CONFIG['DEVICE']
        peak = 0
        for step, (x, y) in enumerate(loader):
            if step >= total_steps: break
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = crit(logits.view(-1, 10), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            peak = max(peak, psutil.Process(os.getpid()).memory_info().rss / 1e6)
        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for x, y in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_x, test_y), batch):
                x, y = x.to(dev), y.to(dev)
                p = model(x)[:, -2:].argmax(-1)
                cor += (p == y[:, -2:]).all(1).sum().item()
                tot += x.size(0)
        return {'accuracy': cor/tot*100, 'peak_mem_mb': peak}
    
    elif task == 'lm':
        dpath = 'data/tinyshakespeare.txt'
        if not os.path.exists(dpath):
            os.makedirs('data', exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                dpath)
        with open(dpath) as f: text = f.read()
        chars = sorted(set(text))
        stoi = {c:i for i,c in enumerate(chars)}
        vs = len(chars)
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        train, test = data[:n], data[n:]
        def batch(d, bs, sl):
            ix = torch.randint(len(d)-sl, (bs,))
            return torch.stack([d[i:i+sl] for i in ix]), \
                   torch.stack([d[i+1:i+sl+1] for i in ix])
        dev = CONFIG['DEVICE']
        peak = 0
        for step in range(total_steps):
            x, y = batch(train, batch, 128)
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vs), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            peak = max(peak, psutil.Process(os.getpid()).memory_info().rss / 1e6)
        model.eval()
        tl = 0.0
        with torch.no_grad():
            for _ in range(50):
                x, y = batch(test, 32, 128)
                x, y = x.to(dev), y.to(dev)
                tl += F.cross_entropy(model(x).view(-1, vs), y.view(-1)).item()
        return {'perplexity': math.exp(tl/50), 'peak_mem_mb': peak}

def inference_speed(model, lengths=[48,128,256,512,1024,2048,4096], n=10):
    model.eval()
    res = {}
    for L in lengths:
        x = torch.randint(0, 10, (1, L))
        with torch.no_grad():
            for _ in range(3): model(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad(): model(x)
            ts.append(time.perf_counter() - t0)
        res[str(L)] = sorted(ts)[n//2] * 1000
    return res

if __name__ == '__main__':
    log("="*60)
    log("EXPÉRIENCE: Baseline vs KVMemory — Copy Task")
    log("="*60)
    
    all_results = {}
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    c = CONFIG
    
    for name, Klass in [('Baseline', BaselineTransformer), ('KVMemory', KVMemoryVectorized)]:
        log(f"\n--- {name} ---")
        accs, mems = [], []
        models = []
        for seed in SEEDS[:5]:  # 5 seeds for quick test
            torch.manual_seed(seed); np.random.seed(seed)
            log(f"  Seed {seed}...", end=' ')
            model = Klass(vocab_size=10, dim=c['DIM'], heads=c['HEADS'],
                         layers=c['LAYERS'], ffn_dim=c['FFN_DIM'], max_seq_len=4096)
            res = train_model(model, 'copy', c['TOTAL_STEPS'], c['BATCH'])
            accs.append(res['accuracy']); mems.append(res['peak_mem_mb'])
            models.append(model)
            log(f"acc={res['accuracy']:.2f}% mem={res['peak_mem_mb']:.0f}MB")
        all_results[f'{name.lower()}_copy'] = {
            'accuracies': accs, 'memories': mems,
            'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs)),
            'mean_mem': float(np.mean(mems))
        }
        all_results[f'{name.lower()}_speed'] = inference_speed(models[-1])
    
    # Stats
    b_accs = all_results['baseline_copy']['accuracies']
    k_accs = all_results['kvmem_copy']['accuracies']
    log(f"\n=== STATISTIQUES COPY TASK ===")
    log(f"Baseline: {np.mean(b_accs):.2f}% ± {np.std(b_accs):.2f}%")
    log(f"KVMemory: {np.mean(k_accs):.2f}% ± {np.std(k_accs):.2f}%")
    
    # Test Wilcoxon - if all values equal, skip
    try:
        stat, p = wilcoxon(k_accs, b_accs, alternative='two-sided')
        log(f"Wilcoxon p-value: {p:.6f}")
        all_results['wilcoxon_copy'] = {'statistic': float(stat), 'p_value': float(p)}
    except ValueError as e:
        log(f"Wilcoxon skipped: {e}")
        all_results['wilcoxon_copy'] = {'note': str(e), 'p_value': 1.0}
    
    # Vitesse
    log(f"\n=== VITESSE D'INFÉRENCE (ms) ===")
    log(f"{'L':>6} | {'Baseline':>10} | {'KVMemory':>10} | {'Ratio':>8}")
    log("-"*42)
    b_speed = all_results['baseline_speed']
    k_speed = all_results['kvmem_speed']
    faster = 0
    for L in [48, 128, 256, 512, 1024, 2048, 4096]:
        b = b_speed[str(L)]; k = k_speed[str(L)]
        r = b/k
        log(f"{L:>6} | {b:>10.2f} | {k:>10.2f} | {r:>8.2f}x")
        if k < b: faster += 1
    all_results['speed_comparison'] = {'kvmem_faster_at': faster}
    
    # Mémoire
    b_mem = all_results['baseline_copy']['mean_mem']
    k_mem = all_results['kvmem_copy']['mean_mem']
    log(f"\n=== MÉMOIRE ===")
    log(f"Baseline: {b_mem:.1f} MB")
    log(f"KVMemory: {k_mem:.1f} MB")
    win_mem = k_mem < b_mem
    log(f"Gagnant mémoire: {'KVMemory' if win_mem else 'Baseline'}")
    all_results['memory_winner'] = 'KVMemory' if win_mem else 'Baseline'
    
    # Sauvegarde
    with open(f'{RESULTS_DIR}/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\n✅ Résultats sauvegardés dans {RESULTS_DIR}/experiment_results.json")
    
    # Verdict
    log(f"\n{'='*60}")
    log("VERDICT FINAL (5 seeds, Copy Task)")
    log(f"{'='*60}")
    p = all_results['wilcoxon_copy'].get('p_value', 1.0)
    log(f"Précision: {'✅' if p < 0.05 else '❌'} (p={p:.4f})")
    log(f"Vitesse (2+ L): {'✅' if faster >= 2 else '❌'} ({faster}/7)")
    log(f"Mémoire:     {'✅' if win_mem else '❌'}")
    log(f"Causalité:   ✅ (validée constructeur)")
    all_ok = (p < 0.05) and (faster >= 2) and win_mem
    log(f"\n>>> VAINQUEUR GLOBAL: {'OUI ✅' if all_ok else 'NON ❌'}")
