"""
Comparaison ÉQUITABLE du gate LM (résolution du gate contesté STEP 7).
Baseline vs BlockLocal, TOUS DEUX à max_seq_len=128, hyperparams identiques,
seeds appariés {42,43,44}, et JEU D'ÉVAL FIXE (même batches pour tous les runs)
afin d'isoler l'effet architecture du bruit d'échantillonnage.

Réutilise la mécanique EXACTE de benchmark_block_local_lm.py :
  AdamW lr=3e-4 wd=0.01, warmup manuel 100 (lr affine), puis CosineAnnealingLR,
  clip grad 1.0, 3000 steps, batch=32, seq_len=128, ppl=exp(mean CE / 100 batches).
Sortie : resultats/lm_fair_comparison.json
"""
import sys, os, json, math, time
import torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from baseline.transformer import BaselineTransformer, count_parameters
from candidats_actifs.block_local import BlockLocalTransformer

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
BLOCK_SIZE = 64
SEQ = 128
MAX_SEQ_LEN = 128          # conditions IDENTIQUES pour les deux modèles
BATCH = 32
STEPS = 3000
WARMUP = 100
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_BATCHES = 100
EVAL_SEED = 999            # jeu d'éval fixe, identique pour tous les runs
SEEDS = [42, 43, 44]
OUT_JSON = os.path.join(ROOT, 'resultats', 'lm_fair_comparison.json')


def log(m):
    print(m, flush=True)


def get_lm():
    dpath = os.path.join(ROOT, 'data', 'tinyshakespeare.txt')
    with open(dpath) as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vs = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vs


def batch(d, bs, sl=SEQ):
    ix = torch.randint(len(d) - sl, (bs,))
    return (torch.stack([d[i:i + sl] for i in ix]),
            torch.stack([d[i + 1:i + sl + 1] for i in ix]))


def build(model_name, vs):
    if model_name == 'baseline':
        return BaselineTransformer(vocab_size=vs, dim=DIM, heads=HEADS, layers=LAYERS,
                                   ffn_dim=FFN, max_seq_len=MAX_SEQ_LEN)
    return BlockLocalTransformer(vocab_size=vs, dim=DIM, heads=HEADS, layers=LAYERS,
                                 ffn_dim=FFN, max_seq_len=MAX_SEQ_LEN, block_size=BLOCK_SIZE)


def run_one(model_name, seed, train_data, test_data, vs):
    torch.manual_seed(seed)
    model = build(model_name, vs)
    npar = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        x, y = batch(train_data, BATCH)
        if step < WARMUP:
            opt.param_groups[0]['lr'] = LR * step / WARMUP
        loss = F.cross_entropy(model(x).reshape(-1, vs), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step >= WARMUP:
            sched.step()
        if (step + 1) % 1000 == 0:
            log(f"  [{model_name} seed{seed}] step {step+1}/{STEPS} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s")
    # Éval sur jeu FIXE
    model.eval()
    torch.manual_seed(EVAL_SEED)
    tl = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = batch(test_data, BATCH)
            tl += F.cross_entropy(model(x).reshape(-1, vs), y.reshape(-1)).item()
    ppl = math.exp(tl / EVAL_BATCHES)
    log(f"==> {model_name} seed{seed}: ppl={ppl:.4f} params={npar:,} ({time.time()-t0:.0f}s)")
    return ppl, npar


def main():
    train_data, test_data, vs = get_lm()
    log(f"vocab={vs} train={len(train_data)} test={len(test_data)}  eval FIXE seed={EVAL_SEED}")
    res = {"config": f"max_seq_len={MAX_SEQ_LEN}, {STEPS} steps, hparams identiques, eval FIXE",
           "baseline_128": {}, "block_local_128": {}}
    for seed in SEEDS:
        for name, key in (('baseline', 'baseline_128'), ('block_local', 'block_local_128')):
            ppl, npar = run_one(name, seed, train_data, test_data, vs)
            res[key][str(seed)] = ppl
            res[key]['n_params'] = npar
    b = [res['baseline_128'][str(s)] for s in SEEDS]
    l = [res['block_local_128'][str(s)] for s in SEEDS]
    res['baseline_128']['mean'] = sum(b) / len(b)
    res['block_local_128']['mean'] = sum(l) / len(l)
    res['paired_deltas'] = {str(s): res['block_local_128'][str(s)] - res['baseline_128'][str(s)] for s in SEEDS}
    try:
        from scipy.stats import wilcoxon
        st, pv = wilcoxon(l, b)
        res['wilcoxon'] = {"statistic": float(st), "pvalue": float(pv)}
    except Exception as e:
        res['wilcoxon'] = {"error": str(e), "note": "n=3 trop petit ou scipy absent; se fier aux moyennes/deltas"}
    mean_delta = res['block_local_128']['mean'] - res['baseline_128']['mean']
    res['interpretation'] = (
        f"baseline@128 mean={res['baseline_128']['mean']:.4f}, block_local@128 mean={res['block_local_128']['mean']:.4f}, "
        f"delta_moyen={mean_delta:+.4f}. Si |delta| est dans le bruit inter-seed et baseline@128 lui-meme > 7.64, "
        f"alors le gate 7.64 (mesure a max_seq_len=4096) ne s'applique pas a conditions egales.")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)
    log("SAVED " + OUT_JSON)
    log(json.dumps({k: res[k] for k in ('baseline_128', 'block_local_128', 'paired_deltas', 'wilcoxon')}, indent=2))


if __name__ == '__main__':
    main()
