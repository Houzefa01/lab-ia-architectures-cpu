"""
STEP 7 — Gate LM perplexité (TinyShakespeare) pour BlockLocalTransformer.
Suit test.md §5 EXACTEMENT :
  - Data/char-vocab/split : identique benchmark_lm.py (chars=sorted(set(text)), 90/10 split).
  - Entraînement/éval : mécanique identique benchmark_retnet_3000.py (même script utilisé
    pour produire les nombres de comparaison 8.07 / 7.59 / 7.64 dans test.md et RAISONS_ECHEC.md) :
    AdamW lr=3e-4 wd=0.01, warmup manuel 100 steps (lr affine 0 -> lr), puis CosineAnnealingLR(opt, STEPS)
    step() seulement après warmup, clip grad norm 1.0, 3000 steps, batch=32, seq_len=128.
  - Eval : moyenne cross-entropy sur 100 batches test -> ppl = exp(mean_ce).
Gate : ppl <= 7.64 (baseline mature 3000 steps = 7.59).
Sortie : resultats/block_local_lm.json
"""
import sys, os, json, math, time, urllib.request
import torch, torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from candidats_actifs.block_local import BlockLocalTransformer, count_parameters

# ---- Hyperparams (test.md §5 + spec chief STEP 7) ----
SEED = 42
DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
BLOCK_SIZE = 64
SEQ = 128
MAX_SEQ_LEN = 128  # design: T=128, C=64 -> n=2 blocs -> équivalent attention causale pleine
BATCH = 32
STEPS = 3000
WARMUP = 100
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_BATCHES = 100
GATE_THRESHOLD = 7.64
BASELINE_PPL = 7.59
LOG_EVERY = 500

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats', 'block_local_lm.json')


def log(m):
    print(m, flush=True)


def get_lm():
    dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tinyshakespeare.txt')
    if not os.path.exists(dpath):
        os.makedirs(os.path.dirname(dpath), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            dpath)
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


def main():
    train_data, test_data, vs = get_lm()
    log(f"Vocab size (char-level TinyShakespeare): {vs}")
    log(f"Train chars: {len(train_data)}  Test chars: {len(test_data)}")

    torch.manual_seed(SEED)

    model = BlockLocalTransformer(
        vocab_size=vs, dim=DIM, heads=HEADS, layers=LAYERS,
        ffn_dim=FFN, max_seq_len=MAX_SEQ_LEN, block_size=BLOCK_SIZE,
    )
    log(f"BlockLocalTransformer params: {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    model.train()
    t0 = time.time()
    loss_curve = []
    running = None
    for step in range(STEPS):
        x, y = batch(train_data, BATCH)
        if step < WARMUP:
            opt.param_groups[0]['lr'] = LR * step / WARMUP
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vs), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step >= WARMUP:
            sched.step()

        lv = loss.item()
        running = lv if running is None else 0.98 * running + 0.02 * lv

        if (step + 1) % LOG_EVERY == 0 or step == 0:
            elapsed = time.time() - t0
            cur_lr = opt.param_groups[0]['lr']
            loss_curve.append({
                "step": step + 1, "train_loss": lv, "running_loss": running,
                "lr": cur_lr, "elapsed_s": elapsed,
            })
            log(f"step {step + 1:5d}/{STEPS}  loss={lv:.4f}  running={running:.4f}  "
                f"lr={cur_lr:.2e}  elapsed={elapsed:.0f}s")

    train_time = time.time() - t0
    log(f"Training finished in {train_time:.1f}s ({STEPS} steps)")

    model.eval()
    tl = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = batch(test_data, BATCH)
            tl += F.cross_entropy(model(x).reshape(-1, vs), y.reshape(-1)).item()
    mean_ce = tl / EVAL_BATCHES
    ppl = math.exp(mean_ce)
    wall_time = time.time() - t0

    passed = ppl <= GATE_THRESHOLD

    log("=" * 60)
    log(f"FINAL PPL = {ppl:.4f}  mean_ce={mean_ce:.4f}  "
        f"(gate <= {GATE_THRESHOLD}, baseline {BASELINE_PPL}) -> {'PASS' if passed else 'FAIL'}")
    log("=" * 60)

    result = {
        "model": "BlockLocalTransformer",
        "seed": SEED,
        "steps": STEPS,
        "final_perplexity": ppl,
        "final_mean_ce": mean_ce,
        "gate_threshold": GATE_THRESHOLD,
        "baseline_ppl": BASELINE_PPL,
        "passed": passed,
        "loss_curve": loss_curve,
        "wall_time_s": wall_time,
        "train_time_s": train_time,
        "hyperparams": {
            "vocab_size": vs, "dim": DIM, "heads": HEADS, "layers": LAYERS,
            "ffn_dim": FFN, "block_size": BLOCK_SIZE, "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH, "seq_len": SEQ, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP, "grad_clip": GRAD_CLIP, "eval_batches": EVAL_BATCHES,
            "n_params": count_parameters(model),
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"Saved {OUT_JSON}")


if __name__ == '__main__':
    main()
