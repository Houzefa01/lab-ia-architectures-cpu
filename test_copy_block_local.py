"""
STEP 6 GATE — Copy task pour BlockLocalTransformer (candidats_actifs/block_local.py)

Suit test.md section 4 EXACTEMENT, en réutilisant les conventions de
test_copy_correct.py (targets shiftés) et benchmark_cdcn.py (perte pondérée x100,
optimiseur AdamW lr=1e-4, CosineAnnealingLR, gap/positions de recall).

Gate: accuracy >= 95% (les 2 positions de recall correctes simultanément).
"""
import torch, torch.nn.functional as F, math, sys, time, os, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from candidats_actifs.block_local import BlockLocalTransformer

SEED = 42
DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
BATCH, STEPS = 32, 3000
GAP = 186
BLOCK_SIZE = 64
N_TRAIN, N_TEST = 8000, 2000
EVAL_EVERY = 500

def log(m):
    print(m, flush=True)

def gen(n, gap, seed):
    """Copy task data-gen, identique à benchmark_cdcn.py / test.md §4."""
    torch.manual_seed(seed)
    sl = 3 + gap + 3
    inp, tgt = [], []
    for _ in range(n):
        k1 = int(torch.randint(2, 9, (1,))[0])
        k2 = int(torch.randint(2, 9, (1,))[0])
        s = torch.zeros(sl, dtype=torch.long)
        s[0] = 0            # SOS
        s[1] = k1
        s[2] = k2
        s[3:3 + gap] = torch.randint(2, 9, (gap,))
        e = 3 + gap
        s[e] = 1             # END
        s[e + 1] = k2        # recall 1
        s[e + 2] = k1        # recall 2
        t = torch.zeros(sl, dtype=torch.long)
        t[:-1] = s[1:]       # next-token shift
        t[-1] = 0
        inp.append(s)
        tgt.append(t)
    return torch.stack(inp), torch.stack(tgt), sl


def eval_accuracy(model, x, y, ep):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        p = logits[:, ep:ep + 2, :].argmax(-1)
        t = y[:, ep:ep + 2]
        cor = (p == t).all(dim=1).sum().item()
    model.train()
    return cor / x.size(0)


def main():
    torch.manual_seed(SEED)

    train_x, train_y, seq_len = gen(N_TRAIN, GAP, seed=42)
    test_x, test_y, _ = gen(N_TEST, GAP, seed=43)
    assert seq_len == 192, seq_len
    ep = 3 + GAP  # = 189, recall positions ep, ep+1

    log(f"seq_len={seq_len}, ep={ep}, train={train_x.shape}, test={test_x.shape}")

    torch.manual_seed(SEED)
    model = BlockLocalTransformer(
        vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS,
        ffn_dim=FFN, max_seq_len=seq_len, block_size=BLOCK_SIZE,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"BlockLocalTransformer params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y), batch_size=BATCH, shuffle=True
    )

    model.train()
    trajectory = []  # list of {step, loss, test_acc}
    t0 = time.perf_counter()

    data_iter = iter(loader)
    for step in range(STEPS):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        logits = model(x)
        loss_all = F.cross_entropy(
            logits.reshape(-1, 10), y.reshape(-1), reduction='none'
        ).reshape(y.shape)
        loss_r = loss_all[:, ep:ep + 2].mean() * 100.0
        loss_o = loss_all[:, :ep].mean() + loss_all[:, ep + 2:].mean()
        total = loss_o + loss_r

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            acc = eval_accuracy(model, test_x, test_y, ep)
            elapsed = time.perf_counter() - t0
            trajectory.append({"step": step, "loss": total.item(), "test_acc": acc})
            log(f"step {step:5d}/{STEPS}  loss={total.item():.4f}  "
                f"(loss_o={loss_o.item():.4f} loss_r={loss_r.item():.4f})  "
                f"test_acc={acc*100:.2f}%  elapsed={elapsed:.1f}s")

    wall_time_s = time.perf_counter() - t0

    # Final accuracy on full test set
    final_acc = eval_accuracy(model, test_x, test_y, ep)
    passed = final_acc >= 0.95

    log(f"\n=== FINAL: copy accuracy = {final_acc*100:.2f}%  "
        f"(gate >=95% -> {'PASS' if passed else 'FAIL'})  wall_time={wall_time_s:.1f}s ===")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats", "block_local_copy.json")
    result = {
        "model": "BlockLocalTransformer",
        "seed": SEED,
        "steps": STEPS,
        "final_copy_accuracy": final_acc,
        "gate_threshold": 0.95,
        "passed": bool(passed),
        "loss_curve": [round(pt["loss"], 4) for pt in trajectory],
        "accuracy_trajectory": [{"step": pt["step"], "test_acc": round(pt["test_acc"], 4)} for pt in trajectory],
        "wall_time_s": wall_time_s,
        "config": {
            "vocab_size": 10, "dim": DIM, "heads": HEADS, "layers": LAYERS,
            "ffn_dim": FFN, "max_seq_len": seq_len, "block_size": BLOCK_SIZE,
            "gap": GAP, "batch": BATCH, "n_train": N_TRAIN, "n_test": N_TEST,
            "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR", "n_params": n_params,
            "train_seed": 42, "test_seed": 43,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
