"""Test DÉCISIF de portée globale (#15 vs #14) — copy gap=400 >> champ réceptif local (192).
Version ÉCONOME MÉMOIRE (machine ~5Go) : batches générés à la volée (pas de dataset pré-stocké),
batch=16, test set fixe réduit, libération du 1er modèle avant le 2e.
Sortie : resultats/copy_longrange.json
"""
import torch, torch.nn.functional as F, sys, time, os, json, gc
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from candidats_actifs.block_local import BlockLocalTransformer
from candidats_actifs.block_local_global import BlockLocalGlobalTransformer

torch.set_num_threads(2)
GAP = 400
DIM, HEADS, LAYERS, FFN, BLOCK = 64, 2, 3, 256, 64
BATCH, STEPS = 16, 3000
N_TEST = 500
SL = 3 + GAP + 3
EP = 3 + GAP


def gen_batch(bs, seed=None):
    if seed is not None: torch.manual_seed(seed)
    k1 = torch.randint(2, 9, (bs,)); k2 = torch.randint(2, 9, (bs,))
    s = torch.randint(2, 9, (bs, SL))
    s[:, 0] = 0; s[:, 1] = k1; s[:, 2] = k2
    s[:, EP] = 1; s[:, EP + 1] = k2; s[:, EP + 2] = k1
    t = torch.zeros(bs, SL, dtype=torch.long); t[:, :-1] = s[:, 1:]
    return s, t


def eval_acc(model, vx, vy):
    model.eval()
    with torch.no_grad():
        # éval par sous-lots pour limiter la mémoire
        cor = 0
        for i in range(0, vx.size(0), 50):
            p = model(vx[i:i+50])[:, EP:EP+2, :].argmax(-1)
            cor += (p == vy[i:i+50, EP:EP+2]).all(dim=1).sum().item()
    model.train()
    return cor / vx.size(0)


def train_eval(name, ctor):
    torch.manual_seed(42)
    model = ctor(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN,
                 max_seq_len=SL, block_size=BLOCK)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    vx, vy = gen_batch(N_TEST, seed=43)
    model.train(); t0 = time.perf_counter(); traj = []
    for step in range(STEPS):
        x, y = gen_batch(BATCH)
        la = F.cross_entropy(model(x).reshape(-1, 10), y.reshape(-1), reduction='none').reshape(y.shape)
        loss = la[:, :EP].mean() + la[:, EP+2:].mean() + la[:, EP:EP+2].mean() * 100.0
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == STEPS - 1:
            acc = eval_acc(model, vx, vy)
            traj.append({"step": step, "acc": acc})
            print(f"  [{name}] step {step}/{STEPS} loss={loss.item():.3f} acc={acc*100:.2f}% {time.perf_counter()-t0:.0f}s", flush=True)
    acc = eval_acc(model, vx, vy)
    print(f"==> {name} FINAL copy acc (gap={GAP}) = {acc*100:.2f}%", flush=True)
    del model, opt, sched; gc.collect()
    return acc, traj


if __name__ == "__main__":
    print(f"=== Copy longue portée gap={GAP} (seq={SL}), RF local={LAYERS*BLOCK}, batch={BATCH} ===", flush=True)
    res = {"gap": GAP, "seq_len": SL, "rf_local": LAYERS * BLOCK, "steps": STEPS, "batch": BATCH}
    a15, t15 = train_eval("BLG_#15", BlockLocalGlobalTransformer)
    a14, t14 = train_eval("BL_#14", BlockLocalTransformer)
    res.update({"blg_15_acc": a15, "bl_14_acc": a14, "blg_15_traj": t15, "bl_14_traj": t14,
                "blg_15_passed_95": a15 >= 0.95})
    res["verdict"] = ("#15 réussit là où #14 échoue -> portée globale prouvée pour le rappel EXACT"
                      if a15 >= 0.95 and a14 < 0.5 else
                      "#15 < 95% -> mémoire linéaire insuffisante pour rappel EXACT longue distance (honnête)"
                      if a15 < 0.95 else "voir chiffres")
    json.dump(res, open(os.path.join(ROOT, "resultats", "copy_longrange.json"), "w"), indent=2)
    print(f"SAVED. RESUME: #15={a15*100:.1f}%  #14={a14*100:.1f}%  ->  {res['verdict']}", flush=True)
