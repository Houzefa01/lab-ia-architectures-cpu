"""LM parité #17 (paginé, gate) vs block-local pur (#14=7.79) / #16 (10.56).
Harness identique à mem_lm_compare.py, mais UN SEUL AdamW (+ aux loss), wd=0 sur mémoire."""
import torch, torch.nn.functional as F, math, os, json, time, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
torch.set_num_threads(2)
from candidats_actifs.block_local_pagemem import BlockLocalPageMemTransformer, count_parameters

SEQ, STEPS, BATCH, WARMUP, LR, EVAL = 128, 3000, 32, 100, 3e-4, 100


def get_lm():
    text = open(os.path.join(ROOT, "data", "tinyshakespeare.txt")).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    d = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(d))
    return d[:n], d[n:], len(chars)


def batch(d):
    ix = torch.randint(len(d) - SEQ, (BATCH,))
    return torch.stack([d[i:i+SEQ] for i in ix]), torch.stack([d[i+1:i+SEQ+1] for i in ix])


def main():
    tr, te, vs = get_lm(); torch.manual_seed(42)
    m = BlockLocalPageMemTransformer(vocab_size=vs, dim=64, heads=2, layers=3, ffn_dim=256,
                                     max_seq_len=SEQ, block_size=64, n_pages=1024, slots_per_page=32, p=2)
    memp = {id(p) for n_, p in m.named_parameters() if any(k in n_ for k in ("K_pages", "K_slots", "V_slots"))}
    g_mem = [p for p in m.parameters() if id(p) in memp]
    g_std = [p for p in m.parameters() if id(p) not in memp]
    opt = torch.optim.AdamW([{"params": g_std, "weight_decay": 0.01},
                             {"params": g_mem, "weight_decay": 0.0}], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    print(f"params={count_parameters(m):,}", flush=True)
    m.train(); t0 = time.time()
    for step in range(STEPS):
        x, y = batch(tr)
        if step < WARMUP: opt.param_groups[0]['lr'] = opt.param_groups[1]['lr'] = LR * step / WARMUP
        loss = F.cross_entropy(m(x).reshape(-1, vs), y.reshape(-1)) + m.aux_loss()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step >= WARMUP: sched.step()
        if (step+1) % 500 == 0: print(f"step {step+1}/{STEPS} loss={loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    m.eval(); torch.manual_seed(999); tl = 0.0
    with torch.no_grad():
        for _ in range(EVAL):
            x, y = batch(te); tl += F.cross_entropy(m(x).reshape(-1, vs), y.reshape(-1)).item()
    ppl = math.exp(tl / EVAL)
    res = {"model": "BlockLocalPageMem_#17", "ppl": ppl, "block_local_14": 7.786, "mem_16": 10.56,
           "gate_alpha_max": float(torch.sigmoid(m.mem_block().alpha).max()), "wall_s": time.time()-t0}
    json.dump(res, open(os.path.join(ROOT, "resultats", "mem17_lm.json"), "w"), indent=2)
    print(f"\n=== #17 ppl = {ppl:.3f}  (#14=7.786, #16=10.56)  gate ouvert à {res['gate_alpha_max']:.3f} ===", flush=True)


if __name__ == "__main__":
    main()
