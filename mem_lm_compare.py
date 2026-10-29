"""LM : Raisonneur+Mémoire (#16 intégré) vs Block-Local pur (#14=7.79).
Même harness que lm_fair_comparison (max_seq_len=128, 3000 steps, eval fixe seed 999).
Deux optimiseurs : AdamW (dense) + SparseAdam (values mémoire). Sortie : resultats/mem_lm.json"""
import torch, torch.nn.functional as F, math, os, json, time, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
torch.set_num_threads(2)
from candidats_actifs.block_local_mem import BlockLocalMemTransformer, count_parameters

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
    tr, te, vs = get_lm()
    torch.manual_seed(42)
    m = BlockLocalMemTransformer(vocab_size=vs, dim=64, heads=2, layers=3, ffn_dim=256,
                                 max_seq_len=SEQ, block_size=64, n_keys=256, k=16)
    sparse_p = [p for n, p in m.named_parameters() if "mem.values" in n]
    dense_p = [p for n, p in m.named_parameters() if "mem.values" not in n]
    print(f"params total={count_parameters(m):,} (dont mémoire {sum(p.numel() for p in sparse_p):,})", flush=True)
    dense = torch.optim.AdamW(dense_p, lr=LR, weight_decay=0.01)
    sparse = torch.optim.SparseAdam(sparse_p, lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(dense, STEPS)
    m.train(); t0 = time.time()
    for step in range(STEPS):
        x, y = batch(tr)
        if step < WARMUP: dense.param_groups[0]['lr'] = LR * step / WARMUP
        loss = F.cross_entropy(m(x).reshape(-1, vs), y.reshape(-1))
        dense.zero_grad(); sparse.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_p, 1.0); dense.step(); sparse.step()
        if step >= WARMUP: sched.step()
        if (step+1) % 500 == 0: print(f"step {step+1}/{STEPS} loss={loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    m.eval(); torch.manual_seed(999); tl = 0.0
    with torch.no_grad():
        for _ in range(EVAL):
            x, y = batch(te); tl += F.cross_entropy(m(x).reshape(-1, vs), y.reshape(-1)).item()
    ppl = math.exp(tl / EVAL)
    memL = None
    for b in m.blocks:
        if hasattr(b, "mem"): memL = b.mem
    res = {"model": "BlockLocalMem", "ppl": ppl, "block_local_pur_ppl": 7.786,
           "mem_slots": memL.N, "octets_lus_par_token_memoire": memL.bytes_read_per_token(),
           "params_total": count_parameters(m), "params_memoire": memL.N * 64,
           "wall_s": time.time() - t0}
    json.dump(res, open(os.path.join(ROOT, "resultats", "mem_lm.json"), "w"), indent=2)
    print(f"\n=== ppl BlockLocalMem = {ppl:.3f}  (block-local pur = 7.786) ===", flush=True)
    print(f"mémoire: {memL.N:,} slots, {memL.bytes_read_per_token()/1024:.1f} Ko lus/token", flush=True)


if __name__ == "__main__":
    main()
