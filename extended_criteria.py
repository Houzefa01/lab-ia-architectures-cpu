"""Critères étendus 8/9/10 (objectif 70B sans GPU) pour Block-Local.
- 8 Généralisation de longueur : LM entraîné à seq_len=256 (2x la fenêtre LM standard),
  ppl globale + ppl par tranche de position (exploitation du contexte long) + audit causal T=256.
- 9 Quantization : int8 dynamique réel (ppl fp32 vs int8, taille poids), int4 estimé.
- 10 Scaling : vitesse+mémoire block-local vs baseline à dim {64,128,256}, projection 70B.
Sorties : resultats/length_gen.json, resultats/quantization.json, resultats/scaling.json
"""
import sys, os, json, math, time, functools
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from baseline.transformer import BaselineTransformer, count_parameters
from candidats_actifs.block_local import BlockLocalTransformer
from bench_efficiency import bench

RES = os.path.join(ROOT, "resultats")


def log(m): print(m, flush=True)


def get_lm():
    with open(os.path.join(ROOT, "data", "tinyshakespeare.txt")) as f:
        text = f.read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def batch(d, bs, sl):
    ix = torch.randint(len(d) - sl, (bs,))
    return (torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix]))


def eval_ppl(model, test_data, vs, sl, seed=999, nb=100):
    model.eval(); torch.manual_seed(seed); tl = 0.0
    with torch.no_grad():
        for _ in range(nb):
            x, y = batch(test_data, 32, sl)
            tl += F.cross_entropy(model(x).reshape(-1, vs), y.reshape(-1)).item()
    return math.exp(tl / nb)


def eval_ppl_by_pos(model, test_data, vs, sl, half, seed=999, nb=100):
    """ppl moyenne sur positions [0:half) vs [half:sl)."""
    model.eval(); torch.manual_seed(seed); c1 = c2 = 0.0
    with torch.no_grad():
        for _ in range(nb):
            x, y = batch(test_data, 32, sl)
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, vs), y.reshape(-1), reduction='none').reshape(x.shape)
            c1 += ce[:, :half].mean().item(); c2 += ce[:, half:].mean().item()
    return math.exp(c1/nb), math.exp(c2/nb)


def train_lm(model, train_data, vs, sl, steps=2000, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train(); t0 = time.time()
    for step in range(steps):
        x, y = batch(train_data, 32, sl)
        if step < 100: opt.param_groups[0]['lr'] = 3e-4 * step / 100
        loss = F.cross_entropy(model(x).reshape(-1, vs), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step >= 100: sched.step()
        if (step+1) % 500 == 0: log(f"  [len256] step {step+1}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s")
    return model


# ===== Critère 8 : généralisation de longueur (LM @ seq_len=256) =====
def crit8():
    log("== Critère 8 : généralisation de longueur (seq_len=256) ==")
    tr, te, vs = get_lm()
    SL = 256
    m = BlockLocalTransformer(vocab_size=vs, dim=64, heads=2, layers=3, ffn_dim=256,
                              max_seq_len=SL, block_size=64)
    m = train_lm(m, tr, vs, SL, steps=2000)
    ppl = eval_ppl(m, te, vs, SL)
    ppl_first, ppl_second = eval_ppl_by_pos(m, te, vs, SL, half=128)
    # audit causal à T=256
    torch.manual_seed(0); s1 = torch.randint(0, vs, (2, SL)); s2 = s1.clone(); s2[:, SL//2:] = torch.randint(0, vs, (2, SL//2))
    m.eval()
    with torch.no_grad(): diff = (m(s1)[:, :SL//2] - m(s2)[:, :SL//2]).abs().max().item()
    out = {"seq_len": SL, "steps": 2000, "ppl_global": ppl,
           "ppl_pos_0_128": ppl_first, "ppl_pos_128_256": ppl_second,
           "exploite_contexte_long": ppl_second < ppl_first, "causal_diff_T256": diff,
           "note": "pos_emb absolues -> fonctionne jusqu'à max_seq_len. Extension STRICTE au-delà "
                   "de la longueur d'entraînement nécessite RoPE/ALiBi (recommandation), petit "
                   "changement compatible avec le design par blocs."}
    json.dump(out, open(os.path.join(RES, "length_gen.json"), "w"), indent=2)
    log(f"  ppl={ppl:.3f} | pos[0:128]={ppl_first:.3f} pos[128:256]={ppl_second:.3f} "
        f"(exploite contexte long={ppl_second<ppl_first}) | causal_diff={diff:.1e}")
    return m, te, vs, SL


# ===== Critère 9 : quantization int8 (réel) + int4 (estimé) =====
def crit9(model, test_data, vs, sl):
    log("== Critère 9 : quantization int8 dynamique ==")
    ppl_fp32 = eval_ppl(model, test_data, vs, sl)
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    ppl_int8 = eval_ppl(qmodel, test_data, vs, sl)
    # tailles : params dans les Linear (quantifiables) vs reste (embeddings/norm)
    lin_params = sum(p.numel() for mod in model.modules() if isinstance(mod, nn.Linear) for p in mod.parameters())
    tot_params = sum(p.numel() for p in model.parameters())
    other = tot_params - lin_params
    size_fp32 = tot_params * 4
    size_int8 = lin_params * 1 + other * 4
    size_int4 = lin_params * 0.5 + other * 4
    out = {"ppl_fp32": ppl_fp32, "ppl_int8": ppl_int8, "delta_ppl_int8": ppl_int8 - ppl_fp32,
           "params_total": tot_params, "params_linear": lin_params,
           "size_fp32_bytes": size_fp32, "size_int8_bytes": size_int8, "size_int4_bytes_est": size_int4,
           "reduction_int8": size_fp32 / size_int8, "reduction_int4_est": size_fp32 / size_int4,
           "note": "int8 dynamique = réel (torch.ao). int4 = estimation de taille (l'exécution int4 "
                   "CPU réelle passe par GGUF/llama.cpp-style, hors PyTorch natif)."}
    json.dump(out, open(os.path.join(RES, "quantization.json"), "w"), indent=2)
    log(f"  ppl fp32={ppl_fp32:.3f} -> int8={ppl_int8:.3f} (delta {ppl_int8-ppl_fp32:+.3f}) | "
        f"RAM int8={size_fp32/size_int8:.2f}x, int4(est)={size_fp32/size_int4:.2f}x")


# ===== Critère 10 : scaling dim {64,128,256} + projection 70B =====
def crit10():
    log("== Critère 10 : scaling vitesse+mémoire + projection 70B ==")
    dims = [64, 128, 256]; lengths = [512, 1024, 2048]
    res = {"lengths": lengths, "by_dim": {}}
    for dim in dims:
        heads = dim // 32; ffn = dim * 4
        blf = functools.partial(BlockLocalTransformer, vocab_size=65, dim=dim, heads=heads,
                                layers=3, ffn_dim=ffn, max_seq_len=2048, block_size=64)
        baf = functools.partial(BaselineTransformer, vocab_size=65, dim=dim, heads=heads,
                                layers=3, ffn_dim=ffn, max_seq_len=2048)
        npar = count_parameters(blf())
        bl = bench(blf, f"BL-d{dim}", lengths=lengths, heads=heads, verbose=False)
        ba = bench(baf, f"Base-d{dim}", lengths=lengths, heads=heads, verbose=False)
        res["by_dim"][str(dim)] = {"n_params": npar, "block_local": bl["lengths"], "baseline": ba["lengths"]}
        log(f"  dim={dim} ({npar:,} params) benché")
    # projection 70B : dimensionnement approx d'un transformer ~70B
    dim70, layers70 = 8192, 80
    params70 = 12 * layers70 * dim70 * dim70  # ~ approx (attn+ffn)
    ram_int4 = params70 * 0.5 / 1e9
    ram_int8 = params70 * 1.0 / 1e9
    ram_fp16 = params70 * 2.0 / 1e9
    res["projection_70b"] = {
        "hypoth_dim": dim70, "hypoth_layers": layers70, "params_approx": params70,
        "ram_fp16_GB": ram_fp16, "ram_int8_GB": ram_int8, "ram_int4_GB": ram_int4,
        "note": "RAM = borne solide (params x octets). L'avantage clé de Block-Local pour ce régime : "
                "l'attention est O(T·C) au lieu de O(T²) -> le KV/activations ne saturent PAS la RAM à "
                "contexte long (le Transformer OOM dès T=4096 à dim=64). Un 70B int4 ≈ 35 Go tient sur "
                "station CPU ; Block-Local rend le contexte long abordable en calcul (matmuls MKL bornés). "
                "Le débit tokens/s à 70B dépend de l'implémentation (GGUF) et reste à mesurer -> roadmap."}
    json.dump(res, open(os.path.join(RES, "scaling.json"), "w"), indent=2)
    log(f"  projection 70B: int4≈{ram_int4:.0f}Go int8≈{ram_int8:.0f}Go fp16≈{ram_fp16:.0f}Go")


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    m, te, vs, sl = crit8()
    crit9(m, te, vs, sl)
    crit10()
    log("EXTENDED CRITERIA DONE")
