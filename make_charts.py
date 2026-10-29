"""Step 12 — visualisations des résultats Block-Local (matplotlib -> PNG).
Lit les JSON de resultats/ et produit resultats/block_local_results.png.
Robuste aux longueurs OOM/timeout (latency_ms absent)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "resultats")
def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else None

BASE = "#4C6EF5"; BL = "#F76707"; OK = "#2F9E44"; GREY = "#868E96"
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "axes.titleweight": "bold"})

base = load("baseline_ref.json"); blsm = load("block_local_speed_mem.json")
copy = load("block_local_copy.json"); lm = load("lm_fair_comparison.json")
scal = load("scaling.json"); quant = load("quantization.json"); lg = load("length_gen.json")

fig, ax = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Block-Local Attention (#14) vs Transformer causal — CPU, dim=64", fontsize=15, fontweight="bold")

def series(d, field):
    xs, ys = [], []
    if not d: return xs, ys
    for T, v in d["lengths"].items():
        if isinstance(v, dict) and v.get(field) is not None:
            xs.append(int(T)); ys.append(v[field])
    z = sorted(zip(xs, ys)); return [a for a, _ in z], [b for _, b in z]

# 1. Latence
a = ax[0, 0]
bx, by = series(base, "latency_ms"); lx, ly = series(blsm, "latency_ms")
a.plot(bx, by, "o-", color=BASE, label="Baseline"); a.plot(lx, ly, "s-", color=BL, label="Block-Local")
a.axvline(4096, color=GREY, ls=":", lw=1); a.text(4096, a.get_ylim()[1], " baseline OOM", color=GREY, fontsize=7, va="top")
a.set(xscale="log", yscale="log", xlabel="T (contexte)", ylabel="latence ms", title="Latence inférence (batch=32)"); a.legend()

# 2. Mémoire
a = ax[0, 1]
bx, by = series(base, "peak_ram_mb"); lx, ly = series(blsm, "peak_ram_mb")
a.plot(bx, by, "o-", color=BASE, label="Baseline"); a.plot(lx, ly, "s-", color=BL, label="Block-Local")
a.set(xscale="log", yscale="log", xlabel="T", ylabel="pic RAM MB", title="Mémoire (O(T²) vs O(T·C))"); a.legend()

# 3. Speedup
a = ax[0, 2]
sp = []
if base and blsm:
    for T in ["256", "512", "1024", "2048"]:
        b = base["lengths"].get(T, {}); l = blsm["lengths"].get(T, {})
        if b.get("latency_ms") and l.get("latency_ms"): sp.append((T, b["latency_ms"]/l["latency_ms"]))
a.bar([t for t, _ in sp], [v for _, v in sp], color=BL)
for i, (t, v) in enumerate(sp): a.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
a.axhline(1, color=GREY, ls="--"); a.set(xlabel="T", ylabel="speedup", title="Accélération vs baseline")

# 4. Copy accuracy
a = ax[0, 3]
if copy:
    tr = copy["accuracy_trajectory"]; a.plot([p["step"] for p in tr], [p["test_acc"]*100 for p in tr], "o-", color=OK)
    a.axhline(95, color=GREY, ls="--", label="gate 95%")
a.set(xlabel="steps", ylabel="copy accuracy %", title="Copy task (gap=186) -> 100%", ylim=(0, 105)); a.legend()

# 5. LM ppl par seed
a = ax[1, 0]
if lm:
    seeds = ["42", "43", "44"]; import numpy as np; x = np.arange(len(seeds)); w = 0.38
    a.bar(x-w/2, [lm["baseline_128"][s] for s in seeds], w, color=BASE, label="Baseline")
    a.bar(x+w/2, [lm["block_local_128"][s] for s in seeds], w, color=BL, label="Block-Local")
    a.set_xticks(x); a.set_xticklabels([f"seed {s}" for s in seeds])
    a.set(ylabel="perplexité", title="LM parité (max_seq_len=128)", ylim=(7.5, 7.9)); a.legend()

# 6. Scaling latence @T=1024 vs dim
a = ax[1, 1]
if scal:
    dims = sorted(int(d) for d in scal["by_dim"])
    bl_l = [scal["by_dim"][str(d)]["block_local"].get("1024", {}).get("latency_ms") for d in dims]
    ba_l = [scal["by_dim"][str(d)]["baseline"].get("1024", {}).get("latency_ms") for d in dims]
    a.plot(dims, ba_l, "o-", color=BASE, label="Baseline"); a.plot(dims, bl_l, "s-", color=BL, label="Block-Local")
    a.set(xlabel="dim", ylabel="latence ms @T=1024", title="Scaling en dim"); a.legend()

# 7. Quantization RAM + ppl
a = ax[1, 2]
if quant:
    labels = ["fp32", "int8", "int4*"]; red = [1.0, quant["reduction_int8"], quant["reduction_int4_est"]]
    a.bar(labels, red, color=[GREY, BASE, BL])
    for i, v in enumerate(red): a.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    a.set(ylabel="réduction RAM (x)", title=f"Quantization (ppl fp32={quant['ppl_fp32']:.2f}->int8={quant['ppl_int8']:.2f})")

# 8. Projection 70B RAM
a = ax[1, 3]
if scal and "projection_70b" in scal:
    p = scal["projection_70b"]; labels = ["fp16", "int8", "int4"]
    vals = [p["ram_fp16_GB"], p["ram_int8_GB"], p["ram_int4_GB"]]
    a.bar(labels, vals, color=[GREY, BASE, BL])
    for i, v in enumerate(vals): a.text(i, v, f"{v:.0f}Go", ha="center", va="bottom", fontsize=8)
    a.axhline(35, color=OK, ls="--", lw=1); a.text(0, 36, "≈ station CPU", color=OK, fontsize=7)
    a.set(ylabel="RAM (Go)", title="Projection 70B (poids)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(RES, "block_local_results.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print("SAVED", out)
