"""Critère 10 (scaling) — processus FRAIS et dédié pour éviter le deadlock
fork+threads : le parent ne touche PAS torch avant les forks de bench_efficiency
(aucun modèle construit avant la boucle bench). Comptage des params APRÈS les benchs.
"""
import sys, os, json, functools
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.append(ROOT)
from bench_efficiency import bench  # importe torch mais ne lance aucune op

RES = os.path.join(ROOT, "resultats")
dims = [64, 128, 256]; lengths = [512, 1024, 2048]

# --- 1. Tous les benchs D'ABORD (parent propre, forks sains) ---
raw = {}
for dim in dims:
    heads = dim // 32; ffn = dim * 4
    from candidats_actifs.block_local import BlockLocalTransformer
    from baseline.transformer import BaselineTransformer
    blf = functools.partial(BlockLocalTransformer, vocab_size=65, dim=dim, heads=heads,
                            layers=3, ffn_dim=ffn, max_seq_len=2048, block_size=64)
    baf = functools.partial(BaselineTransformer, vocab_size=65, dim=dim, heads=heads,
                            layers=3, ffn_dim=ffn, max_seq_len=2048)
    bl = bench(blf, f"BL-d{dim}", lengths=lengths, heads=heads, verbose=True)
    ba = bench(baf, f"Base-d{dim}", lengths=lengths, heads=heads, verbose=True)
    raw[str(dim)] = {"block_local": bl["lengths"], "baseline": ba["lengths"]}

# --- 2. Comptage des params APRÈS (plus aucun fork ensuite) ---
from candidats_actifs.block_local import BlockLocalTransformer
from baseline.transformer import count_parameters
res = {"lengths": lengths, "by_dim": {}}
for dim in dims:
    heads = dim // 32; ffn = dim * 4
    npar = count_parameters(BlockLocalTransformer(vocab_size=65, dim=dim, heads=heads,
                            layers=3, ffn_dim=ffn, max_seq_len=2048, block_size=64))
    res["by_dim"][str(dim)] = {"n_params": npar, **raw[str(dim)]}

# --- 3. Projection 70B (RAM = borne solide) ---
dim70, layers70 = 8192, 80
params70 = 12 * layers70 * dim70 * dim70
res["projection_70b"] = {
    "hypoth_dim": dim70, "hypoth_layers": layers70, "params_approx": params70,
    "ram_fp16_GB": params70 * 2 / 1e9, "ram_int8_GB": params70 * 1 / 1e9, "ram_int4_GB": params70 * 0.5 / 1e9,
    "note": "RAM = borne solide (params x octets). Avantage Block-Local : attention O(T·C) au lieu de "
            "O(T²) -> activations ne saturent PAS la RAM à contexte long (Transformer OOM dès T=4096 à "
            "dim=64). 70B int4 ≈ 35 Go tient sur station CPU ; contexte long abordable en calcul (matmuls "
            "MKL bornés). Débit tokens/s à 70B dépend de l'implémentation (GGUF) -> roadmap."}

json.dump(res, open(os.path.join(RES, "scaling.json"), "w"), indent=2)
print("SAVED scaling.json")
for dim in dims:
    d = res["by_dim"][str(dim)]
    def ms(m, T):
        v = d[m].get(str(T), {}); return f"{v['latency_ms']:.0f}ms" if v.get("latency_ms") else v.get("status", "?")
    print(f"  dim={dim} ({d['n_params']:,}p)  BL@1024={ms('block_local',1024)} Base@1024={ms('baseline',1024)} | BL@2048={ms('block_local',2048)} Base@2048={ms('baseline',2048)}")
