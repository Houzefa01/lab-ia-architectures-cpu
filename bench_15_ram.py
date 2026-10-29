"""RAM réelle (RSS) de #15 (Block-Local-Global) vs #14 vs baseline + RAM des poids fp32/int8/int4.
Processus frais (fork sain). Sortie : resultats/block_local_global_ram.json"""
import functools, json, os
import torch.nn as nn
from bench_efficiency import bench
from candidats_actifs.block_local_global import BlockLocalGlobalTransformer, count_parameters

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "resultats")

# 1. RAM RSS réelle sur les 7 longueurs (même méthode que baseline_ref / block_local_speed_mem)
factory = functools.partial(BlockLocalGlobalTransformer, vocab_size=65, dim=64, heads=2,
                            layers=3, ffn_dim=256, max_seq_len=4096, block_size=64)
res = bench(factory, "BlockLocalGlobal")

# 2. RAM des POIDS (fp32 vs int8 vs int4) — dominante pour un grand modèle
m = factory()
lin = sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Linear) for p in mod.parameters())
tot = sum(p.numel() for p in m.parameters()); other = tot - lin
res["weights_ram"] = {
    "params_total": tot, "params_linear": lin,
    "fp32_MB": tot * 4 / 1e6, "int8_MB": (lin * 1 + other * 4) / 1e6,
    "int4_MB": (lin * 0.5 + other * 4) / 1e6,
    "reduction_int8": tot * 4 / (lin * 1 + other * 4), "reduction_int4": tot * 4 / (lin * 0.5 + other * 4)}
json.dump(res, open(os.path.join(RES, "block_local_global_ram.json"), "w"), indent=2)
print("SAVED block_local_global_ram.json")

# 3. Comparaison RSS directe avec #14 et baseline
def load(n):
    p = os.path.join(RES, n); return json.load(open(p)) if os.path.exists(p) else None
base = load("baseline_ref.json"); bl = load("block_local_speed_mem.json")
print(f"\n=== RAM RSS réelle (Mo, batch=32) ===")
print(f"{'T':>6} | {'baseline':>10} {'#14 BL':>9} {'#15 BLG':>9}")
for T in ["48","256","1024","2048","4096"]:
    def mb(d):
        v = (d or {}).get("lengths",{}).get(T,{}) if d else {}
        return f"{v['peak_ram_mb']:.0f}" if v.get("peak_ram_mb") else v.get("status","-")
    print(f"{T:>6} | {mb(base):>10} {mb(bl):>9} {mb(res):>9}")
print(f"\n=== RAM POIDS #15 : fp32={res['weights_ram']['fp32_MB']:.2f}Mo  "
      f"int8={res['weights_ram']['int8_MB']:.2f}Mo ({res['weights_ram']['reduction_int8']:.2f}x)  "
      f"int4={res['weights_ram']['int4_MB']:.2f}Mo ({res['weights_ram']['reduction_int4']:.2f}x)")
