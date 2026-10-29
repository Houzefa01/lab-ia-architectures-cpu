"""STEP 8 — vitesse + mémoire de BlockLocalTransformer, comparé au baseline.
Réutilise bench_efficiency.bench (même méthode que resultats/baseline_ref.json)."""
import functools, json, os
from bench_efficiency import bench
from candidats_actifs.block_local import BlockLocalTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
factory = functools.partial(BlockLocalTransformer, vocab_size=65, dim=64, heads=2,
                            layers=3, ffn_dim=256, max_seq_len=4096, block_size=64)
res = bench(factory, "BlockLocalTransformer")
out = os.path.join(ROOT, "resultats", "block_local_speed_mem.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print("SAVED", out)

# Comparaison directe au baseline
with open(os.path.join(ROOT, "resultats", "baseline_ref.json")) as f:
    base = json.load(f)

print("\n=== COMPARAISON vitesse+mémoire (batch=32) ===")
print(f"{'T':>6} | {'baseline ms':>12} {'BL ms':>10} {'speedup':>8} | {'base MB':>9} {'BL MB':>8} {'mem x':>7}")
for T in ["48","128","256","512","1024","2048","4096"]:
    b = base["lengths"].get(T, {}); l = res["lengths"].get(T, {})
    bms = b.get("latency_ms"); lms = l.get("latency_ms")
    bmb = b.get("peak_ram_mb"); lmb = l.get("peak_ram_mb")
    bstat = b.get("status"); lstat = l.get("status")
    sp = f"{bms/lms:.2f}x" if (bms and lms) else "-"
    mx = f"{bmb/lmb:.2f}x" if (bmb and lmb) else "-"
    bms_s = f"{bms:.1f}" if bms else bstat
    lms_s = f"{lms:.1f}" if lms else lstat
    bmb_s = f"{bmb:.0f}" if bmb else "-"
    lmb_s = f"{lmb:.0f}" if lmb else "-"
    print(f"{T:>6} | {str(bms_s):>12} {str(lms_s):>10} {sp:>8} | {bmb_s:>9} {lmb_s:>8} {mx:>7}")
