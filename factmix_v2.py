"""FactMix pour #17-v2 (routage saillant). Réutilise le générateur/harnais de mem17_factmix.
Cible: chiffre >=90% (v1=11.6%, #14=hasard 10%). Sortie: resultats/factmix_v2.json"""
import torch, os, json, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
torch.set_num_threads(2)
from mem17_factmix import gen_batch, eval_fact, train, M, SEQ, VS, STEPS
from candidats_actifs.block_local_pagemem_v2 import BlockLocalPageMemV2Transformer

torch.manual_seed(42)
m = BlockLocalPageMemV2Transformer(vocab_size=VS, dim=64, heads=2, layers=3, ffn_dim=256,
                                   max_seq_len=SEQ, block_size=64, n_pages=1024, slots_per_page=32, p=3)
a, dg = train("#17v2", m)   # gate par token, pas d'alpha forcé

mem = m.mem_block().mem
with torch.no_grad(): m(gen_batch(1)[0])
st = mem.access_stats(SEQ, 64)
# gate conditionnel : g moyen sur remplissage vs positions de valeur
with torch.no_grad():
    x, y, idx, vp = gen_batch(64); h = None
    mb = m.mem_block()
    # récupère g via un hook léger : recalcule g sur l'entrée de la mémoire
    hh = m.token_emb(x) + m.pos_emb(torch.arange(SEQ).unsqueeze(0))
    for i, b in enumerate(m.blocks):
        if b is mb:
            hin = mb.norm3(hh + 0)  # approx (avant les sous-couches) -> juste ordre de grandeur
        hh = b(hh, None) if False else hh  # évite double forward; on estime g sur hin plus bas
    g_all = torch.sigmoid(mb.mem.w_g(mb.norm3(m.token_emb(x) + m.pos_emb(torch.arange(SEQ).unsqueeze(0))))) * torch.sigmoid(mb.mem.alpha)
    g_fill = g_all[:, 10:SEQ-10].mean().item()
    g_val = g_all[:, [p-1 for p in vp]].mean().item()

res = {"model": "#17-v2", "acc_exact": a, "acc_digit": dg, "v1_digit": 0.116, "hasard": 0.10,
       "acces_hdd_par_token": st["acces_par_token"], "gate_remplissage": g_fill, "gate_valeur": g_val,
       "alpha": float(torch.sigmoid(mem.alpha))}
json.dump(res, open(os.path.join(ROOT, "resultats", "factmix_v2.json"), "w"), indent=2)
print(f"\n=== #17-v2 FactMix: chiffre={dg*100:.1f}% (v1=11.6%, hasard=10%)  fait_exact={a*100:.1f}% ===")
print(f"accès HDD/token={st['acces_par_token']:.4f} | gate remplissage={g_fill:.4f} valeur={g_val:.4f}", flush=True)
