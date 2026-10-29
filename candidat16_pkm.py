"""Candidat #16 — preuve de concept : mémoire à ACCÈS PARCIMONIEUX sur DISQUE.
Product-Key Memory (Lample 2019) : N slots (savoir), mais seulement k lus/token.
Démontre le DÉCOUPLAGE : capacité mémoire (sur disque) ⟂ octets lus par token.
Le raisonneur (petit, en RAM) reste responsable de l'intelligence ; la PKM ajoute le savoir.
Sortie : resultats/candidat16.json + valeurs mémoire dumpées en memmap disque.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, json, time
ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "resultats")
os.makedirs(RES, exist_ok=True)
torch.manual_seed(0)

QDIM, VDIM = 128, 64
H = 512                 # sous-clés par moitié -> N = H*H slots
N = H * H               # = 262144 slots de "savoir"
KH = 16                 # top-k par moitié -> KH*KH candidats
K = 32                  # top-k final (slots réellement lus)
M = 2000                # nombre d'associations à mémoriser (le "savoir")


class PKM(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = nn.Parameter(torch.randn(H, QDIM // 2) * 0.1)
        self.k2 = nn.Parameter(torch.randn(H, QDIM // 2) * 0.1)
        self.values = nn.Embedding(N, VDIM, sparse=True)  # "savoir" ; grad SPARSE (slots lus only)
        nn.init.normal_(self.values.weight, 0.0, 0.02)

    def topk_idx(self, q):
        q1, q2 = q[:, :QDIM // 2], q[:, QDIM // 2:]
        s1, i1 = (q1 @ self.k1.t()).topk(KH, dim=1)     # (B,KH)
        s2, i2 = (q2 @ self.k2.t()).topk(KH, dim=1)
        cand = (s1[:, :, None] + s2[:, None, :]).view(q.size(0), -1)   # (B,KH*KH)
        idx = (i1[:, :, None] * H + i2[:, None, :]).view(q.size(0), -1)
        sc, sel = cand.topk(K, dim=1)                    # (B,K)
        return torch.gather(idx, 1, sel), F.softmax(sc, dim=1)

    def forward(self, q):
        idx, w = self.topk_idx(q)                        # (B,K)
        vals = self.values(idx)                          # (B,K,VDIM) — seuls K slots lus
        return (w.unsqueeze(-1) * vals).sum(1)           # (B,VDIM)


def read_from_disk(mm, idx_np):
    """Lecture PARCIMONIEUSE depuis le memmap disque : seulement les lignes top-k."""
    return torch.from_numpy(np.stack([mm[i] for i in idx_np]))


if __name__ == "__main__":
    t0 = time.time()
    # raisonneur minimal : id -> query (embedding appris) ; en usage réel = block-local
    emb = nn.Embedding(M, QDIM)
    reasoner = nn.Sequential(nn.Linear(QDIM, QDIM), nn.GELU(), nn.Linear(QDIM, QDIM))
    pkm = PKM()
    targets = F.normalize(torch.randn(M, VDIM), dim=1)   # le "fait" à mémoriser par item
    dense = torch.optim.AdamW(list(emb.parameters()) + list(reasoner.parameters()) + [pkm.k1, pkm.k2], lr=1e-3)
    sparse = torch.optim.SparseAdam(list(pkm.values.parameters()), lr=1e-2)  # slots lus only

    ids = torch.arange(M)
    for step in range(1200):
        b = ids[torch.randint(0, M, (128,))]
        q = reasoner(emb(b))
        out = F.normalize(pkm(q), dim=1)
        loss = (1 - (out * targets[b]).sum(1)).mean()    # 1 - cosine
        dense.zero_grad(); sparse.zero_grad(); loss.backward(); dense.step(); sparse.step()
        if step % 200 == 0: print(f"step {step} loss={loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

    # recall global (argmax cosine sur les M cibles)
    with torch.no_grad():
        q = reasoner(emb(ids)); out = F.normalize(pkm(q), dim=1)
        sims = out @ targets.t()                          # (M,M)
        recall = (sims.argmax(1) == ids).float().mean().item()

    # DUMP du "savoir" sur DISQUE (memmap) puis lecture PARCIMONIEUSE
    path = os.path.join(RES, "pkm_values.dat")
    mm = np.memmap(path, dtype='float32', mode='w+', shape=(N, VDIM))
    mm[:] = pkm.values.weight.detach().numpy(); mm.flush()
    disk_bytes = os.path.getsize(path)
    with torch.no_grad():
        idx, w = pkm.topk_idx(reasoner(emb(ids[:256])))
        vals_disk = read_from_disk(mm, idx[0].numpy())    # lit K lignes depuis le disque
    bytes_per_token = K * VDIM * 4

    res = {
        "N_slots": N, "M_associations": M, "K_lus_par_token": K,
        "recall_global": recall,
        "memoire_disque_MB": disk_bytes / 1e6,
        "octets_lus_par_token": bytes_per_token,
        "fraction_lue_par_token": bytes_per_token / disk_bytes,
        "params_raisonneur": sum(p.numel() for p in list(emb.parameters()) + list(reasoner.parameters())),
        "hdd_150MBs_temps_lecture_par_token_ms": bytes_per_token / (150e6) * 1000,
    }
    json.dump(res, open(os.path.join(RES, "candidat16.json"), "w"), indent=2)
    print("\n=== #16 PKM résultats ===")
    print(f"Savoir: {N:,} slots ({disk_bytes/1e6:.1f} Mo sur disque), {M} associations mémorisées")
    print(f"Recall associatif global: {recall*100:.1f}%")
    print(f"Lu par token: {bytes_per_token/1024:.1f} Ko = {res['fraction_lue_par_token']*100:.4f}% de la mémoire")
    print(f"Temps lecture disque/token @HDD 150Mo/s: {res['hdd_150MBs_temps_lecture_par_token_ms']:.3f} ms")
    print("SAVED resultats/candidat16.json")
