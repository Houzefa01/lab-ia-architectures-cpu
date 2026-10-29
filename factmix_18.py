"""Candidat #18 — RETRIEVER À CLÉ EXACTE (façon RAG, routage NON appris).
L'entité @eXXXX (4 chiffres) = adresse directe dans une table-mémoire sur DISQUE.
Lookup exact O(1), 1 accès/séquence. Le modèle apprend seulement à stocker/lire la valeur.
Prouve que l'archi raisonneur+mémoire-disque MARCHE quand le routage est exact (vs #17 appris = hasard).
Sortie: resultats/factmix_18.json
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, json, time, sys, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
torch.set_num_threads(2)
from mem17_factmix import gen_batch, M, SEQ, VS
from candidats_actifs.block_local import _build_block_mask, BlockLocalBlock

DIM, LAYERS, BLOCK, STEPS = 64, 3, 64, 2000
N_ADDR = 10000   # entités possibles (0000-9999)


def addr_of(seq):   # retriever déterministe : les 4 chiffres d'entité (positions 2..5) -> adresse
    return seq[:, 2] * 1000 + seq[:, 3] * 100 + seq[:, 4] * 10 + seq[:, 5]


class RetrievalMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = VS
        self.token_emb = nn.Embedding(VS, DIM); self.pos_emb = nn.Embedding(SEQ, DIM)
        self.blocks = nn.ModuleList([BlockLocalBlock(DIM, 2, 256, BLOCK) for _ in range(LAYERS)])
        self.mem = nn.Embedding(N_ADDR, DIM)            # savoir : adresse -> repr valeur (→ disque)
        self.W_read = nn.Linear(DIM, DIM, bias=False)
        self.norm = nn.LayerNorm(DIM); self.head = nn.Linear(DIM, VS, bias=False)
        self.mem_layer = LAYERS // 2
        self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, x, addr, mem_vec=None):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = _build_block_mask((T + BLOCK - 1) // BLOCK, BLOCK, x.device)
        mv = self.mem(addr) if mem_vec is None else mem_vec     # (B,DIM) lookup exact
        inj = self.W_read(mv).unsqueeze(1)                      # savoir récupéré
        seg = (torch.arange(T, device=x.device) >= 6).float().view(1, T, 1)  # entité déjà vue
        for i, b in enumerate(self.blocks):
            h = b(h, mask)
            if i == self.mem_layer: h = h + inj * seg
        return self.head(self.norm(h))


def evaluate(model, n=800, from_disk=None):
    model.eval(); cor_dig = cor_all = tot = 0
    with torch.no_grad():
        for _ in range(n // 32 + 1):
            x, y, idx, vp = gen_batch(32); addr = addr_of(x)
            mv = torch.from_numpy(np.stack([from_disk[a] for a in addr.numpy()])) if from_disk is not None else None
            logits = model(x, addr, mem_vec=mv)
            pred = torch.stack([logits[:, p - 1].argmax(-1) for p in vp], 1)
            true = torch.stack([y[:, p - 1] for p in vp], 1)
            cor_dig += (pred == true).float().sum().item(); cor_all += (pred == true).all(1).float().sum().item(); tot += 32
    model.train(); return cor_all / tot, cor_dig / (tot * 4)


if __name__ == "__main__":
    print(f"#18 retriever exact: M={M} faits, SEQ={SEQ}, table={N_ADDR} adresses", flush=True)
    torch.manual_seed(42); model = RetrievalMem()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t0 = time.time()
    for step in range(STEPS):
        x, y, idx, vp = gen_batch(32)
        loss = F.cross_entropy(model(x, addr_of(x)).reshape(-1, VS), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 400 == 0:
            a, dg = evaluate(model, 200); print(f"  step {step} loss={loss.item():.3f} exact={a*100:.1f}% chiffre={dg*100:.1f}% {time.time()-t0:.0f}s", flush=True)
    a_ram, dg_ram = evaluate(model)
    # DUMP mémoire sur DISQUE (memmap) puis éval avec lookup exact depuis le disque
    path = os.path.join(ROOT, "resultats", "mem18.dat")
    mm = np.memmap(path, dtype='float32', mode='w+', shape=(N_ADDR, DIM)); mm[:] = model.mem.weight.detach().numpy(); mm.flush()
    a_disk, dg_disk = evaluate(model, from_disk=mm)
    res = {"model": "#18 retriever exact", "acc_exact_RAM": a_ram, "acc_digit_RAM": dg_ram,
           "acc_exact_DISQUE": a_disk, "acc_digit_DISQUE": dg_disk,
           "v17_digit": 0.121, "hasard": 0.10,
           "octets_par_lookup": DIM * 4, "acces_disque_par_sequence": 1,
           "acces_disque_par_token": 1.0 / SEQ, "memoire_disque_Mo": os.path.getsize(path) / 1e6}
    json.dump(res, open(os.path.join(ROOT, "resultats", "factmix_18.json"), "w"), indent=2)
    print(f"\n=== #18: chiffre RAM={dg_ram*100:.1f}%  DISQUE={dg_disk*100:.1f}%  (fait_exact disque={a_disk*100:.1f}%) ===")
    print(f"vs #17 appris=12.1%, hasard=10%.  1 accès disque/séquence ({DIM*4} o), table {os.path.getsize(path)/1e6:.1f} Mo sur disque", flush=True)
