"""FactMix — le test de VALEUR de #17. Faits @eXXXX -> =vYYYY séparés par un remplissage
> champ réceptif local (192). #14 pur ne peut PAS retrouver le fait ; #17 (mémoire paginée) doit.
Compare accuracy factuelle #17 vs #14, + accès HDD/token. Sortie: resultats/factmix.json
"""
import torch, torch.nn.functional as F, os, json, time, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
torch.set_num_threads(2)
from candidats_actifs.block_local import BlockLocalTransformer
from candidats_actifs.block_local_pagemem import BlockLocalPageMemTransformer

M = 1000                 # nb de faits (associations à mémoriser)
SEQ = 256                # distance entité->valeur ~242 > RF local 192
STEPS, BATCH = 3000, 32
# vocab : 0-9 (chiffres), a-z (remplissage/langage), '@','=',' ' -> 39 chars
CHARS = list("0123456789abcdefghijklmnopqrstuvwxyz@= ")
STOI = {c: i for i, c in enumerate(CHARS)}; VS = len(CHARS)
DIG = [STOI[str(d)] for d in range(10)]
torch.manual_seed(123)
ENT = torch.randint(0, 10, (M, 4))         # 4 chiffres d'entité
VAL = torch.randint(0, 10, (M, 4))         # 4 chiffres de valeur (mapping fixe)
FILL = SEQ - 14                            # "@e"+4 +" " + fill + " =v"+4


def gen_batch(bs):
    idx = torch.randint(0, M, (bs,))
    seq = torch.full((bs, SEQ), STOI[' '], dtype=torch.long)
    seq[:, 0] = STOI['@']; seq[:, 1] = STOI['a'] + (ord('e') - ord('a'))  # '@e'
    for j in range(4): seq[:, 2 + j] = ENT[idx, j] + STOI['0']
    # remplissage aléatoire (lettres a-z) positions 7 .. 7+FILL
    fill = torch.randint(STOI['a'], STOI['z'] + 1, (bs, FILL))
    seq[:, 7:7 + FILL] = fill
    vpos = 7 + FILL + 1
    seq[:, vpos] = STOI['='] ; seq[:, vpos + 1] = STOI['v'] if 'v' in STOI else STOI['a'] + (ord('v') - ord('a'))
    for j in range(4): seq[:, vpos + 2 + j] = VAL[idx, j] + STOI['0']
    tgt = torch.zeros_like(seq); tgt[:, :-1] = seq[:, 1:]
    val_pos = [vpos + 2 + j for j in range(4)]     # positions des chiffres de valeur
    return seq, tgt, idx, val_pos


def eval_fact(model, n=500):
    model.eval(); cor_all = 0; cor_dig = 0; tot = 0
    with torch.no_grad():
        for _ in range(n // BATCH + 1):
            x, y, idx, vp = gen_batch(BATCH)
            logits = model(x)
            pred = torch.stack([logits[:, p - 1].argmax(-1) for p in vp], 1)  # (B,4) chiffres prédits
            true = torch.stack([y[:, p - 1] for p in vp], 1)
            cor_dig += (pred == true).float().sum().item()
            cor_all += (pred == true).all(1).float().sum().item()
            tot += x.size(0)
    model.train(); return cor_all / tot, cor_dig / (tot * 4)


def train(name, model, alpha=None):
    if alpha is not None and hasattr(model, "mem_block"):
        model.mem_block().alpha.data.fill_(alpha)
    if hasattr(model, "mem_block"):
        memp = {id(p) for n_, p in model.named_parameters() if any(k in n_ for k in ("K_pages", "K_slots", "V_slots"))}
        g = [{"params": [p for p in model.parameters() if id(p) not in memp], "weight_decay": 0.01},
             {"params": [p for p in model.parameters() if id(p) in memp], "weight_decay": 0.0}]
        aux = lambda: model.aux_loss()
    else:
        g = [{"params": model.parameters(), "weight_decay": 0.01}]; aux = lambda: 0.0
    opt = torch.optim.AdamW(g, lr=5e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    model.train(); t0 = time.time()
    for step in range(STEPS):
        x, y, idx, vp = gen_batch(BATCH)
        loss = F.cross_entropy(model(x).reshape(-1, VS), y.reshape(-1)) + aux()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0:
            a, dg = eval_fact(model, 200)
            print(f"  [{name}] step {step} loss={loss.item():.3f} fait_exact={a*100:.1f}% chiffre={dg*100:.1f}% {time.time()-t0:.0f}s", flush=True)
    a, dg = eval_fact(model)
    print(f"==> {name} FINAL: fait_exact(4/4)={a*100:.1f}% chiffre={dg*100:.1f}%", flush=True)
    return a, dg


if __name__ == "__main__":
    print(f"FactMix: M={M} faits, SEQ={SEQ}, distance entité→valeur={FILL+4}>RF192, vocab={VS}", flush=True)
    torch.manual_seed(42)
    m17 = BlockLocalPageMemTransformer(vocab_size=VS, dim=64, heads=2, layers=3, ffn_dim=256,
                                       max_seq_len=SEQ, block_size=64, n_pages=1024, slots_per_page=32, p=2)
    a17, d17 = train("#17", m17, alpha=-2.0)
    mem = m17.mem_block().mem
    with torch.no_grad(): m17(gen_batch(1)[0])
    st = mem.access_stats(SEQ, 64)
    torch.manual_seed(42)
    m14 = BlockLocalTransformer(vocab_size=VS, dim=64, heads=2, layers=3, ffn_dim=256, max_seq_len=SEQ, block_size=64)
    a14, d14 = train("#14", m14)
    res = {"M": M, "SEQ": SEQ, "distance": FILL + 4, "rf_local": 192,
           "acc17_exact": a17, "acc17_digit": d17, "acc14_exact": a14, "acc14_digit": d14,
           "acces_hdd_par_token_17": st["acces_par_token"], "gate_ouvert_17": float(torch.sigmoid(m17.mem_block().alpha).max()),
           "hasard_digit": 0.1}
    json.dump(res, open(os.path.join(ROOT, "resultats", "factmix.json"), "w"), indent=2)
    print(f"\n=== FactMix ===\n#17: fait {a17*100:.1f}% / chiffre {d17*100:.1f}%  |  #14: fait {a14*100:.1f}% / chiffre {d14*100:.1f}%  (hasard chiffre=10%)")
    print(f"#17 accès HDD/token={st['acces_par_token']:.4f}, gate ouvert à {res['gate_ouvert_17']:.3f}", flush=True)
