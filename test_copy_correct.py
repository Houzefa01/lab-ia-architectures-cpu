"""Test copie correcte (next-token prediction, targets shiftés)"""
import torch, torch.nn.functional as F, math, sys
sys.path.append('.')
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory_v3 import KVMemoryV3

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
STEPS = 3000; BATCH = 32

def log(m): print(m, flush=True)

def gen_copy(num, seed=42, seq_len=192, gap=186):
    """Génère séquences + targets shiftés (next-token prediction)"""
    torch.manual_seed(seed)
    inputs, targets = [], []
    for _ in range(num):
        k1 = int(torch.randint(2, 9, (1,))[0])
        k2 = int(torch.randint(2, 9, (1,))[0])
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0] = 0       # SOS
        seq[1] = k1
        seq[2] = k2
        seq[3:3+gap] = torch.randint(2, 9, (gap,))
        end = 3 + gap
        seq[end] = 1     # END
        seq[end+1] = k2  # recall 1
        seq[end+2] = k1  # recall 2
        
        # Next-token prediction: target[t] = input[t+1]
        tgt = torch.zeros(seq_len, dtype=torch.long)
        tgt[:-1] = seq[1:]  # shifted by 1
        tgt[-1] = 0          # dernier token prédit rien (ignoré dans la loss)
        
        inputs.append(seq)
        targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets)

for name, Klass, Cs in [('Baseline', BaselineTransformer, None), ('KVv3', KVMemoryV3, 64)]:
    log(f"\n=== {name} ===")
    torch.manual_seed(42)
    kw = dict(vocab_size=10, dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
    if Cs: kw['chunk_size'] = Cs
    model = Klass(**kw)
    
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    def lr_fn(s):
        if s < 100: return s/100
        return 0.5*(1+math.cos(math.pi*(s-100)/(STEPS-100)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    train_x, train_y = gen_copy(8000, 42)
    test_x, test_y = gen_copy(2000, 43)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), BATCH, shuffle=True)
    
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= STEPS: break
        logits = model(x)
        # Next-token loss: prédire y[t] à partir de x[t]
        loss = F.cross_entropy(logits.view(-1, 10), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    
    model.eval()
    # Mesure de l'accuracy sur les positions de recall (END doit prédire k2, k2 doit prédire k1)
    end_pos = 3 + 186  # position du token END
    # Output[end_pos] doit prédire target[end_pos] = input[end_pos+1] = k2
    # Output[end_pos+1] doit prédire target[end_pos+1] = input[end_pos+2] = k1
    cor, tot = 0, 0
    with torch.no_grad():
        preds = model(test_x)
        # Accuracy sur les 2 positions de recall
        p = preds[:, end_pos:end_pos+2, :].argmax(-1)
        t = test_y[:, end_pos:end_pos+2]
        cor = (p == t).all(dim=1).sum().item()
        tot = test_x.size(0)
    
    log(f"Accuracy recall: {cor/tot*100:.1f}% ({cor}/{tot})")
    
    # Vérification individuelle
    pred_tokens = p[:10]
    true_tokens = t[:10]
    for i in range(10):
        p_t = pred_tokens[i].tolist()
        t_t = true_tokens[i].tolist()
        ok = "OK" if p_t==t_t else "FAIL"
        log(f"  {i}: pred={p_t} true={t_t} {ok}")