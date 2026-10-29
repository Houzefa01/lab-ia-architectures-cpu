"""Vérification v3 : est-ce que la copy task est vraiment résolue ? + LM perplexity"""
import sys, math, os, time, psutil, urllib.request, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory_v3 import KVMemoryV3

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
STEPS = 300; BATCH = 32
def log(m): print(m, flush=True)

def gen(n, seed=42):
    torch.manual_seed(seed); d=[]
    for _ in range(n):
        k1,k2=[int(torch.randint(2,9,(1,))[0]) for _ in range(2)]
        s=torch.zeros(192,dtype=torch.long)
        s[0]=0;s[1]=k1;s[2]=k2;s[3:3+186]=torch.randint(2,9,(186,))
        ep=3+186;s[ep]=1;s[ep+1]=k2;s[ep+2]=k1;d.append(s)
    return torch.stack(d), torch.stack(d)

# Train both models + check individual predictions
for name,Klass,Cs in [('Baseline',BaselineTransformer,None), ('KVv3',KVMemoryV3,64)]:
    log(f"\n=== {name} ===")
    torch.manual_seed(42)
    model=Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096) if Cs is None else Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096,chunk_size=Cs)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    def lr_fn(s):
        if s<100: return s/100
        return 0.5*(1+math.cos(math.pi*(s-100)/(STEPS-100)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    tx,ty=gen(8000,42); tsx,tsy=gen(2000,43)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),BATCH,shuffle=True)
    model.train()
    for step,(x,y) in enumerate(loader):
        if step>=STEPS: break
        l=F.cross_entropy(model(x).view(-1,10),y.view(-1))
        opt.zero_grad();l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()

    # Vérification individuelle
    model.eval()
    with torch.no_grad():
        preds = model(tsx[:10, :])  # first 10 test samples
        pred_tokens = preds[:, -2:, :].argmax(-1)
        true_tokens = tsy[:10, -2:]
        for i in range(10):
            p = pred_tokens[i].tolist()
            t = true_tokens[i].tolist()
            ok = "OK" if p==t else "FAIL"
            log(f"  Sample {i}: pred={p} true={t} {ok}")

    # Accuracy
    loader2=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tsx,tsy),BATCH)
    cor,tot=0,0
    with torch.no_grad():
        for x,y in loader2:
            p=model(x)[:,-2:].argmax(-1)
            cor+=(p==y[:,-2:]).all(1).sum().item();tot+=x.size(0)
    log(f"  Accuracy: {cor/tot*100:.1f}% ({cor}/{tot})")