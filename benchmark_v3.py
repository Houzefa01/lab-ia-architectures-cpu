"""Quick benchmark: v3 local attention"""
import sys, os, math, time, psutil, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.kv_memory_v3 import KVMemoryV3

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
SEEDS = [42, 43]; STEPS = 300; BATCH=32
def log(m): print(m, flush=True)

def gen(n, seed=42):
    torch.manual_seed(seed); d=[]
    for _ in range(n):
        k1,k2=[int(torch.randint(2,9,(1,))[0]) for _ in range(2)]
        s=torch.zeros(192,dtype=torch.long)
        s[0]=0;s[1]=k1;s[2]=k2;s[3:3+186]=torch.randint(2,9,(186,))
        ep=3+186;s[ep]=1;s[ep+1]=k2;s[ep+2]=k1;d.append(s)
    return torch.stack(d), torch.stack(d)

def speed(m, ls=[48,128,256,512,1024,2048,4096], n=10):
    m.eval(); r={}
    for L in ls:
        x=torch.randint(0,10,(1,L))
        with torch.no_grad():
            for _ in range(5): m(x)
        ts=[]
        for _ in range(n):
            t0=time.perf_counter()
            with torch.no_grad(): m(x)
            ts.append((time.perf_counter()-t0)*1000)
        r[str(L)]=sorted(ts)[n//2]
    return r

for name,Klass,Cs in [('Baseline',BaselineTransformer,None), ('KVv3',KVMemoryV3,64)]:
    log(f"\n--- {name} ---")
    m=Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096,chunk_size=Cs) if Cs else Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096)
    sp=speed(m)
    for L,ms in sp.items(): log(f"  T={L:>5}: {ms:.2f}ms")
    accs=[]
    for se in SEEDS:
        torch.manual_seed(se); np.random.seed(se)
        model=Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096,chunk_size=Cs) if Cs else Klass(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096)
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
        model.eval()
        cor,tot=0,0
        with torch.no_grad():
            for x,y in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tsx,tsy),BATCH):
                p=model(x)[:,-2:].argmax(-1)
                cor+=(p==y[:,-2:]).all(1).sum().item();tot+=x.size(0)
        a=cor/tot*100;accs.append(a);log(f"  seed={se}: acc={a:.1f}%")
    log(f"  Mean: {np.mean(accs):.1f}%")