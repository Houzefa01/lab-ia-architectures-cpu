"""Quick benchmark: CDCN vs Baseline (copy task + speed)"""
import torch, torch.nn.functional as F, math, sys, time, os, psutil
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline.transformer import BaselineTransformer
from candidats_actifs.cdcn import CDCN

DIM, HEADS, LAYERS, FFN = 64, 2, 3, 256
CDCN_LAYERS = 8  # champ 2^8=256 > gap 188
BATCH=32; STEPS=3000; GAP=186

def log(m): print(m, flush=True)

def gen(n,gap,seed=42):
    torch.manual_seed(seed); sl=3+gap+3
    inp,tgt=[],[]
    for _ in range(n):
        k1=int(torch.randint(2,9,(1,))[0]);k2=int(torch.randint(2,9,(1,))[0])
        s=torch.zeros(sl,dtype=torch.long)
        s[0]=0;s[1]=k1;s[2]=k2;s[3:3+gap]=torch.randint(2,9,(gap,))
        e=3+gap;s[e]=1;s[e+1]=k2;s[e+2]=k1
        t=torch.zeros(sl,dtype=torch.long);t[:-1]=s[1:];t[-1]=0
        inp.append(s);tgt.append(t)
    return torch.stack(inp),torch.stack(tgt),sl

tx,ty,sl=gen(8000,GAP,42); tsx,tsy,_=gen(2000,GAP,43)
ep=3+GAP

for name,Klass,kw in [
    ('Baseline',BaselineTransformer,dict(vocab_size=10,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096)),
    ('CDCN',CDCN,dict(vocab_size=10,dim=DIM,layers=CDCN_LAYERS,ffn_dim=FFN,max_seq_len=4096)),
]:
    log(f"\n=== {name} ===")
    
    # Speed bench
    m=Klass(**kw)
    log("  Speed benchmark:")
    for L in [48,128,256,512,1024,2048,4096]:
        x=torch.randint(0,10,(1,L))
        with torch.no_grad():
            for _ in range(5): m(x)
        ts=[]
        for _ in range(10):
            t0=time.perf_counter()
            with torch.no_grad(): m(x)
            ts.append((time.perf_counter()-t0)*1000)
        log(f"    T={L:>5}: {sorted(ts)[5]:.2f}ms")
    
    # Copy task training
    log(f"  Copy task training ({STEPS} steps):")
    torch.manual_seed(42)
    model=Klass(**kw)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    load=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),BATCH,shuffle=True)
    model.train()
    for s,(x,y) in enumerate(load):
        if s>=STEPS:break
        L=model(x)
        loss_all=F.cross_entropy(L.reshape(-1,10),y.reshape(-1),reduction='none').reshape(y.shape)
        loss_r=loss_all[:,ep:ep+2].mean()*100.0
        loss_o=loss_all[:,:ep].mean()+loss_all[:,ep+2:].mean()
        total=loss_o+loss_r
        opt.zero_grad();total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();sched.step()
        if s%500==0:log(f"    step {s}: loss={total.item():.3f}")
    
    model.eval()
    with torch.no_grad():
        p=model(tsx)[:,ep:ep+2].argmax(-1)
        t=tsy[:,ep:ep+2]
        cor=(p==t).all(1).sum().item()
    log(f"  >>> Copy Accuracy: {cor/2000*100:.1f}%")