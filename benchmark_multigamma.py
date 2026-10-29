import torch, torch.nn.functional as F, math, sys, os, time, urllib.request
sys.path.append('.')
from baseline.transformer import BaselineTransformer
from candidats_actifs.retnet_multigamma import RetNetMultiGamma

DIM,HEADS,LAYERS,FFN=64,2,3,256; BATCH=32; STEPS=3000; WARMUP=100; CHUNK=64
GAP=186; SL=192
BASE_KW = dict(dim=DIM, heads=HEADS, layers=LAYERS, ffn_dim=FFN, max_seq_len=4096)
MULTI_KW = {**BASE_KW, 'chunk_size': CHUNK}

def gen_copy(n, gap=GAP, seed=42):
    torch.manual_seed(seed); sl=3+gap+3
    inp,tgt=[],[]
    for _ in range(n):
        k1=int(torch.randint(2,9,(1,))[0]);k2=int(torch.randint(2,9,(1,))[0])
        s=torch.zeros(sl,dtype=torch.long)
        s[0]=0;s[1]=k1;s[2]=k2;s[3:3+gap]=torch.randint(2,9,(gap,))
        e=3+gap;s[e]=1;s[e+1]=k2;s[e+2]=k1
        t=torch.zeros(sl,dtype=torch.long);t[:-1]=s[1:];t[-1]=0
        inp.append(s);tgt.append(t)
    return torch.stack(inp),torch.stack(tgt)
tx,ty=gen_copy(8000); tsx,tsy=gen_copy(2000,seed=43); ep=3+GAP

dpath='data/tinyshakespeare.txt'
if not os.path.exists(dpath):
    os.makedirs('data',exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',dpath)
with open(dpath) as f: txt=f.read()
chars=sorted(set(txt)); vs=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
n=int(0.9*len(data)); tr,te=data[:n],data[n:]

def lm_batch(d,bs,sl=128):
    ix=torch.randint(len(d)-sl,(bs,))
    return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])

for name,Klass,base_kw in [
    ('Baseline',BaselineTransformer,BASE_KW),
    ('MultiG',RetNetMultiGamma,MULTI_KW),
]:
    copy_kw = {**base_kw, 'vocab_size': 10}
    lm_kw = {**base_kw, 'vocab_size': vs}
    
    print(f'\n{"="*60}\n{name}\n{"="*60}')

    # SPEED
    print(f'\n--- SPEED ---')
    m=Klass(vocab_size=10,**base_kw)
    for L in [48,128,256,512,1024,2048,4096]:
        x=torch.randint(0,10,(1,L))
        with torch.no_grad():
            for _ in range(5): m(x)
        ts=[]
        for _ in range(10):
            t0=time.perf_counter()
            with torch.no_grad(): m(x)
            ts.append((time.perf_counter()-t0)*1000)
        print(f'T={L:>5}: {sorted(ts)[5]:.2f}ms')

    # COPY
    print(f'\n--- COPY ({STEPS} steps) ---')
    torch.manual_seed(42)
    model=Klass(**copy_kw)
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
        if s%1000==0:print(f'  step {s}: loss={total.item():.3f}')
    model.eval()
    with torch.no_grad():
        p=model(tsx)[:,ep:ep+2].argmax(-1)
        cor=(p==tsy[:,ep:ep+2]).all(1).sum().item()
    print(f'>>> Copy Acc: {cor/2000*100:.1f}%')

    # LM
    print(f'\n--- LM ({STEPS} steps) ---')
    torch.manual_seed(42)
    model_lm=Klass(**lm_kw)
    opt_lm=torch.optim.AdamW(model_lm.parameters(),lr=3e-4,weight_decay=0.01)
    sched_lm=torch.optim.lr_scheduler.CosineAnnealingLR(opt_lm,STEPS)
    model_lm.train()
    for s in range(STEPS):
        x,y=lm_batch(tr,BATCH)
        if s<WARMUP: opt_lm.param_groups[0]['lr']=3e-4*s/WARMUP
        L=model_lm(x); loss=F.cross_entropy(L.reshape(-1,vs),y.reshape(-1))
        opt_lm.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model_lm.parameters(),1.0)
        opt_lm.step()
        if s>=WARMUP: sched_lm.step()
    model_lm.eval()
    tl=0.0
    with torch.no_grad():
        for _ in range(100):
            x,y=lm_batch(te,32)
            tl+=F.cross_entropy(model_lm(x).reshape(-1,vs),y.reshape(-1)).item()
    print(f'>>> LM PPL: {math.exp(tl/100):.4f}')
