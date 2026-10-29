import torch, torch.nn.functional as F, math, sys, os, urllib.request
sys.path.append('.')
from baseline.transformer import BaselineTransformer
from candidats_actifs.retnet_chunked import RetNetChunked

DIM,HEADS,LAYERS,FFN=64,2,3,256; BATCH=32; STEPS=3000; WARMUP=100; CHUNK=64; SEQ=128

dpath='data/tinyshakespeare.txt'
if not os.path.exists(dpath):
    os.makedirs('data',exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',dpath)
with open(dpath) as f: txt=f.read()
chars=sorted(set(txt)); vs=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
n=int(0.9*len(data))
tr,te=data[:n],data[n:]

def batch(d,bs,sl=SEQ):
    ix=torch.randint(len(d)-sl,(bs,))
    return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])

for name,Klass,kw in [
    ('Baseline',BaselineTransformer,dict(vocab_size=vs,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096)),
    ('RChunk',RetNetChunked,dict(vocab_size=vs,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096,chunk_size=CHUNK)),
]:
    for seed in [42]:
        print(f'\n=== {name} seed={seed} ===')
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        model=Klass(**kw)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
        model.train()
        for s in range(STEPS):
            x,y=batch(tr,BATCH)
            if s<WARMUP: opt.param_groups[0]['lr']=3e-4*s/WARMUP
            L=model(x); loss=F.cross_entropy(L.reshape(-1,vs),y.reshape(-1))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            if s>=WARMUP: sched.step()
        model.eval()
        tl=0.0
        with torch.no_grad():
            for _ in range(100):
                x,y=batch(te,32)
                tl+=F.cross_entropy(model(x).reshape(-1,vs),y.reshape(-1)).item()
        print(f'>>> PPL={math.exp(tl/100):.4f}')
