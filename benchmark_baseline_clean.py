import torch, torch.nn.functional as F, math, sys, os, urllib.request, json, time
sys.path.append('.')
from baseline.transformer import BaselineTransformer

DIM,HEADS,LAYERS,FFN=64,2,3,256
SEQ=128; BATCH=32; STEPS=3000; WARMUP=100; LR=3e-4

dpath='data/tinyshakespeare.txt'
if not os.path.exists(dpath):
    os.makedirs('data',exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',dpath)
with open(dpath) as f: txt=f.read()
chars=sorted(set(txt)); vs=len(chars); stoi={c:i for i,c in enumerate(chars)}
data=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
n=int(0.9*len(data))
train,test=data[:n],data[n:]

def batch(d,bs,sl=SEQ):
    ix=torch.randint(len(d)-sl,(bs,))
    return torch.stack([d[i:i+sl] for i in ix]), torch.stack([d[i+1:i+sl+1] for i in ix])

seeds=[42,43,44]
lm_results={}
for seed in seeds:
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model=BaselineTransformer(vocab_size=vs,dim=DIM,heads=HEADS,layers=LAYERS,ffn_dim=FFN,max_seq_len=4096)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    model.train()
    for s in range(STEPS):
        x,y=batch(train,BATCH,SEQ)
        if s<WARMUP: opt.param_groups[0]['lr']=LR*s/WARMUP
        L=model(x); loss=F.cross_entropy(L.reshape(-1,vs),y.reshape(-1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if s>=WARMUP: sched.step()
    model.eval()
    losses=[]
    with torch.no_grad():
        for _ in range(100):
            x,y=batch(test,BATCH,SEQ)
            losses.append(F.cross_entropy(model(x).reshape(-1,vs),y.reshape(-1)).item())
    avg_loss=sum(losses)/len(losses)
    ppl=math.exp(avg_loss)
    lm_results[seed]=ppl
    print(f'seed={seed}: loss={avg_loss:.4f} ppl={ppl:.4f}')

all_ppls=[v for v in lm_results.values()]
print(f'\n=== RÉSULTATS ===')
print(f'Perplexités: {[f"{p:.4f}" for p in all_ppls]}')
print(f'Moyenne: {sum(all_ppls)/len(all_ppls):.4f}')
print(f'Écart-type: {torch.tensor(all_ppls).std().item():.4f}')
