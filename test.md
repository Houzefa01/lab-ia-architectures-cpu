# Tests obligatoires avant validation d'un modèle

## 1. Shapes
- Entrée: `(2, 48)` → sortie: `(2, 48, vocab_size)`
- Vérifier que `out.shape == (B, T, vocab_size)`

## 2. Gradients
- `out.mean().backward()`
- Vérifier `p.grad is not None` ET `not p.grad.isnan().any()` pour TOUS les paramètres
- Interrompre si échec : architecture mal construite (gradients qui ne propagent pas)

## 3. Audit causal OBLIGATOIRE
- Générer `seq1` (aléatoire), `seq2 = seq1.clone(); seq2[:, T//2:] = random`
- Mode eval, `with torch.no_grad(): o1=model(seq1); o2=model(seq2)`
- `diff = (o1[:, :T//2] - o2[:, :T//2]).abs().max().item()`
- `assert diff < 1e-6` — sinon fuite causale (le modèle regarde le futur)
- Interrompre immédiatement si échec — toute fuite causale invalide le modèle

## 4. Copy task (next-token prediction)
- Gap=186, seq_len=192, 10 tokens (0=SOS, 1=END, 2-9=données)
- Targets shiftés: `tgt[:-1] = seq[1:]; tgt[-1] = 0`
- PONDÉRATION ×100 sur les 2 positions de recall (3+gap, 3+gap+1)
  - `loss_r = loss_all[:, ep:ep+2].mean() * 100.0`
  - `loss_o = loss_all[:, :ep].mean() + loss_all[:, ep+2:].mean()`
  - `total = loss_o + loss_r`
- Optimiseur: AdamW lr=1e-4, weight_decay=0.01, CosineAnnealingLR
- 3000 steps, batch=32, 8000 train / 2000 test
- Seuil validation: **≥95%** (2 positions recall correctes simultanément)
- Si >95% → passer au LM. Sinon → ARCHITECTURE INVALIDÉE pour la mémoire associative

## 5. LM perplexity (TinyShakespeare)
- Vocab complet (65 caractères), dim=64, heads=2, layers=3, ffn=256
- Optimiseur: AdamW lr=3e-4, CosineAnnealingLR, warmup=100 steps
- 3000 steps, batch=32, seq_len=128
- Évaluation: moyenne cross-entropy sur 100 batches test → `ppl = exp(mean_ce)`
- Seuil validation: **perplexité ≤ 7.64** (perdre moins de 0.05 ppl face au baseline 7.59)
- Si ≤7.64 → candidat prometteur. Sinon → LM dégradé

## 6. Benchmark vitesse
- Batch=32, 7 longueurs: 48, 128, 256, 512, 1024, 2048, 4096
- 5 warmup runs + 5 mesures, médiane
- Interrompre si un run dépasse 3 min CPU
- Seuil validation: **plus rapide que le baseline à T≥1024**

## 7. Critères de victoire (dans l'ordre)
1. ✅ Copy task ≥ 95%
2. ✅ LM perplexity ≤ baseline (7.59) ou très proche (≤7.64)
3. ✅ Plus rapide que le baseline à T≥1024
4. ✅ Causal (vérifié section 3)
5. ✅ Gradients OK

## 8. Échecs connus (ne pas retester)
| Architecture | Raison échec |
|---|---|
| Élémentwise (KVMemory v1) | 1.8% copy — ne peut PAS faire de recall associatif |
| GLA récurrent/chunké | Boucle Python rédhibitoire sur CPU |
| SSM/Mamba | 1.9% copy — état N=16 insuffisant |
| Convs dilatées | 1.7% copy — translation-équivariant |
| RetNet multi-gamma | 87.6% copy — gamma court tue le recall |
| Sliding Window | 100% copy mais pas plus rapide sur CPU |
| RetNet chunké | 96.4% copy mais 8.07 ppl (perd face à 7.59) |
