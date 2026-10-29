# Objectif global : Architecture IA battant le Transformer sur CPU

## Mission
Trouver une architecture de réseau de neurones alternative au Transformer causal (softmax attention) qui le bat simultanément sur :
1. **Copy task** : rappel associatif longue distance (gap=186, 2 tokens parmi 10)
2. **LM perplexité** : modélisation de langage (TinyShakespeare, ~65 caractères)
3. **Vitesse CPU** : inférence plus rapide qu'un Transformer de référence
4. **Causalité stricte** : aucune fuite d'information future
5. **Gradients corrects** : backpropagation complète et stable

## 🎯 Zone visée : CONTEXTE LONG (priorité)
La cible prioritaire est le régime **contexte long** : **T ≥ 1024, jusqu'à T = 4096 et au-delà**. C'est là que :
- le Transformer softmax devient O(T²) : il sature la RAM (**OOM dès T=4096** sur cette machine) et sa latence explose (10.6 s à T=2048) ;
- un avantage vitesse/mémoire se traduit **directement** en faisabilité d'un grand modèle sur CPU (objectif final ~70B sans GPU).

À **court contexte (T ≤ 128)**, l'attention T×T est déjà triviale : **ce n'est PAS la zone d'intérêt**. Un candidat peut y être plus lent sans être disqualifié, tant qu'il **gagne à contexte long**. (C'est le cas du vainqueur #14 Block-Local : plus lent à T≤128, mais 4.43× à T=1024, 10.85× à T=2048, et seul à tourner à T=4096.)

## Critères étendus (objectif final : IA ~70B fonctionnant sans GPU)
En plus des 5 critères ci-dessus, pour qu'une architecture soit une brique viable vers un grand modèle CPU :

6. **Mémoire bornée** : empreinte **sous-quadratique** (idéalement O(T·C) par couche, voire état O(1) en longueur), qui tient en RAM à grand T. *Sine qua non* pour héberger un 70B sans GPU.
7. **Débit CPU à contexte long** : tokens/s exploitable à T≥1024, via matmuls MKL (`bmm`/`sgemm`) — **jamais** de boucle Python sur T, ni d'ops hors-MKL (unfold/cumsum/einsum).
8. **Généralisation de longueur** : rester causal et cohérent **au-delà de la longueur d'entraînement** (contexte extensible sans réentraînement).
9. **Robustesse à la quantization** : accuracy préservée en **int8/int4** (réduction RAM indispensable — un 70B en int4 ≈ 35 Go, hébergeable sur station CPU).
10. **Scaling favorable** : coût vitesse/mémoire croissant lentement avec dim/#params → **extrapolation viable vers 70B** (loi d'échelle mesurée, pas supposée).

## Méthodologie
1. Architecture candidate → codage (PyTorch, CPU, dim=64, heads=2, layers=3, ffn=256)
2. Validation unitaire : shapes, gradients, audit causal
3. Benchmark copy task (3000 steps, pondération ×100 recall, next-token)
4. Benchmark LM perplexité (3000 steps, CosineAnnealingLR, warmup 100)
5. Benchmark vitesse (7 longueurs: 48-4096, batch=32, 5 runs médiane)
6. Si tous les critères passent → validation multi-seed + Wilcoxon

## Contraintes permanentes
- CPU uniquement (pas de GPU)
- dim=64, heads=2, layers=3, ffn=256 (fixe pour comparaison)
- Next-token prediction (pas d'autoencodage)
- Gap=186, seq_len=192 pour la copy task

## Architectures testées (13)

| # | Architecture | Résultat |
|---|-------------|----------|
| 1 | KVMemory v1 (élémentwise cumsum) | Copy 1.8% — échec recall associatif |
| 2 | KVMemory v2 (D×D exact vectorisé) | LM < Baseline — bug autoencodage |
| 3 | KVMemory v3 (chunks C=64 non-chevauchement) | Copy 1.8% — pas de propagation inter-chunk |
| 4 | CDCN (convs dilatées WaveNet L=8) | Copy 1.7% — translation-équivariant |
| 5 | CDCN-Attn (7 convs + 1 attention) | Copy 17.5% — features trop transformées |
| 6 | RetNet récurrent (boucle Python) | Copy 97.9% — trop lent (boucle T) |
| 7 | RetNet chunké (γ=0.999, C=64) | Copy 96.4%, LM 8.07 — meilleur compromis mais LM perd |
| 8 | RetNet multi-gamma (0.9 + 0.999) | LM 7.53 bat Baseline ! mais Copy 87.6% |
| 9 | Sliding Window W=64 (softmax exact) | Copy 100%, LM 7.64 — ex æquo mais pas plus rapide |
| 10 | Mamba SSM sélectif (N=16) | Copy 1.9% — état N=16 insuffisant |
| 11 | GLA chunké (gate g_t = sigmoid) | 2-6× plus lent que Baseline (boucle Python) |
| 12 | Sparse Clustered K=4 (cumsum par cluster) | 17× plus lent que Baseline (K×cumsum) |
| 13 | Sliding Window Fast (unfold) | Pas testé (dépassé par échecs précédents) |

## Conclusion (révisée 2026-07-02)
**Un vainqueur a été trouvé : #14 Block-Local Attention** (voir RAPPORT_VICTOIRE.md). La conclusion négative du 2026-07-01 avait mal ciblé le blocage : ce n'était pas *softmax vs alternative* mais **MKL vs non-MKL**. Block-Local garde le **softmax exact** (copy 100 %, LM à parité) mais le calcule en **attention locale par blocs via `torch.bmm` (batched sgemm = MKL)** → **contexte long** gagné : 4.43× à T=1024, 10.85× à T=2048, fonctionne à T=4096 (baseline OOM), mémoire 2.3–4× moindre.

**Reste à faire vers l'objectif 70B sans GPU** (critères étendus 6–10) : quantization int8/int4, généralisation de longueur, et analyse de scaling dim/#params pour projeter le comportement à l'échelle 70B.

Date de clôture (négative) : 2026-07-01 · Date de percée : 2026-07-02
