# MÉMOIRE CENTRALE — MISSION AUTONOME IA

---

## Section 1 — État de la mission

**Dernière mise à jour :** 2026-07-02
**Candidats testés :** 14
**Vainqueur trouvé :** OUI — **Block-Local Attention (candidat #14)** — voir RAPPORT_VICTOIRE.md
**Conclusion révisée :** La conclusion négative du 2026-07-01 tenait car TOUTES les alternatives testées quittaient les matmuls MKL (einsum/cumsum/unfold/boucle Python). Le candidat #14 lève le blocage en calculant une attention **locale par blocs en `torch.bmm` (batched sgemm, donc MKL)** : softmax exact conservé → copy 100% + LM à parité exacte avec le Transformer, MAIS O(T·C·d) → **4.43× plus rapide à T=1024, 10.85× à T=2048, et fonctionne à T=4096 là où le Transformer est en OOM**, avec 2.3–4× moins de mémoire. Nuance d'honnêteté : le LM est en **parité** (pas battu — normal, block-local≡attention pleine à T=128), et block-local est **plus lent à T≤128** (petit contexte). Le gain est vitesse+mémoire à **contexte long** — précisément le régime utile pour un grand modèle CPU.
**Note incohérence résolue :** La perplexité 15.02 de benchmark_v2.json venait de 300 steps d'entraînement. Le vrai Transformer à maturité (3000 steps) fait 7.59 ppl.

---

## Section 2 — Tableau cumulatif (jamais effacé, jamais résumé)

| # | Nom | Mécanisme (1 phrase) | Précision | Vitesse | Causal | Verdict | Raison échec |
|---|-----|---------------------|-----------|---------|--------|---------|--------------|
| 1 | KVMemory v1 | Attention linéaire élément-par-élément cumsum O(T·D) | Copy: 100% (bug autoencodage), LM: 15.2ppl (@300 steps) | Gagne à T≥512 (15× 2048, 11× 4096) | ✅ | PROMETTEUR | Bug: entraînement en autoencodage pas next-token ; LM 15.2 @300 steps (Baseline @300 steps = 15.0) |
| 2 | KVMemory v2 | Attention linéaire D×D exact O(T·D²) vectorisée | Copy: 100% (bug autoencodage), LM: 15.13 vs 15.02 (@300 steps) | Gagne à T≥2048 (2.3× 4096) | ✅ | ÉCHOUÉ | Précision LM inférieure @300 steps ; O(T·D²) pas compétitif à T<2048 |
| 3 | KVMemory v3 | Attention locale en chunks C=64 (sliding window) | Copy: 1.8% (next-token correct) — pas de recopie | Gagne à T≥256 (14× 4096) | ✅ | ÉCHOUÉ | Chunks non-chevauchement : propagation d'info impossible entre chunks |
| 4 | CDCN | Convs causales dilatées (WaveNet, L=8, champ 255) | Copy: 1.7% (next-token) | Gagne à T≥512 (18× 4096) | ✅ | ÉCHOUÉ | Convs dilatées ne peuvent pas faire de rappel associatif (positions fixes) |
| 5 | CDCN-Attn | 7 convs dilatées + 1 couche attention causale | Copy: 17.5% (next-token) | Gagne à T≥1024 (2.6× 4096) | ✅ | ÉCHOUÉ | Attention seule insuffisante : features trop transformées par les convs |
| 6 | RetNet récurrent | Retention causale (QK exact + decay gamma) boucle Python | Copy: 97.9% (next-token) | Pire que Baseline à TOUTES les longueurs | ✅ | ÉCHOUÉ | Boucle Python rédhibitoire pour la vitesse |
| 7 | RetNet chunké | Retention chunkée (intra-chunk decay + cross-chunk S) | Copy: 96.4%, LM: 8.07ppl (perd vs 7.59) | Gagne à T≥1024 (7.4× 4096) | ✅ | ÉCHOUÉ | Précision LM inférieure : le decay gamma^{t-tau} est différent du softmax |
| 8 | RetNet multi-gamma (0.9, 0.999) | Retention avec gamma court/long par tête | Copy: 87.6%, LM: 7.53ppl (bat 7.64) | Gagne à T≥1024 (6.1× 4096) | ✅ | ÉCHOUÉ | Le gamma=0.9 améliore LM mais tue la copy (87.6%) |
| 9 | Sliding Window (W=64) | Attention softmax locale avec fenêtre coulissante W=64 | Copy: 100%, LM: 7.64 (ex æquo) | Comparable au baseline | ✅ | ÉCHOUÉ | Précision égale mais vitesse pas meilleure ; banded matmul non optimisé sur CPU |
| 10 | Mamba SSM sélectif | SSM avec A,B,C,Δ dépendants de l'input (N=16), boucle récurrente | Copy: 1.9%, LM: 7.86ppl (perd vs 7.64) | Gagne à T≥2048 (1.8× 4096) | ✅ | ÉCHOUÉ | Le SSM (même sélectif) ne peut pas faire le rappel associatif — l'état cumulatif perd l'information des positions précises via les transformations linéaires |
| 11 | GLA chunké (gate content) | Gate g_t = sigmoid, S_t = g_t·S + k⊗v, boucle chunkée | Non testé (trop lent) | 2-6× pire Baseline | ✅ | ÉCHOUÉ | Boucle Python rédhibitoire ; récurrence séquentielle sans closed-form |
| 12 | Sparse Clustered K=4 | Routage vers 4 clusters, chaque S_k = cumsum(k⊗v) | Non testé (trop lent) | 17× pire Baseline | ✅ | ÉCHOUÉ | K×cumsum trop coûteux ; 17× Baseline même à T=48 |
| 14 | **Block-Local Attention** | **Softmax exact par blocs (C=64, contexte=bloc i-1++i) en `torch.bmm` batché (MKL)** | **Copy: 100%, LM: 7.77 = parité exacte baseline (delta +0.0006, mêmes conditions)** | **4.43× @1024, 10.85× @2048, tourne @4096 (baseline OOM) ; mém 2.3–4× moindre. Plus lent à T≤128** | ✅ | ✅ **VAINQUEUR** | Aucun — bat le Transformer sur vitesse+mémoire à contexte long, à parité d'accuracy. 1re archi à ne quitter ni le softmax exact ni les matmuls MKL |

---

## Section 3 — Leçons apprises

- "L'attention causale élément-par-élément perd le cross-feature mixing nécessaire à la modélisation de langage"
- "Le point de croisement Transformer vs attention linéaire D×D est à T≈2048 sur CPU — pas assez pour des séquences de taille moyenne"
- "L'attention locale en chunks non-chevauchement ne peut PAS propager d'information entre chunks, même à travers les connexions résiduelles"
- "La copy task avec next-token prediction et gap=188 est TRÈS dure : nécessite 3000+ steps avec pondération ×100 sur les positions de recall"
- "BUG DÉTECTÉ dans les premiers tests : l'entraînement utilisait l'autoencodage (predire input[t]) au lieu du next-token (predire input[t+1]), donnant 100% trivial"
- "Les convolutions dilatées (même avec champ réceptif 255) ne peuvent PAS faire de rappel associatif car elles sont translation-équivariantes"
- "Le RetNet (QK exact + decay gamma) résout la copy task à 96-98% mais perd sur LM perplexité car gamma^{t-tau} ≠ softmax"
- "Le point de croisement O(T²·d) vs O(T·d²) est repoussé à T≈2048 en pratique à cause de l'optimisation MKL des matmul sur CPU"
- "L'attention sliding window avec softmax exact atteint 100% copy task et perplexité identique au baseline — mais pas de gain vitesse car PyTorch ne fait pas de banded matmul sur CPU"
- "Le RetNet multi-gamma bat le baseline sur LM (7.53 vs 7.64) grâce à la tête gamma=0.9 qui focalise sur le contexte local, mais la même tête nuit à la copy task longue distance"
- "Le SSM sélectif (Mamba-like) échoue sur la copy task : l'état cumulatif vectoriel N=16 ne peut pas retenir l'information positionnelle précise nécessaire au rappel associatif sur 190 positions"
- "CONCLUSION (2026-07-01, révisée) : Aucun mécanisme alternatif testé jusqu'au #13 ne bat le Transformer. MAIS la vraie leçon était mal formulée : le problème n'était pas 'softmax vs alternative', c'était 'MKL vs non-MKL'. Le candidat #9 (Sliding Window softmax) avait DÉJÀ résolu l'accuracy (copy 100%, LM 7.64) ; son seul échec était la vitesse, car PyTorch matérialise le T×T ou utilise unfold/einsum (hors MKL)."
- "PERCÉE (#14, 2026-07-02) : garder le softmax exact MAIS le calculer en blocs locaux via `torch.bmm` (batched sgemm = MKL). L'attention locale par blocs (bloc i attend bloc i-1 ++ i, 2C clés) est O(T·C·d), sans unfold/cumsum/boucle Python. Champ réceptif 3 couches × 64 = 192 ≥ gap 186 → copy 100%. À T=128 c'est exactement l'attention causale pleine → LM identique. Résultat : bat le Transformer sur vitesse (4.43× @1024) + mémoire (2.3–4×) à contexte long, à parité d'accuracy."
- "La distinction clé qui a débloqué : candidat #13 (swin_fast) utilisait unfold+einsum (hors MKL, lent) ; #14 utilise reshape en blocs + bmm (MKL, rapide). Même idée d'attention locale, implémentation opposée en efficacité CPU."

## Section 4 — Pistes en attente — TOUTES TESTÉES — ÉCHEC

| Priorité | Piste | Résultat |
|----------|-------|----------|
| 1 | RetNet multi-gamma | LM 7.53 ✅ mais Copy 87.6% ❌ |
| 2 | Sliding Window W=64 | Ex æquo sur les deux tâches ❌ |
| 3 | Mamba SSM sélectif | Copy 1.9% ❌, LM 7.86 ❌ |
| 4 | Algèbre géométrique/Clifford | Non testée — probabilité de succès faible car mécanisme non-linéaire sans rappel associatif garanti |
| 5 | Neural ODE causal | Non testée — coût CPU prohibitif (résolution ODE à chaque pas) |

**Verdict final :** Les 5 priorités de Section 4 sont épuisées (testées ou annulées). Aucune architecture ne bat le Transformer sur tous les critères.

## Section 5 — Mécanismes interdits (déjà invalidés)

| Mécanisme | Raison de l'échec |
|-----------|------------------|
| RFN (Resonance Field Networks) | Précision ~12%, pas de lookup associatif |
| TCE (Temporal Compression) | Précision ~12%, compression trop agressive |
| PRP (Predictive Residual Pathways) | Précision ~15%, insuffisant |
| RSC (Recursive Sparse Columns) | Précision ~9%, sous le hasard |
| CGW (Competitive Global Workspace) | Précision ~9%, sous le hasard |
| PHE (Phase Harmonic Encoding) | Précision ~11%, métaphore ondes invalide |
| FDN (Fractal Depth Networks) | Bug causal via avg_pool1d, puis 14% corrigé |
| TTP (Tensor Train Propagation) | Précision ~12% |
| SFN (Sparse Fourier Networks) | Précision ~19% |
| IBE (Information Bottleneck) | Précision ~14% |
| TFN/Linear Attention | Triche non-causale; version causale plus lente |
| ESN (Echo State Networks) | Précision ~12%, réservoir incapable de rappel |
| CA (Dilated MLP/Automate) | Précision ~18% |
| SSM linéaire | Précision ~11% |
| VSA (Vector Symbolic) | Précision ~11% |
| FNet (FFT) | Fuite causale, non corrigeable proprement |
| MambaLite | Précision ~13% |
| Hyperbolique | Précision ~15%, nécessite dim≥128 |
| Rhizome résiduel | Instable, variance excessive |
| KVMemory v2 (D×D exact) | Précision LM < Baseline, vitesse O(T·D²) pas compétitive |
| KVMemory v3 (attention locale) | Chunks non-chevauchement : impossible de propager l'info longue distance |
| RetNet multi-gamma (0.9, 0.999) | LM bat baseline mais copy 87.6% : le gamma court tue le rappel longue distance |
| Sliding Window W=64 (softmax) | Précision ex æquo mais vitesse pas meilleure sur CPU (pas de banded matmul) |
| Mamba SSM sélectif | SSM (même sélectif) incapable de rappel associatif : l'état N=16 ne préserve pas l'info positionnelle |
| GLA chunké vectorisé | Boucle Python rédhibitoire sur CPU (2-6× plus lent que Baseline) |
| Sparse Clustered Attention K=4 | K×cumsum 17× plus lent que Baseline sur CPU |

---

## Section 6 — Toutes les pistes sont épuisées

### Résultat : 13 architectures testées, 0 vainqueur

Voir RAPPORT_FINAL_NEGATIF.md pour l'analyse complète.

**Causes fondamentales :**
1. MKL rend les matmuls $T \times T$ du Transformer si efficaces que le point de croisement avec $O(T \cdot d^2)$ est repoussé à $T > 4096$
2. Toute récurrence séquentielle (GLA, SSM, RetNet) nécessite une boucle Python qui tue la performance sur CPU
3. Les alternatives non-MKL (einsum, cumsum, broadcast, unfold) ne sont pas optimisées par PyTorch CPU

**Dernier test : Sparse Clustered Attention (SCA, K=4)**
- Validé : shapes ✅, gradients ✅, causalité ✅
- Vitesse : 17× plus lent que Baseline (408ms vs 25ms à T=48)
- Cause : $K$ cumsum indépendants × $T$ positions

---

## Section 7 — Arc #14 → #18 : efficacité CPU + mémoire-savoir (2026-07-02)

Nouvelle mission (élargie) : viser une IA **knowledge-augmented légère** pour matériel contraint
(CPU 2 cœurs, 6 Go RAM, HDD 500 Go, sans GPU) — objectif ultime « puissance ~70B en local ».

### Tableau cumulatif (suite, jamais effacé)

| # | Nom | Mécanisme (1 phrase) | Résultat mesuré | Verdict |
|---|-----|---------------------|-----------------|---------|
| 14 | **Block-Local** | Attention causale par blocs (bloc i-1 ⧺ i) en `torch.bmm` batché (MKL), O(T·C) | **Bat le Transformer : 4.43× @T=1024, 10.85× @2048, tourne @4096 (baseline OOM), mém 2.3-4× moindre ; copy 100% ; LM parité (7.77)** | ✅ **1er VAINQUEUR (efficacité)** |
| 15 | Block-Local-Global | + mémoire globale linéaire compressive (double portée) | Copy longue portée gap=400 : **92.8%** (vs 2% pour #14 local pur) mais non convergé (<95%) | ⚠️ demi-succès |
| 16 | PKM (mémoire disque) | Product-Key Memory, N slots, K=32 lus/token, valeurs sur memmap disque | **Recall direct 100%** de 2000 faits, 8 Ko/token, 262K slots sur disque | ✅ (en isolation) |
| 17 | Paged Memory (v1/v2) | Mémoire paginée hiérarchique, routage APPRIS, cache LRU | **Accès HDD ÷1365** (0.023/token), causal, MAIS rappel factuel = **hasard (12%)** | ❌ routage appris échoue |
| 18 | **Retriever exact** | Entité = adresse directe (clé exacte, façon RAG), lookup non appris sur disque | **Recall 95%** (91.5% fait complet) depuis le disque, **256 o/token, 1 accès/séquence** | ✅ **savoir end-to-end** |

### Leçons apprises (Arc #14-#18)

- "La conclusion négative du 2026-07-01 était mal ciblée : le clivage n'était pas *softmax vs alternative* mais *MKL vs non-MKL*. Block-Local (#14) garde le softmax exact MAIS le calcule en `bmm` batché (MKL) → 1er candidat à battre le Transformer sur vitesse+mémoire à contexte long, à parité d'accuracy."
- "Le SAVOIR se déporte sur disque à coût minime (256 o-8 Ko/token, extensible à 500 Go), MAIS seulement avec un routage EXACT/par-similarité (#18=95%), PAS un routage appris par gradient (#17=hasard : problème œuf-poule, gradient mort à travers la sélection discrète top-k). C'est pourquoi les vrais systèmes (RAG) utilisent un retriever à similarité, jamais un routage end-to-end."
- "SAVOIR ≠ INTELLIGENCE. La mémoire externe ajoute la connaissance (déportable sur HDD lent), pas le raisonnement. Le raisonnement = poids actifs lus à CHAQUE token → doit rester en RAM. Aucune mémoire-disque ne relève le plafond de raisonnement."
- "Le sur-échantillonnage d'une mémoire non-utile nuit : entraîner une grosse mémoire sur des données sans faits (TinyShakespeare) injecte du bruit et dégrade le LM (#16=10.56, #17=13 vs #14=7.77). La mémoire ne doit s'activer que sur les tokens factuels (gate conditionnel)."

### Verdict final — objectif « IA très intelligente sur 6 Go/CPU sans GPU »

**Objectif scindé en deux :**
- ✅ **Volet EFFICACITÉ + SAVOIR = ATTEINT** (briques réelles : #14 raisonneur efficace, #18 savoir sur disque 95%). Une IA *savante, spécialisable, efficace CPU* est faisable.
- ❌ **Volet INTELLIGENCE de niveau frontier (Claude/GPT) sous cette contrainte = IMPOSSIBLE** — bornes DURES, non contournables par l'architecture :
  1. Plancher de capacité (Shannon) : l'intelligence = information ⇒ minimum de params incompressible.
  2. Plancher de calcul (complexité des circuits, Transformer profondeur fixe ≈ TC⁰) : raisonnement profond ⇒ profondeur/récurrence.
  3. RAM des poids : générer 1 mot = lire tous les poids ⇒ sur disque = ~70 s/mot (SSD SATA) à ~230 s/mot (HDD), inutilisable. Les poids actifs DOIVENT être en RAM.
  4. Scaling laws sous-linéaires : ÷2 erreur = ×10 ressources ; une meilleure archi change la constante, pas l'exposant.
- 🟢 **Seul levier prouvé** vers plus d'intelligence sans plus de params : le **raisonnement lent (chain-of-thought / test-time compute)** — le temps de calcul augmente prouvablement le pouvoir d'un modèle de taille fixe (Merrill & Sabharwal 2023). Sur CPU, le temps est disponible.
- L'inspiration cerveau ne lève pas le mur : le cerveau a ~10¹⁴ synapses (~centaines de To) — énorme capacité sur substrat neuromorphique basse énergie ; il confirme que l'intelligence exige de la capacité, il ne la comprime pas dans 6 Go.

**Chemin réaliste (recommandé) :** modèle open ~14B (Qwen2.5) quantifié int4 (~7 Go) + fine-tuning LoRA (spécialisation) + retriever/mémoire disque #18 (savoir) + raisonnement lent (intelligence) + block-local #14 (contexte long). Étonnamment capable — pas frontier.

---
