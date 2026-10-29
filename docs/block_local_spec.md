# Block-Local Attention — Design Spec (Step 2, candidat #14)

**Statut** : spec de conception — AUCUNE implémentation ici (Step 3 = `candidats_actifs/block_local.py`).
**Auteur** : délégué Fable (difficulté 8). **Validation** : chef Opus.
**Fondement** : thèse du plan (EXECUTION_PLAN.md) — sliding-window softmax W=64 résout déjà
l'accuracy (copy 100 %, LM 7.64) ; son unique échec est la vitesse, car ses ops lourdes
(matrice T×T du #9, `unfold`+`einsum` du #13) sont **hors MKL**. Ici, chaque op lourde est un
**sgemm batché** (`torch.bmm` / `@` sur tenseurs 3D).

**Notation** : B=batch (32 aux benchmarks), T=longueur, d=dim=64, H=heads=2, h=d/H=32 (head_dim),
L=layers=3, F=ffn=256, C=taille de bloc (défaut **64**), n=⌈T/C⌉ (nombre de blocs),
T_pad=n·C. fp32 partout, CPU, MKL threads=2 (cf. `resultats/baseline_ref.json`).

---

## 1. Partition en blocs

### 1.1 Découpage
La séquence (après embedding) est découpée en **n = ⌈T/C⌉ blocs contigus** de C positions,
identiques à toutes les couches (partition fixe, alignée sur les positions absolues) :

- bloc i couvre les positions globales **[iC, (i+1)C)**, i ∈ {0, …, n−1} ;
- la position globale t appartient au bloc **i = ⌊t/C⌋**, à la ligne intra-bloc **r = t mod C**.

**Contexte du bloc i = bloc i−1 ⧺ bloc i** (concaténation), soit **2C clés/valeurs**.
C'est le chevauchement qui a manqué à KVMemory v3 (chunks disjoints → copy 1.8 %) : ici chaque
token voit toujours **au moins les C tokens précédents** (cf. §4).

### 1.2 Padding quand C ∤ T
On pad **à droite** jusqu'à T_pad = ⌈T/C⌉·C, au niveau des tenseurs Q, K, V (zéros) :

- **Clés/valeurs paddées** : elles n'existent que dans le **dernier** bloc, qui ne sert de
  « bloc précédent » à personne. À l'intérieur du dernier bloc, une clé paddée (position
  globale ≥ T) est **strictement future** pour toute requête réelle (position < T) du même
  bloc → elle est déjà tuée par le masque triangulaire causal de §3. **Aucun masque de
  padding supplémentaire n'est requis pour les clés.**
- **Requêtes paddées** : leurs lignes de softmax sont bien définies (la diagonale j = C+r est
  toujours visible → jamais de ligne toute-−inf, donc pas de NaN) ; leurs sorties sont
  simplement **tronquées** au retour : `out[:, :T]`.
- Les longueurs du protocole tombent bien : copy seq_len=192=3C, LM seq_len=128=2C,
  benchmarks {128,256,512,1024,2048,4096} tous multiples de 64 ; seul T=48 est paddé (n=1,
  T_pad=64).

### 1.3 Premier bloc (pas de bloc précédent)
Le bloc 0 n'a pas de prédécesseur. On **préfixe un bloc de zéros** sur l'axe des blocs
(décalage/shift, cf. §2 étape 4) pour que toutes les formes soient uniformes, et on masque à
**−inf toute la moitié « bloc précédent »** du bloc 0 (§3). Les zéros ne servent qu'à
uniformiser les shapes du bmm ; ils ne contribuent jamais au softmax.

**Cas limite T ≤ C** : n=1, seul le bloc 0 existe → moitié gauche entièrement masquée +
triangulaire à droite = **attention causale pleine exacte** sur les T tokens. Un seul chemin
de code, pas de branche spéciale (contrairement au `if T <= W` de `sliding_window.py`).
L'audit causal du protocole (T=48) s'exécute donc dans le régime « attention pleine ».

---

## 2. Formulation batched-bmm (shapes exactes)

Entrée de la couche d'attention : `x : (B, T, d)` (déjà normalisé par LN, pré-LN comme baseline).

| # | Opération | Shape sortie | Nature |
|---|-----------|--------------|--------|
| 1 | `qkv = Linear(d, 3d, bias=False)(x)` | (B, T, 3d) | **sgemm** (B·T, d)@(d, 3d) |
| 2 | reshape + permute → q, k, v | 3 × (B, H, T, h) | view/permute + 1 `.contiguous()` |
| 3 | pad droit T → T_pad (zéros), puis view en blocs | (B, H, n, C, h) | `F.pad` + view, O(T·d) |
| 4 | **shift bloc précédent** : K_prev = concat([0-bloc, K_blk[:, :, :n−1]], axe blocs) ; idem V_prev | (B, H, n, C, h) | `cat`, memcpy O(T·d) |
| 5 | **contexte** : K_ctx = concat([K_prev, K_blk], axe intra-bloc) ; idem V_ctx | (B, H, n, 2C, h) | `cat`, memcpy O(T·d) |
| 6 | aplatir pour bmm : Q′, K′, V′ | (B·H·n, C, h), (B·H·n, 2C, h), (B·H·n, 2C, h) | view (gratuit) |
| 7 | **scores = (Q′ · h^{−1/2}) @ K′ᵀ** | (B·H·n, C, 2C) | **`torch.bmm` = sgemm batché**, B·H·n GEMMs de (C×h)@(h×2C) |
| 8 | masque : view (B, H, n, C, 2C), `masked_fill(M, −inf)` avec M : (n, C, 2C) broadcast | (B·H·n, C, 2C) | élémentaire O(B·H·T·2C) |
| 9 | `P = softmax(scores, dim=−1)` | (B·H·n, C, 2C) | élémentaire O(B·H·T·2C) |
| 10 | **out = P @ V′** | (B·H·n, C, h) | **`torch.bmm` = sgemm batché** |
| 11 | view (B, H, n, C, h) → (B, H, T_pad, h) → transpose → (B, T_pad, d) → slice [:, :T] | (B, T, d) | view/transpose + 1 copie |
| 12 | `proj = Linear(d, d, bias=False)` | (B, T, d) | **sgemm** |

Le facteur d'échelle 1/√h (=1/√32) est appliqué **à Q avant le bmm** (tenseur 2× plus petit
que les scores).

**Bilan des ops lourdes** — toutes sont des sgemm MKL :
qkv (étape 1), les **deux bmm** (7, 10), proj (12), plus les deux Linear du FFN.
À T=4096, B=32 : les bmm sont **B·H·n = 32·2·64 = 4096 GEMMs de (64×32)@(32×128)**, dispatchés
par MKL en batch (parallélisation naturelle sur l'axe batch, adaptée à threads=2).

**Ops non-GEMM restantes** (toutes O(T), négligeables devant les sgemm) :
pad/cat/contiguous = memcopies O(B·T·d) ; masque + softmax = O(B·H·T·2C) élémentaire
(à T=2048 : 16,8 M d'éléments, contre 268 M pour le softmax T×T du baseline — 16× moins).
**Interdits respectés : aucun `unfold`, aucun `einsum`, aucun `cumsum`, aucune boucle Python
sur T ni sur les blocs** — le « bloc précédent » s'obtient par un unique `cat` décalé sur
l'axe des blocs (étape 4), pas par boucle. (Boucle Python sur les L=3 couches : autorisée.)

Contraste avec `swin_fast.py` (candidat #13) : son `einsum('bhtd,bhtdw', …)` sur une vue
stridée d'`unfold` tombe dans le chemin lent non-MKL et fait O(B·H·T·W·h)=268 M MACs hors
sgemm à T=2048 ; ici le même travail est DANS les bmm, et le hors-GEMM est réduit à 16,8 M
d'opérations élémentaires contiguës.

---

## 3. Masque causal exact

Scores du bloc i : matrice (C, 2C). Ligne r ↔ requête globale t = iC + r.
Colonne j ∈ [0, 2C) ↔ clé globale :

- **moitié gauche** j ∈ [0, C) : bloc i−1, position t′ = (i−1)C + j ;
- **moitié droite** j ∈ [C, 2C) : bloc i, position t′ = iC + (j − C).

Causalité (t′ ≤ t) :

- Gauche : t′ ≤ t ⟺ (i−1)C + j ≤ iC + r ⟺ j ≤ C + r — **toujours vrai** (j < C ≤ C + r).
  → La moitié « bloc précédent » est **entièrement visible** pour i ≥ 1.
- Droite : t′ ≤ t ⟺ j − C ≤ r. → **Triangulaire inférieure (diagonale incluse)** sur la
  moitié courante.
- **Bloc 0** : pas de prédécesseur → moitié gauche **entièrement −inf** ; moitié droite
  triangulaire identique.

Masque booléen M ∈ {visible, −inf}^(n × C × 2C), broadcasté sur (B, H) :

```
M_base[r, j]  (blocs i ≥ 1) :   visible ⟺ (j < C) ∨ (j − C ≤ r)
M_0[r, j]     (bloc 0)      :   visible ⟺ (j ≥ C) ∧ (j − C ≤ r)
M = stack([M_0, M_base, …, M_base])        # shape (n, C, 2C)
```

Exemple C = 4 (·=visible, ×=−inf), bloc i ≥ 1 puis bloc 0 :

```
   bloc précédent | bloc courant          bloc précédent | bloc courant
r=0   · · · ·     |  · × × ×          r=0   × × × ×      |  · × × ×
r=1   · · · ·     |  · · × ×          r=1   × × × ×      |  · · × ×
r=2   · · · ·     |  · · · ×          r=2   × × × ×      |  · · · ×
r=3   · · · ·     |  · · · ·          r=3   × × × ×      |  · · · ·
```

Propriétés :
- **Zéro fuite** : toute entrée avec t′ > t est −inf par construction (dérivation ci-dessus) ;
  le padding droit est couvert par le même argument (§1.2). L'audit causal du protocole
  (diff < 1e−6 sur les 24 premières positions quand on modifie les 24 dernières) doit passer
  structurellement.
- **Pas de NaN** : chaque ligne r garde au moins la diagonale j = C + r visible (y compris
  bloc 0 ligne 0, et les lignes de requêtes paddées).
- M ne dépend que de (n, C) : **construit une fois par forward** (524 Ko en bool à T=4096)
  et **partagé par les 3 couches** — pas 3 constructions.
- Coût du `masked_fill` : un passage O(B·H·T·2C) sur le tenseur de scores, inévitable et
  linéaire en T.
- Fenêtre effective par token : t voit [(⌊t/C⌋−1)·C, t], soit entre C+1 (r=0) et 2C (r=C−1)
  clés — **lookback garanti ≥ C = 64, maximum 2C−1 = 127**.

---

## 4. Champ réceptif : preuve que C=64 suffit pour gap=186

**Par couche.** La sortie de la couche à la position t dépend des entrées aux positions
s ∈ [b(t), t] avec **b(t) = (⌊t/C⌋ − 1)·C** (début du bloc précédent, clippé à 0).

**Composition sur ℓ couches.** b est croissante, donc le minimum atteignable après 2 couches
est b(b(t)) ; b(t) étant un multiple de C, b(b(t)) = (⌊t/C⌋ − 2)·C. Par récurrence :

> après ℓ couches, t dépend des entrées sur **[(⌊t/C⌋ − ℓ)·C, t]** (clippé à 0),
> soit un lookback **RF(t, ℓ) = ℓ·C + (t mod C)**.

Le champ réceptif croît d'**exactement C par couche** dans le pire cas (t multiple de C) :

| Couches ℓ | Lookback garanti (min sur t) | Max (t mod C = 63) |
|---|---|---|
| 1 | 64 | 127 |
| 2 | 128 | 191 |
| 3 | **192** | 255 |

**Verdict : L·C = 3·64 = 192 ≥ gap = 186** ✅ (marge = 6 positions, dans le pire alignement
de bloc ; marge 6 à 69 selon t mod C). Le FFN et les résiduelles ne réduisent pas le RF.

**Argument de domination (transfert de l'accuracy de SW-64).** Le sliding window W=64
(candidat #9, copy 100 %, LM 7.64) voit [t−63, t] par couche, soit 3·63 = 189 composé.
Or pour tout t = iC + r : (i−1)C = t − 64 − r ≤ t − 64 < t − 63, donc l'ensemble visible du
block-local **contient strictement** celui de SW-64 **à chaque position et chaque couche**
(et 192 > 189 en composé). Tout motif de rappel réalisable par SW-64 est réalisable par
block-local — l'accuracy attendue est ≥ celle du candidat #9, avec les mêmes softmax exacts.

**Bonus structurel pour le gate LM.** À T ≤ 2C = 128, le bloc 1 voit tout le bloc 0 + son
triangle causal → chaque position voit **tout son passé** : block-local est **exactement**
l'attention causale pleine. Le LM étant entraîné à seq_len=128, le modèle est
**fonctionnellement identique au baseline** sur ce gate (mêmes ensembles visibles, mêmes
softmax) → ppl ≈ 7.6 attendue par construction, au bruit de seed près.

**C minimal.** La contrainte est L·C ≥ 186 ⟹ C ≥ 62. **C = 64 est le minimum pratique**
(aligné registres/SIMD, multiple des longueurs du protocole 128/192, marge ≥ 6). Pas besoin
de C par couche ni de C plus grand ; augmenter C améliorerait la marge mais coûte des FLOPs
linéairement (§7). Fallback si la copy échouait empiriquement malgré la preuve : C = 96
(RF min 288) — non attendu.

---

## 5. Profil mémoire : O(T·C), borné à T=4096

**Tenseur dominant** : les scores/softmax `(B·H·n, C, 2C)`, soit **8·B·H·T_pad·C octets**
(fp32) par couche — **linéaire en T** — contre **4·B·H·T² octets** pour le baseline
(formule vérifiée : elle reproduit exactement `attn_matrix_theoretical_mb` de
`resultats/baseline_ref.json`).

Mémoire du tenseur d'attention par couche, B=32, H=2, C=64 (Mo décimaux) :

| T | Baseline 4·B·H·T² | Block-local 8·B·H·T·C | Ratio T/(2C) |
|---|---|---|---|
| 512 | 67,1 Mo | 16,8 Mo | 4× |
| 1024 | 268,4 Mo | 33,6 Mo | 8× |
| 2048 | 1073,7 Mo | 67,1 Mo | 16× |
| 4096 | **4295,0 Mo → OOM mesuré** | **134,2 Mo** | 32× |

**Pic RAM prédit (forward, no_grad, B=32, T=4096)** — ensemble vivant au pire moment :
scores 134 Mo (+ P si non in-place 134 Mo) + K_ctx/V_ctx 2×67 Mo + q,k,v 3×32 Mo + résiduel
et sorties ~64 Mo + overhead allocateur → **≈ 450–700 Mo**, à comparer au baseline **OOM à
4,3 Go** (et 2287 Mo mesurés dès T=2048 ; block-local prédit ≈ 300–450 Mo à T=2048).
En entraînement (autograd conserve P par couche : 3×134 Mo à T=4096) : ~1,2–1,6 Go — encore
faisable, là où le baseline exigerait > 13 Go.

**État d'inférence incrémentale** (pertinent pour le barreau 70B) : le KV-cache utile est
borné à **2C = 128 clés/valeurs par couche** (fenêtre roulante alignée bloc), soit
O(L·C·d) = 2(K,V)·3·128·64·4 o ≈ 197 Ko/exemple — **constant en T**, vs O(L·T·d) croissant
pour le baseline.

---

## 6. Embeddings positionnels

**Identiques au baseline** : `token_emb(vocab, d)` + `pos_emb(max_seq_len, d)` appris,
additionnés à l'entrée, init N(0, 0.02), `max_seq_len=4096` par défaut (comme
`sliding_window.py` ; le défaut 2048 du baseline OOMait de toute façon avant).
Compatibilité avec le blocage : le découpage en blocs intervient **après** l'addition des
embeddings et n'est qu'un `view` sur l'axe T — chaque token garde son embedding de position
absolue ; la partition étant elle-même alignée sur les positions absolues, aucune interaction.
Les positions paddées (t ≥ T) n'ont pas besoin d'embedding (le pad se fait sur Q/K/V, après
la projection). Le rappel positionnel exact exigé par la copy task reste porté par les
embeddings absolus appris — même mécanisme qui a donné 100 % au candidat #9. Aucune
extrapolation au-delà de max_seq_len (limite partagée avec le baseline, hors périmètre).

---

## 7. Table de complexité et point de croisement (honnête)

**FLOPs par token** (1 MAC = 2 FLOPs ; d=64, h=32, H=2, F=256, L=3, C=64) :

- Linéaires par couche (identiques aux deux modèles) : qkv 6d² + proj 2d² + FFN 4dF
  = 98 304 → ×3 couches = **294 912**/token.
- Attention par couche : baseline QKᵀ+PV = 4·T·d = 256·T ; block-local = 8·C·d = **32 768**
  (constant). ×3 couches : baseline 768·T, block-local 98 304.

| T | Baseline FLOPs/token | Block-local FLOPs/token | Ratio | Forward total B=32 (GFLOP) : base / BL | Latence baseline mesurée | Latence BL prédite* | Speedup prédit |
|---|---|---|---|---|---|---|---|
| 512 | 688 k | 393 k | 1,75× | 11,3 / 6,4 | 557 ms | 450–750 ms | **0,8–1,2× (parité, peut perdre)** |
| 1024 | 1 081 k | 393 k | 2,75× | 35,4 / 12,9 | 2 103 ms | 900–1 600 ms | **1,3–2,3×** |
| 2048 | 1 868 k | 393 k | 4,75× | 122,4 / 25,8 | 10 632 ms | 1 800–3 300 ms | **3–6×** |
| 4096 | 3 441 k | 393 k | 8,75× | 451 / 51,5 | **OOM** | 3 600–6 600 ms | **gagné par forfait (mémoire) ; ~6–8× vs extrapolation** |

\* Hypothèses de la prédiction : les linéaires (75 % des FLOPs de BL) tournent à l'efficacité
sgemm du baseline (~11–20 GFLOP/s mesurés sur cette machine, threads=2) ; les bmm
(64×32)@(32×128) à 0,3–1× de cette efficacité (petits GEMMs batchés MKL) ; + overhead
élémentaire (masque/softmax/cat) O(T), 16× moindre que l'élémentaire O(T²) du baseline.

**Point de croisement — engagement honnête** :
- T ≤ 512 : **probablement plus lent** — les gros GEMMs T×T du baseline sont très efficaces
  en MKL, et le ratio FLOPs (1,75×) ne couvre pas la perte d'efficacité des petits bmm.
- T = 1024 : **croisement plausible mais NON garanti** (prédit 1,3–2,3×, incertitude ±40 % ;
  la leçon du projet — MEMOIRE.md : « le croisement est repoussé à T≈2048 en pratique à
  cause de MKL » — impose la prudence).
- T = 2048 : **victoire quasi certaine** (≥ 3× prédit ; le baseline y est déjà
  memory-bound : 10,6 s et 2,3 Go).
- T = 4096 : **victoire par forfait** — le baseline OOM (4,3 Go), block-local tient en
  < 0,7 Go.
- Le gate Step 8 exige « plus rapide à T ≥ 1024 » : si T=1024 est en parité, le fallback du
  plan s'applique (documenter le vrai croisement ; ne PAS augmenter C — les FLOPs
  d'attention croissent linéairement en C ; C=64 est déjà le minimum RF-admissible, §4).

Mémoire : voir table §5 (O(T·C) vs O(T²), 32× à T=4096).

---

## 8. Variante mémoire O(1) — esquisse (candidat ultérieur, PAS le primaire)

**Hybride local-softmax + état global linéaire (chunké, RetNet-style).** On garde la branche
block-local ci-dessus telle quelle, et on ajoute par tête un état récurrent
S_i ∈ R^{h×h} agrégé par chunk (chunk = le même bloc C) :
S_i = γ^C · S_{i−1} + Σ_{t∈bloc i} γ^{(fin_bloc−t)} k_t v_tᵀ, contribution pour les requêtes
du bloc i : o_global = (Q_blk · scale) @ S_{i−1} — un **bmm** de (B·H·n, C, h)@(h, h), la
récurrence ne portant que sur n = T/C états (T/64 itérations sur des tenseurs (B,H,h,h) —
boucle sur les *chunks*, pas sur T ; vectorisable en parallel-scan si besoin). Sortie :
o = o_local + λ·o_global (λ appris par tête, init petit). L'état S est **(B,H,h,h) = constant
en T** → mémoire d'inférence O(1) en contexte, rappel global non borné.
**Risque connu** : RetNet chunké seul a perdu le LM (8,07 vs 7,64, candidat #7) car le decay
γ^{t−τ} approxime mal les dépendances locales fines qui dominent la perplexité. L'hybride
devrait le récupérer : la softmax exacte locale (2C=128) prend en charge exactement ces
dépendances-là, et la branche linéaire n'a plus qu'à ajouter un signal long-terme grossier —
le candidat #8 (multi-γ) montre qu'une composante γ long-terme peut coexister avec un LM
à 7,53 ; son échec était la copy (87,6 %), précisément parce qu'il n'avait PAS de softmax
locale exacte, que l'hybride possède. À explorer au barreau 2 (70B) seulement si la mémoire
O(T·C) du primaire devient limitante ; sur ce lab, le block-local pur suffit (§5 : état
d'inférence déjà O(1) en T, 2C par couche — seul le *rappel au-delà de L·C* justifierait
l'hybride).

---

## Annexe A — Contraintes d'implémentation pour Step 3 (`candidats_actifs/block_local.py`)

1. **Interface projet** : `BlockLocalTransformer(vocab_size, dim=64, heads=2, layers=3,
   ffn_dim=256, max_seq_len=4096, block_size=64)` ; `forward(x:(B,T) long) → (B,T,vocab)`.
   Structure et conventions du baseline : pré-LN (`x = x + attn(norm1(x))` ;
   `x = x + ffn(norm2(x))`), LayerNorm/GELU/Linear(bias=False), LN final + head,
   init normal_(0, 0.02) via `.apply`. Aucun paramètre supplémentaire vs baseline (C est un
   hyperparamètre) → comptage de paramètres identique à `max_seq_len` égal.
2. **Ops autorisées pour le cœur** : `Linear`, `reshape/view/permute/transpose/contiguous`,
   `F.pad`, `torch.cat`, `torch.bmm` (ou `@` 3D), `masked_fill`, `softmax`.
   **Interdites** : `unfold`, `einsum`, `cumsum`, boucle Python sur T ou sur les blocs.
3. Masque construit **une fois par forward** ((n, C, 2C), bool), partagé entre couches.
4. **Tests attendus (Step 4)** : shapes (2,48)→(2,48,vocab) ; gradients non-None/non-NaN ;
   audit causal diff < 1e−6 (T=48 → régime attention pleine, §1.3).
   **Test d'équivalence fort recommandé** : à T=128 (=2C), block-local ≡ attention causale
   pleine (§4) — avec `torch.manual_seed` identique et même ordre de modules que le baseline,
   les sorties doivent coïncider à ~1e−5 près (seules différences : ordre des flottants dans
   des GEMMs partitionnés différemment).
5. Benchmarks Step 8 : mêmes protocoles que `resultats/baseline_ref.json` (B=32, threads
   documentés, médiane de 5 mesures, pic RAM par subprocess).
