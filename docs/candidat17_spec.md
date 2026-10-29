# Candidat #17 — Block-Local + Mémoire Paginée Hiérarchique (BL-PM) — Design Spec

**Statut** : spec de conception — AUCUNE implémentation ici (implémentation = chef Opus,
fichier cible `candidats_actifs/block_local_pagemem.py`).
**Auteur** : architecte Fable. **Validation** : chef Opus.
**Matériel cible (dur)** : CPU 2 cœurs, 6 Go RAM, HDD mécanique 500 Go, AUCUN GPU.
fp32 en RAM, int8 sur disque, MKL threads=2, `torch.bmm` pour toute op lourde.

**Fondement** : le raisonneur reste **block-local #14** (`candidats_actifs/block_local.py`,
ppl LM 7.786) — inchangé. Le verdict de #16 (`candidats_actifs/block_local_mem.py`,
`mem_lm_compare.py`) : le savoir déporté FONCTIONNE (recall 100 % sur `candidat16_pkm.py`)
mais (1) K=32 lectures disque **aléatoires par TOKEN** → ~320 ms/token sur HDD (10 ms/seek),
(2) la mémoire **remplace** le FFN → ppl 10.56 vs 7.79, (3) routage product-key sparse →
gradient bruité, deux optimiseurs (AdamW + SparseAdam).

**Thèse #17** : sur un HDD, le coût dominant n'est pas les octets lus mais le **nombre et la
localité des accès**. On applique la hiérarchie mémoire d'un ordinateur au savoir du modèle :

| Niveau | Contenu | Analogie |
|---|---|---|
| Poids du raisonneur (RAM, ~1 Mo) | intelligence | registres/L1 |
| Banc de segment (RAM, ~16 Ko) | savoir actif du segment courant | L2 |
| Cache de pages LRU (RAM, réglable 1 Mo → 1 Go) | pages chaudes | DRAM cache |
| Store paginé memmap (HDD, jusqu'à ~centaines de Go) | tout le savoir, int8 | disque |

Trois principes, un par défaut de #16 :
1. **Amortissement temporel + localité disque** : le savoir est rangé en **pages contiguës**
   et récupéré **une fois par SEGMENT de C=64 tokens** (les segments ≡ les blocs de #14),
   par **≤ p=2 lectures séquentielles**. Soit **≤ p/C = 0,031 accès HDD/token** à froid
   (vs 32 pour #16 : ×1024 de réduction), et ~0 à chaud (cache).
2. **Mémoire EN PLUS du FFN**, branche résiduelle **gatée** (α init −4 → g≈0,018) :
   à l'init le modèle ≡ #14 à ~2 % près, à g→0 ≡ #14 exactement → la ppl LM ne peut pas
   régresser structurellement.
3. **Routage hiérarchique stable** : scoring **dense** des P centroïdes de pages (1 GEMM RAM),
   attention **pleine** (softmax dense) sur les p·S slots chargés, couplage par biais →
   gradients denses partout ; un seul optimiseur AdamW ; perte auxiliaire d'équilibrage.

**Notation** : B=batch, T=longueur, d=dim=64, H=2 têtes (raisonneur), L=3 couches, F=ffn=256,
C=64 (taille de bloc **et** de segment mémoire — même partition), n=⌈T/C⌉,
P=nb de pages, S=slots/page, p=pages lues/segment (défaut 2), d_k=d_v=64 (clé/valeur mémoire),
N = P·S slots au total.

---

## 1. Vue d'ensemble

```
                         RAM                                    HDD
        ┌───────────────────────────────────┐   ┌─────────────────────────────────┐
tokens →│ emb + pos                          │   │ pm_store.dat (memmap, int8)     │
        │ Bloc 0 : attn block-local + FFN    │   │  page 0 [scales|K int8|V int8]  │
        │ Bloc 1 : attn block-local + FFN    │   │  page 1 [ ...  contigu  ... ]   │
        │          + g ⊙ MémoirePaginée ◄────┼─┐ │  ...                            │
        │ Bloc 2 : attn block-local + FFN    │ │ │  page P-1                       │
        │ norm + head                        │ │ └─────────────▲───────────────────┘
        └───────────────────────────────────┘ │               │ ≤ p lectures
        ┌───────────────────────────────────┐ │               │ SÉQUENTIELLES
        │ MémoirePaginée (couche milieu)    │ │               │ par SEGMENT
        │  centroïdes pages (P, d_k)  [RAM] │ │   ┌───────────┴───────────┐
        │  1/ requête de segment q_seg      │ └───┤ Cache LRU de pages     │
        │     (passé STRICT : segment i-1   │     │ (R pages fp32 en RAM)  │
        │      + moyenne document 0..i-1)   │     │ hit → 0 accès HDD      │
        │  2/ top-p pages (GEMM RAM)        │     └───────────────────────┘
        │  3/ fetch pages → banc (p·S, d)   │
        │  4/ cross-attention par token     │
        │     sur le banc (2 bmm, RAM)      │
        └───────────────────────────────────┘
```

Le raisonneur (#14, intact) fait l'intelligence en RAM. La couche mémoire, insérée à la couche
du milieu, récupère le savoir par **segment** (pas par token) : une requête causale résume le
passé, sélectionne ≤ p pages, les charge (cache RAM ou 1 lecture séquentielle HDD chacune),
puis **tous les C tokens du segment** y accèdent par cross-attention **en RAM** (0 accès HDD).

---

## 2. Amortissement des accès HDD (le cœur)

### 2.1 Pages : l'unité de localité disque

Les N slots de savoir ne sont plus indépendants (#16) mais regroupés en **P pages de S slots
contigus**. Une page = un enregistrement contigu sur disque :

```
page j, offset j·PAGE_BYTES dans pm_store.dat :
  [scale_K f32 | scale_V f32]                    8 octets
  [K int8 : S × d_k octets]                      clés des slots
  [V int8 : S × d_v octets]                      valeurs des slots
PAGE_BYTES = 8 + S·(d_k + d_v)      (S=32, d=64 → 4104 o ; S=64 → 8200 o)
```

Lire une page = **1 seek + 1 transfert séquentiel de ~4–8 Ko**. C'est l'inverse exact de #16
(K=32 lignes de 256 o éparpillées dans une table de 64 Mo = 32 seeks).

### 2.2 Routage à deux niveaux (remplace le product-key)

- **Niveau 1 — page (grossier, en RAM)** : une matrice de centroïdes `K_pages (P, d_k)`
  apprise, résidente en RAM (P=65536, d_k=64 fp32 → 16 Mo ; démo P=1024 → 256 Ko).
  Une requête **par segment** la score en **un seul GEMM**, top-p (p=2).
- **Niveau 2 — slot (fin, en RAM après fetch)** : les p pages chargées forment un **banc**
  de p·S slots (clés + valeurs, dé-quantifiés fp32). Chaque token du segment fait une
  attention softmax **pleine** sur ces p·S ≤ 128 slots (2 bmm). Aucun top-k fin, aucun
  accès disque à ce niveau.

C'est la structure du product-key (grossier × fin) mais le niveau fin est devenu **gratuit en
I/O** parce qu'il opère sur des données déjà en RAM, et le niveau grossier ne coûte que
**p accès par segment**.

### 2.3 Comptage exact des accès (l'argument central)

Par segment de C=64 tokens, dans le pire cas (cache entièrement froid) :

| | #16 (PKM/token) | #17 (pages/segment) | facteur |
|---|---|---|---|
| accès HDD aléatoires / segment | K·C = 32·64 = **2048** | ≤ p = **2** | **1024×** |
| accès HDD aléatoires / token | **32** | ≤ p/C = **0,031** | **1024×** |
| octets lus / token | 32·64·4 = **8192 o** | ≤ p·PAGE_BYTES/C = **128 o** (S=32) | **64×** |
| nature des lectures | aléatoires (4 o utiles/seek amortis) | **séquentielles** (page entière utile) | — |
| latence disque / token (HDD : 10 ms/seek + 150 Mo/s) | 32·10 ms = **~320 ms** | 0,031·(10 + 0,03) ms = **~0,31 ms** | **~1000×** |

Avec un cache chaud de taux de hit ρ : accès/token = (1−ρ)·p/C. À ρ=0,9 → **0,003
accès/token, ~0,03 ms/token**. Le raisonneur #14 tournant à quelques ms/token sur 2 cœurs,
le disque cesse d'être le goulot **même à froid**.

### 2.4 Cache RAM chaud (LRU) et préchargement

- **Cache LRU** de R pages fp32 dé-quantifiées : `OrderedDict{page_id → (K fp32, V fp32)}`.
  Coût RAM = R·S·(d_k+d_v)·4 o. Démo : R=256, S=32 → 4 Mo. Échelle : R=65536, S=64 →
  4 Go (rentre dans 6 Go avec le modèle ~négligeable). Compteurs `hits/misses/bytes` intégrés.
- **Hot-set épinglé (option)** : les F pages les plus fréquentes sur le corpus d'entraînement
  sont épinglées (jamais évincées) — analogie L1.
- **Préchargement (option v2, pas requis pour valider)** : la requête du segment i+1 ne dépend
  que des segments ≤ i ; en génération on peut émettre la lecture de ses pages **pendant** le
  calcul du segment i (thread lecteur + file), masquant totalement la latence résiduelle.

---

## 3. Causalité stricte

### 3.1 Construction de la requête de segment (passé uniquement)

Soit `h (B, T, d)` l'état caché entrant dans la couche mémoire (il ne dépend que du passé de
chaque position : l'attention block-local est causale, audit #14). Vue par blocs
`h_blk (B, n, C, d)` — **même partition que l'attention** (positions [iC, (i+1)C)).

```
seg_sum  = h_blk.sum(2)                              # (B, n, d)
prev     = cat([zeros(B,1,d), seg_sum[:, :n-1]/C], 1)      # moyenne du segment i-1
csum     = cumsum(seg_sum, dim=1)                          # cumul sur l'axe n (n ≤ 64,
doc      = cat([zeros(B,1,d), csum[:, :n-1]], 1) / max(iC,1)  # convention #15 : autorisé)
q_seg    = W_qseg( cat([prev, doc], -1) )            # Linear(2d → d_k) → (B, n, d_k)
```

La requête du segment i n'utilise que les positions **< iC**, toutes strictement passées pour
tout token t = iC + r du segment i. Le décalage se fait par le même `cat` décalé sur l'axe des
blocs que `k_prev` dans #14 (aucune boucle sur T). `cumsum` porte sur l'axe des blocs
(n petit), précédent établi par #15 (`block_local_global.py`).

Note padding : si C ∤ T, le dernier bloc contient des zéros de padding ; il ne sert de passé à
aucun segment réel → aucune correction nécessaire (le harnais utilise T multiples de 64 ;
sinon diviser `seg_sum` par le compte réel de tokens, vecteur (n,)).

### 3.2 Pourquoi rien ne fuit

- **Sélection de pages** : fonction des positions < iC uniquement (§3.1).
- **Contenu des pages** : `K_slots/V_slots` sont des **paramètres statiques** du modèle — ils
  ne dépendent pas de la séquence, donc les lire ne peut véhiculer aucune information future.
- **Cross-attention par token** : requête = h[t] (causal), clés/valeurs = banc statique.
- **Segment 0** : pas de passé → sortie mémoire **multipliée par 0** (masque (1,n,1,1) avec
  m₀=0 ; PAS un masque −inf sur tous les slots, qui produirait des NaN au softmax).

### 3.3 Staleness assumée (contrat d'usage)

Le savoir disponible au segment i est sélectionné d'après le contexte ≤ iC−1 : un indice
(entité, sujet) doit apparaître **avant** le segment où le savoir est requis — retard maximal
d'un segment (64 tokens). C'est le compromis RETRO (chunked cross-attention sur la
récupération du chunk précédent) : les documents réels ont une persistance thématique qui le
satisfait, et la tâche démo (§9) est construite pour l'exercer honnêtement.

### 3.4 Tests de causalité exigés (convention repo, dans `__main__`)

1. Audit causal standard (#14/#16) : perturber la 2ᵉ moitié, diff 1ʳᵉ moitié < 1e-6,
   T ∈ {48, 100, 200, 320} — **en mode entraînement RAM et en mode memmap+cache**.
2. Audit de la sélection : perturber des tokens du segment i ne doit changer `page_idx`
   d'aucun segment ≤ i (et doit pouvoir changer ceux > i).

---

## 4. Intégration sans régression LM

Le bloc mémoire **conserve le FFN** (défaut n°2 de #16 corrigé). Couche du milieu (i = L//2) :

```
x = x + attn(norm1(x), mask)          # raisonnement local  (identique #14)
x = x + ffn(norm2(x))                 # raisonnement FFN    (identique #14, CONSERVÉ)
x = x + g ⊙ mem(norm3(x))             # savoir, branche additive gatée
g = sigmoid(α),  α ∈ R^d  init −4.0  → g ≈ 0,018
```

- À l'init : contribution mémoire ~2 % → dynamique d'entraînement ≈ #14 ; le gate ne s'ouvre
  (par gradient) que là où la mémoire aide (tokens factuels), par canal.
- Garantie structurelle : à α → −∞ le modèle est **exactement** un `BlockLocalTransformer`
  aux mêmes poids denses. **Test d'équivalence exigé** : copier les poids denses dans un #14,
  poser α=−20, diff sorties < 1e-5.
- Les couches 0 et 2 restent des `StdBlock` #16 (attn + FFN). Une seule couche mémoire
  (suffisant pour la démo ; extensible).
- Critère de succès ppl (§8) : parité avec #14 (7.786) à ±0.1 sur le harnais
  `mem_lm_compare.py`, contre 10.56 pour #16.

---

## 5. Formes de tenseurs et ops (chemin d'entraînement, vectorisé)

Entrée : `h (B, T, d)` (post-norm3), T = n·C. Tout est bmm/Linear/gather ; **aucun unfold,
aucun einsum, aucune boucle Python sur T** ; `cumsum` uniquement sur l'axe n (convention #15).

| # | Opération | Shape sortie | Nature |
|---|---|---|---|
| 1 | vue blocs `h_blk` | (B, n, C, d) | view |
| 2 | `prev`, `doc` (§3.1 : sum, cat décalé, cumsum sur n) | 2 × (B, n, d) | O(B·n·d) |
| 3 | `q_seg = W_qseg(cat([prev,doc],-1))` | (B, n, d_k) | **sgemm** (2d→d_k) |
| 4 | `s = q_seg @ K_pagesᵀ` | (B, n, P) | **sgemm** (d_k→P), RAM |
| 5 | `s_p, page_idx = s.topk(p, -1)` | 2 × (B, n, p) | topk sur P (comme PKM #16) |
| 6 | gather `Kb = K_slots[page_idx]`, `Vb = V_slots[page_idx]` | (B, n, p·S, d_k/d_v) | index_select, O(B·n·p·S·d) memcpy |
| 7 | `q_tok = W_qtok(h)` puis vue | (B·n, C, d_k) | **sgemm** |
| 8 | `scores = bmm(q_tok·d_k^{-1/2}, Kbᵀ)` | (B·n, C, p·S) | **bmm** (batch B·n) |
| 9 | biais de page : vue (B,n,C,p,S) ; `scores += τ·s_p[:,:,None,:,None]` | idem | élémentaire |
| 10 | `attn = softmax(scores, -1)` | (B·n, C, p·S) | élémentaire, p·S ≤ 128 |
| 11 | `o = bmm(attn, Vb)` | (B·n, C, d_v) | **bmm** |
| 12 | `out = W_o(o)` ; vue (B, T, d) ; `out *= m_seg0` ; `x += g⊙out` | (B, T, d) | **sgemm** + élémentaire |

τ : température scalaire apprise (init 1). Ops lourdes : 4 sgemm + 2 bmm, toutes MKL.
Coût hors-GEMM : softmax O(B·T·p·S) = B·T·128 — du même ordre que le softmax local 2C=128
de #14. Paramètres mémoire (démo P=1024, S=32, d=64) : centroïdes 65 K + slots K/V 2×2,1 M
≈ 4,3 M fp32 en RAM d'entraînement (17 Mo) — un seul AdamW, **weight_decay=0 sur
{K_pages, K_slots, V_slots}** (sinon les pages non sélectionnées se dégradent).

**Chemin d'inférence (le disque existe)** — boucle sur les segments (n petit : autorisé) :

```
for i in range(n):                                # ex. n=8 à T=512
    if i == 0: bank = None                        # m_seg0 = 0
    else:
        q  = W_qseg(cat([prev_i, doc_i]))         # (B, d_k)   — passé uniquement
        s  = q @ K_pages.T                        # (B, P)     GEMM RAM
        s_p, pid = s.topk(p)                      # (B, p)
        for j in pids_uniques(pid):               # ≤ B·p, typiquement p
            page = lru.get(j)                     # hit → RAM, 0 I/O
            if page is None:                      # MISS :
                raw  = mm[j*PB:(j+1)*PB]          #   1 lecture SÉQUENTIELLE memmap
                page = dequant_int8(raw)          #   (K,V) fp32 (S, d)
                lru.put(j, page); ctr.miss += 1; ctr.bytes += PB
        bank = assemble(pid, s_p)                 # (B, p·S, d_k), (B, p·S, d_v), biais
    # les C tokens du segment i : étapes 7–12 en RAM (bmm), 0 accès HDD
```

Phase T (entraînement) : tables en RAM, chemin vectorisé du tableau. Phase D (déploiement) :
quantization int8 par page → dump memmap `resultats/pm_store.dat` → éval avec cache + compteurs
(même schéma deux-phases que `candidat16_pkm.py`, qui dumpe `pkm_values.dat`).

---

## 6. Entraînement stable (défaut n°3 de #16 corrigé)

1. **Scoring de pages dense** : le GEMM de l'étape 4 score **tous** les P centroïdes ;
   le gradient atteint les centroïdes sélectionnés via le **biais couplé au softmax**
   (étape 9) — pas de straight-through, pas d'estimateur bruité.
2. **Gradient fin dense** : softmax plein sur p·S slots → chaque slot chargé reçoit un
   gradient à chaque token du segment. Le gather (étape 6) rétro-propage par scatter-add sur
   des **pages entières contiguës** : mises à jour cohérentes par page (vs #16 : K=32 slots
   épars parmi 262 144, SparseAdam dédié). **Un seul optimiseur AdamW.**
3. **Équilibrage de charge** (Switch) : `L_aux = λ·P·Σ_j f_j·m_j`, f_j = fréquence de
   sélection top-1 de la page j dans le batch (sur les B·(n−1) segments), m_j = prob softmax
   moyenne de j (softmax complet sur s, calculé de toute façon). λ = 0,01. Évite les pages
   mortes et **fait émerger la localité** : des faits co-requis migrent vers la même page.
4. **Exploration (option)** : bruit de Gumbel σ=0,1 sur `s` pendant les 500 premiers steps.
5. Le gate α (init −4) protège la ppl pendant que le routage se stabilise : la mémoire
   n'influence la perte que lorsqu'elle devient utile.

---

## 7. Configurations

| | démo (validation) | échelle (cible matériel) |
|---|---|---|
| raisonneur | #14 : d=64, H=2, L=3, F=256, C=64 | idem, agrandi au besoin |
| P (pages) | 1024 | 65 536 – 1 M |
| S (slots/page) | 32 | 64 |
| N = P·S | 32 768 | 4,2 M – 67 M |
| p (pages/segment) | 2 | 2 |
| store disque int8 | 4,2 Mo | 0,5 – 8,6 Go (500 Go dispo) |
| centroïdes RAM | 256 Ko | 16 – 256 Mo |
| cache LRU R | 256 pages (4 Mo) | 65 536 pages (~4 Go < 6 Go) |
| banc/segment | 64 slots (16 Ko) | 128 slots (64 Ko) |

---

## 8. Métriques et protocole de comparaison (vs #16 et #14)

Sorties JSON dans `resultats/` (convention `candidat16.json` / `mem_lm.json`).

| métrique | comment | cible #17 | référence #16 |
|---|---|---|---|
| **accès HDD aléatoires/token** | compteur `ctr.miss / tokens` au fetch memmap | **≤ 0,031 à froid, ≪ 0,01 à chaud** | 32 |
| octets disque/token | `ctr.bytes / tokens` | ≤ 128 o | 8192 o |
| latence disque/token (modèle analytique HDD : 10 ms/seek + 150 Mo/s, convention `candidat16.json`) | `miss/token · (10 ms + PB/150e6)` | **≤ 0,31 ms** | ~320 ms |
| taux de hit cache ρ + RAM cache | compteurs LRU + R·PB_fp32 | ρ ≥ 0,8 sur FactMix ; RAM conforme §7 | n/a |
| **ppl LM TinyShakespeare** | harnais EXACT de `mem_lm_compare.py` (SEQ=128, 3000 steps, B=32, seed 42/999) | **≤ 7,9** (parité #14 = 7,786) | 10,56 |
| accuracy factuelle | FactMix §9, chiffres de valeur | ≥ 95 % (#17) vs ~chance (#14) | (recall 100 % hors-LM) |
| ppl langage dans FactMix | ppl restreinte aux chars de remplissage | parité #14 sur le même corpus | n/a |

Mesure réelle optionnelle : éval memmap après `echo 3 > /proc/sys/vm/drop_caches` (cache OS
froid) pour un chiffre de latence mesuré et non seulement analytique. Les compteurs restent
la métrique primaire (indépendants du disque de dev, qui peut être un SSD).

---

## 9. Tâche de démonstration : **FactMix** (pas TinyShakespeare seul)

Générateur `data/make_factmix.py` (seed 123), char-level (convention repo) :

- **Base de faits** : M = 2000 associations entité→valeur, entité `@eXXXX`, valeur `#vYYYY`
  (4 chiffres chacune), mapping aléatoire fixe.
- **Document** : `@eXXXX : <remplissage TinyShakespeare de F chars> ; code @eXXXX = #vYYYY .`
  avec F ~ U[192, 384] → la **première mention** de l'entité est 3 à 6 segments **avant** la
  valeur (exerce la staleness §3.3), et **au-delà du champ réceptif local L·C = 192** de #14 :
  un raisonneur sans mémoire ne peut ni copier la valeur (jamais vue) ni la retrouver par
  attention. Documents concaténés, **SEQ = 512** (n = 8 segments).
- **Éval** : (a) accuracy teacher-forced sur les 4 chiffres de `#vYYYY` (le savoir) ;
  (b) ppl sur les chars de remplissage (le langage, parité #14 exigée) ;
  (c) tous les compteurs disque §8 en phase D (memmap int8 + cache LRU) ;
  (d) diagnostic de localité : nb de pages distinctes utilisées, histogramme de fréquence
  (justifie le hot-set et mesure ρ atteignable).
- **Baselines sur le même corpus** : #14 pur (attendu : proche du hasard sur les chiffres,
  M=2000 faits ≫ capacité de ~200 K params + gap > RF) et #16 (attendu : accuracy correcte
  mais 32 accès/token et ppl langage dégradée). #17 doit gagner sur les trois axes à la fois.

---

## 10. Plan de fichiers et tests structurels

- `candidats_actifs/block_local_pagemem.py` — réutilise `BlockLocalAttention`, `FeedForward`,
  `_build_block_mask` de `block_local.py` (mêmes imports que #15/#16).
  Classes : `PagedMemory` (§5, chemin entraînement vectorisé + chemin inférence par segments),
  `PageStore` (quantize/dump/fetch memmap + LRU + compteurs), `PageMemBlock`
  (attn + FFN + mémoire gatée, §4), `BlockLocalPageMemTransformer`.
- `data/make_factmix.py` — générateur §9.
- `mem17_lm_compare.py` — clone de `mem_lm_compare.py` sur #17 → `resultats/mem17_lm.json`.
- `mem17_factmix.py` — entraînement FactMix + phase D memmap → `resultats/candidat17.json`.

Tests `__main__` (convention repo #14/#15/#16) :
1. Shapes T ∈ {48, 100, 200, 512}.
2. Gradients non-None/non-NaN pour TOUS les paramètres (centroïdes, K/V slots, α, τ inclus)
   avec un seul AdamW.
3. Audit causal < 1e-6 (deux modes : RAM et memmap+cache) + audit de sélection (§3.4).
4. Équivalence α=−20 ≡ #14 mêmes poids, diff < 1e-5 (§4).
5. Cohérence RAM/disque : sorties du chemin vectorisé (tables fp32 RAM) vs chemin
   memmap int8 + cache — diff bornée par l'erreur de quantization (rapportée, attendue ~1e-2).
6. Compteurs : sur T=512, B=1, cache vide → `miss ≤ p·(n−1) = 14`, soit ≤ 0,027 accès/token,
   assertion `miss/token < 0,05`.

---

## Annexe — pourquoi #17 sera « bien meilleur sur HDD » que #16, en une équation

Coût disque par token : `t = a · (t_seek + b/BW)` avec a = accès aléatoires/token,
b = octets/accès. #16 : a = 32 (t ≈ 320 ms/token, inutilisable). #17 : a = (1−ρ)·p/C ≤ 0,031,
b = PAGE_BYTES séquentiels (t ≤ 0,31 ms/token à froid, ~0,03 ms à ρ=0,9, ~0 avec préchargement).
La réduction vient de trois multiplications indépendantes : **amortissement temporel** (÷C=64,
une récupération par segment), **localité** (÷K/p=16, pages contiguës au lieu de slots épars),
**cache** (÷1/(1−ρ)). Le tout sans toucher au raisonneur #14 ni à sa ppl (gate §4).
