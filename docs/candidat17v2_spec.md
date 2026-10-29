# Candidat #17-v2 — Mémoire Paginée à Routage Saillant (BL-PM v2) — Design Spec

**Statut** : spec de conception (implémentation = chef Opus, cible `candidats_actifs/block_local_pagemem.py` v2).
**Auteur** : architecte Fable. **Base** : #17-v1 (`docs/candidat17_spec.md`) — infra conservée
(pages, banc, cache LRU, chemin inférence par segments, aux loss, tests). Seuls changent :
la **requête de routage**, la **requête fine par token**, et le **gate**.

## 0. Autopsie de v1 (pourquoi 11,6 % = hasard)

Trois canaux devaient porter l'identité de l'entité `@eXXXX` jusqu'à la récupération ; les trois étaient morts :

1. `prev` = moyenne du segment i−1 : dans FactMix, le segment précédant la valeur est du
   remplissage pur → **zéro signal entité**.
2. `doc` = moyenne du document : l'entité pèse 6 tokens / 192 → ~3 % d'amplitude, noyée dans
   une moyenne de remplissage quasi identique d'une séquence à l'autre → routage
   **indépendant de l'entité**.
3. `q_tok = W_qtok(h_t)` à la position de rappel : l'entité est **hors du champ réceptif
   local** à la couche mémoire → le niveau fin ne peut pas discriminer les slots.

Conséquence : aucun gradient ne pointe vers le bon routage (plateau), la mémoire renvoie du
bruit entité-indépendant, et le gate scalaire l'injecte partout → ppl 13. v2 restaure les
trois canaux **sans toucher au budget I/O** : toute la précision ajoutée opère en RAM.

## 1. Principe : découpler grossier (I/O) et fin (RAM)

- **Niveau grossier, PAR SEGMENT** (coûte l'I/O) : sélectionner p=3 pages d'après les
  **tokens saillants** de TOUT le passé (max-pooling causal, §2.2), pas des moyennes.
  ≤ p accès HDD séquentiels/segment → **≤ p/C = 0,047 accès/token à froid garanti**, ∀T.
- **Niveau fin, PAR TOKEN** (gratuit en I/O) : chaque token attend sur le banc de p·S = 96
  slots **déjà en RAM**, avec une requête enrichie du **contexte-entité** c_i (§2.3).
- **Gate PAR TOKEN** (§2.5) : la mémoire ne s'écrit dans le résiduel que là où elle est
  pertinente → parité LM structurelle.

Notation v1 conservée : d=64, C=64, n=⌈T/C⌉, P=1024, S=32 ; nouveaux : **r=2** têtes de
saillance, **p=3** pages/segment (banc 96 ≤ 128 slots, coût softmax ≈ v1).

## 2. Mécanisme (chemin d'entraînement vectorisé, aucune boucle sur T)

`h (B,T,d)` post-norm3 ; `hb = h.view(B,n,C,d)`.

### 2.1 Résumés saillants par segment (remplace les moyennes)

```
e = hb @ W_sal                    # (B,n,C,r)   W_sal: Linear(d, r, bias=False)
a = softmax(e, dim=2)             # softmax sur les C tokens du segment
U = a^T · hb   (bmm par (B·n))    # (B,n,r,d)   r "soft top-1" par segment
```

Le softmax sur C **concentre** le poids sur les tokens saillants (motif `@e`+chiffres,
apprenable : à la couche milieu, h encode déjà le contexte local ±64) : l'entité survit au
pooling à amplitude ~1 au lieu de 6/192. r=2 têtes = 2 candidats saillants par segment.

### 2.2 Routage grossier : cummax causal sur tout le passé

```
Z     = U @ K_pages.T                       # (B,n,r,P)  1 GEMM RAM (comme v1)
Zseg  = Z.max(dim=2).values                 # (B,n,P)    max sur les têtes
Zcum  = torch.cummax(Zseg, dim=1).values    # cummax sur l'axe SEGMENTS (n petit, conv. #15)
route = cat([-inf·ones(B,1,P), Zcum[:, :n-1]], 1)   # décalage → passé STRICT
s_p, pidx = route.topk(p, -1)               # (B,n,p)   p=3
```

**max, pas moyenne** : une page fortement matchée par UN token saillant de N'IMPORTE quel
segment passé garde son score plein quel que soit le remplissage qui suit → l'entité
mentionnée 3+ segments avant reste routable (v1 : invisible dès i−2). Bonus I/O : `route`
est **monotone en i** → les top-p se stabilisent d'un segment à l'autre → pages répétées →
hits LRU, accès distincts mesurés ≪ p·(n−1).

### 2.3 Contexte-entité c_i (répare le niveau fin, 0 I/O)

Le token de rappel ne voit pas l'entité localement ; on la lui apporte par une attention
minuscule sur les résumés passés (pondérée par leur "routabilité", aucun paramètre neuf) :

```
w  = Z.max(dim=-1).values.reshape(B, n·r)         # score max-page de chaque résumé
A  = softmax( w.unsqueeze(1) + M_causal, dim=-1 ) # (B,n,n·r), M[i,(j,·)] = -inf si j ≥ i
c  = A @ U.reshape(B, n·r, d)                     # (B,n,d) ; ligne i=0 → zéro (nan_to_num)
```

c_i ≈ "le résumé passé qui route le plus fort" ≈ l'embedding de l'entité. Coût O(B·n²·r·d),
négligeable (n ≤ 64).

### 2.4 Fetch + lecture fine par token (identique v1, requête enrichie)

```
Kb, Vb = K_slots[pidx], V_slots[pidx]                    # (B,n,p·S,d)  SEUL point d'I/O
q  = ( W_qtok(hb) + W_c(c).unsqueeze(2) ).reshape(B·n, C, d)   # W_c: Linear(d,d)
sc = bmm(q·d^-½, Kb^T) + τ·s_p[broadcast par page]       # (B·n, C, p·S)
o  = bmm(softmax(sc, -1), Vb)                            # (B·n, C, d)
```

Division du travail : la **page** = quelle entité (via s_p + c_i), le **slot** = quel
fragment de fait (ex. quel chiffre de `#vYYYY`, discriminable par l'indice positionnel local
présent dans h_t). C'est le point-clé demandé : la sélection fine par token est **précise ET
gratuite en disque**, le banc étant chargé.

### 2.5 Gate conditionnel par token (répare le LM)

```
g = sigmoid(h @ w_g + b_g) * sigmoid(α)     # (B,T,1) ⊙ (d,)   w_g: Linear(d,1), b_g init −4, α init 0
x = x + g ⊙ (W_o(o).view(B,T,d)) * seg0_mask
```

Sur TinyShakespeare la mémoire n'aide pas → w_g apprend g_tok≈0 partout → ≡ #14 (parité
structurelle, test α/b_g → −20 conservé). Sur FactMix g_tok s'ouvre aux positions de rappel
(`=v…`). Init produit ≈ 0,009 : même régime "quasi fermé mais gradient vivant" que v1.

## 3. Ce qui ne change pas

Aux loss d'équilibrage (calculée sur `route`, segments ≥1, λ=1e-4), un seul AdamW
(weight_decay=0 sur K_pages/K_slots/V_slots), chemin d'inférence par boucle sur segments
(remplacer la construction de q par §2.1–2.3, l'état incrémental = `(Zcum_{i-1}, U_{≤i-1}, w_{≤i-1})`,
O(P+n·r·d) en RAM), PageStore/memmap/LRU, causalité (résumés du segment j n'utilisent que le
segment j ; décalage + cummax strictement passés ; segment 0 masqué), audits __main__ v1.

Paramètres : −W_qseg(2d·d) ; +W_sal(d·r) +W_c(d·d) +w_g(d+1) → net ≈ −4 K params.
Ops lourdes : 4 sgemm + 2 bmm (comme v1) + 1 bmm (B·n,r,C)×(B·n,C,d) pour U + cummax(B,n,P).

## 4. Budget I/O (l'invariant)

| | froid pire cas | attendu (cummax + LRU) |
|---|---|---|
| accès/segment | p = 3 | ~0–1 (pages persistantes) |
| **accès/token** | **p/C = 0,047 ≤ 0,05 ∀T** | ~0,01 (v1 mesuré : 0,023 avec p=2) |

Si rappel < 90 % : passer p=4 (0,0625 pire cas mais mesuré ≪ grâce à la monotonie du
cummax) AVANT de toucher au reste.

## 5. Tests supplémentaires (en plus des 6 de v1)

1. **Test de survie de l'entité** (l'échec exact de v1) : h synthétique = bruit + 1 token
   "entité" injecté dans le segment 0 ; vérifier que `route[·, 3]` et `c_3` changent quand
   l'entité change (‖Δ‖ > 0,1·‖route‖), et PAS quand seul le remplissage change de graine.
2. **Gate conditionnel** : variance de g_tok sur les tokens > 0 après entraînement FactMix ;
   g_tok moyen sur remplissage < g_tok moyen sur positions `=v`.
3. Causal : audits v1 + perturber le segment i ne change ni `pidx`, ni `c`, ni `route` des
   segments ≤ i.
4. Accès : assertion `miss/token < 0,05` à froid, T ∈ {256, 512}.

## 6. Critères de succès

| métrique | harnais | cible v2 | v1 |
|---|---|---|---|
| FactMix chiffre | `mem17_factmix.py` (M=1000, SEQ=256) | **≥ 90 %** | 11,6 % |
| ppl LM | `mem17_lm_compare.py` | **≤ 7,9** (parité #14) | 13 |
| accès HDD/token à froid | compteurs | **< 0,05** | 0,023 ✓ |

**Pourquoi v2 réussit où v1 a échoué, en une phrase** : v1 n'avait aucun chemin par lequel
l'identité de l'entité pouvait atteindre la récupération (moyennes qui noient, requête fine
aveugle) ; v2 fait porter l'entité par des tokens saillants poolés en soft-top-1, propagés
causalement par cummax (routage) et par c_i (sélection fine), pendant que le gate par token
éteint la branche partout où elle n'apporte rien — le tout sans un seul accès disque de plus
que le pire cas p/C.
