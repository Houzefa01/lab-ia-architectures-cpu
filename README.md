# Lab IA — architectures alternatives au Transformer sur CPU

Laboratoire de recherche personnel (PyTorch, CPU uniquement) : trouver une architecture
de réseau de neurones qui **bat le Transformer causal** (attention softmax) sur CPU,
en particulier à **contexte long** (T ≥ 1024), avec à terme l'objectif d'un grand modèle
exécutable sans GPU.

## Protocole

Chaque candidat est évalué à taille fixe (dim=64, heads=2, layers=3, ffn=256) sur :

1. **Copy task** : rappel associatif longue distance (gap = 186, seq_len = 192)
2. **Perplexité LM** : TinyShakespeare (niveau caractère)
3. **Vitesse CPU** : 7 longueurs de 48 à 4096, batch 32, médiane sur 5 runs
4. **Causalité stricte** : audit de fuite d'information future (diff < 1e-6)
5. **Gradients** : tous non-nuls et sans NaN

Critères étendus : mémoire sous-quadratique, généralisation de longueur,
robustesse à la quantization int8/int4, loi d'échelle. Voir `objectif.md` et `test.md`.

## Résultats

| Candidat | Verdict |
|---|---|
| KVMemory v1–v3, CDCN, CDCN-Attn | Échec du rappel associatif |
| RetNet chunké (γ = 0.999) | Copy 96,4 %, 2,7× plus rapide à T = 1024, perplexité légèrement moins bonne |
| RetNet multi-gamma | Bat la perplexité de référence, mais copy à 87,6 % |
| Mamba sélectif, GLA, Sparse Clustered, Sliding Window | Trop lent ou rappel insuffisant sur CPU |
| **#14 Block-Local Attention** | **Vainqueur** : copy 100 %, perplexité identique, **4,43× à T = 1024, 10,85× à T = 2048**, seul à tourner à T = 4096 (le Transformer sature la RAM) |

Détails : `RAPPORT_VICTOIRE.md` (Block-Local), `RAPPORT_FINAL_NEGATIF.md`
(premières 13 architectures), `MEMOIRE.md` (journal de mission).

Le dépôt contient aussi deux journaux d'optimisation de l'inférence locale
**qwen2.5:14b via Ollama/Vulkan** sur GeForce 930M (`MEMOIRE_OPTIMISATION.md`,
`MEMOIRE_30TPS.md`).

## Organisation

```
baseline/               Transformer de référence
candidats_actifs/       architectures testées (block_local, retnet, mamba, gla…)
candidats_echoues/      candidats abandonnés + RAISONS_ECHEC.md
candidats_prometteurs/
benchmark*.py           benchmarks copy / LM / vitesse / mémoire
resultats/              mesures JSON + graphiques
docs/                   spécifications des candidats
data/tinyshakespeare.txt
```

## Lancer un benchmark

```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy psutil matplotlib
python benchmark_v3.py            # exemple
python benchmark_block_local.py   # candidat vainqueur
```

> Les gros fichiers de mémoire (`resultats/*.dat`) ne sont pas versionnés.
