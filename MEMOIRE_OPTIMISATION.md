# Mémoire d'optimisation — qwen2.5:14b

## État actuel (résumé, mis à jour à chaque fois)
- **Config qui fonctionne le mieux à ce jour** : `num_thread=2` (offload GPU Vulkan auto
  déjà actif via OLLAMA_VULKAN=1 ; flash-attention ON ; KV-cache q8_0).
  Les autres PARAMETER par défaut sont conservés (num_ctx=2048, num_gpu=auto).
- **Vitesse mesurée** : **~1,02 tok/s** (stable sur plusieurs runs, génér. chaude).
  Baseline défauts = 0,79 tok/s -> gain +29 % via num_thread=2 seul.
- **RAM/VRAM observées** : RAM 11 Gi total / ~6,9 Gi dispo repos / 4 Gi swap.
  Au run : ~1,1 GiB sur le 930M (Vulkan), ~7,9 Go de poids en RAM+swap => **swap actif (~1 Go)**.
  Mesures MemAvailable bruitées (swap), mais VRAM stable à ~1112 MiB = preuve d'offload.
- **Qualité de raisonnement** : OK, non dégradée. Test identique ("pourquoi le ciel est bleu")
  donne à chaque run une explication correcte de la diffusion de Rayleigh (~95-145 tok, cohérent).

## Diagnostic (Étape A, 2026-07-18)
- Machine : Intel i5-5200U (2 cœurs physiques / 4 threads logiques), AVX2 (pas AVX-512).
- RAM : 11 Gi total, ~6,9 Gi dispo au repos, 4 Gi swap (511 Mi utilisés au repos).
- VRAM : GeForce 930M 2004 MiB (Maxwell 2014), Vulkan dispo.
- Ollama 0.30.2. Tag présent : `qwen2.5:14b` (9,0 Go). **Aucun autre quant 14B en local**
  (le seul autre modèle est qwen2.5-coder:3b, modèle différent ; un 7B-coder gguf traîne
  dans ~ mais c'est un autre modèle).
- Modelfile : aucun PARAMETER -> défauts Ollama (num_ctx=2048, num_thread=4, num_gpu=auto).
- Env daemon DÉJÀ active : OLLAMA_VULKAN=1, OLLAMA_FLASH_ATTENTION=1,
  OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_KEEP_ALIVE=5m.
  => Flash-attention et KV-cache q8_0 SONT DÉJÀ par défaut. Leviers 3 et 7 pré-appliqués.
- Contrainte dure : 9 Go de poids > 6,9 Go dispo => **swap inévitable** au chargement.

## Historique des tentatives
### 2026-07-18 — Essai 0 (BASELINE, défauts Ollama)
- Paramètre : aucun (options {}).
- Résultat : 0,79 tok/s (chaude, run2), VRAM 1112 MiB. Stable mais très lent.

### 2026-07-18 — Essai 1 : num_thread=2
- Avant -> après : défaut (4 threads logiques) -> 2 (cœurs physiques).
- Résultat : **1,02 tok/s**, VRAM 1112 MiB. -> **GAIN +29 %**.
- Conclusion : GARDER. Hyperthreading (4 threads) surcharge le cache/bande passante mémoire
  sur ce CPU 2 cœurs -> 2 threads physiques est optimal.

### 2026-07-18 — Essai 2 : num_thread=2 + num_ctx=1024
- Résultat : 0,98 tok/s. Contexte plus court ne change pas la vitesse de génération
  (goulot = poids 9 Go en bande passante mémoire, pas KV-cache).
- Conclusion : rejeter (pas de gain vitesse ; réduit juste la RAM KV, négligeable ici).
  Garder num_ctx=2048 par défaut pour ne pas tronquer les échanges.

### 2026-07-18 — Essai 3 : + kv_cache_type=q4_0
- Résultat : 1,01 tok/s. Pas de gain de vitesse mesurable.
- Conclusion : rejeter (q8_0 par défaut suffit ; q4_0 n'apporte rien ici, risque qualité).

### 2026-07-18 — Essai 4 : num_gpu=0 (CPU pur, pas d'offload Vulkan)
- Résultat : 0,96 tok/s, VRAM 0, ~8 Go en RAM/swap.
- Conclusion : rejeter. L'offload Vulkan sur le 930M (même faible) est *légèrement* meilleur
  (1,02 vs 0,96) et soulage la RAM. Garder num_gpu=auto (défaut).

### 2026-07-18 — Essai 5 : num_thread=2 + num_ctx=512
- Résultat : 1,01 tok/s.
- Conclusion : rejeter (même constat que Essai 2 : ctx n'impacte pas la gen).

## Problèmes identifiés (à ne jamais refaire deux fois)
- **Le 14B q4 (9 Go) ne rentre pas dans 11 Go RAM sans swap** : ~7,9 Go doivent être en
  RAM après offload GPU ; avec ~6,9 Go dispo + 4 Go swap, ~1 Go swappé en continu.
  C'est le vrai goulot : chaque token relit les 9 Go de poids via la bande passante mémoire.
- **Aucun réglage d'inférence ne peut faire "rapide" un 9-Go-sur-11-Go** : num_thread/ctx/kv
  ne font que gratter quelques %, pas changer l'ordre de grandeur. Vérifié par mesure.
- Hyperthreading (4 threads) NUIT sur ce CPU 2 cœurs (0,79 vs 1,02). Toujours fixer num_thread=2.

## Pistes écartées / non applicables (honnêteté)
- Pas de re-quantification plus agressive possible : pas d'outil `quantize` local vérifié,
  pas de poids F16 source du 14B sur le disque (le gguf présent est le q4_K_M lui-même).
  Téléchargement interdit par consigne.
- Pas d'architecture "block-local" greffable : qwen2.5:14b est pré-entraîné, poids figés
  (voir RAPPORT_VICTOIRE §Honnêteté : validation seulement à dim<=256, pas une preuve
  pour un 14B). Interdit + hors de portée. Non tenté.
- mlock : à éviter (verrouillerait ~9 Go en RAM, pire que le mmap actuel). Non tenté.

## Prochaines étapes / alternative à présenter à l'utilisateur
Pour une VRAIE fluidité, seules 2 issues (hors consigne de non-téléchargement) :
1. **Modèle plus petit déjà local** : `qwen2.5-coder:3b` (1,9 Go) tournera de façon fluide
   (quelques tok/s, pas de swap). Pour du généraliste, il faudrait un 7B (non présent en local,
   à télécharger -> bloqué).
2. **Re-quantification q2_K / q3_K** du 14B : nécessite outil local + source F16 -> indispo ici.
Le compromis retenu (num_thread=2, offload Vulkan auto) est le **meilleur atteignable sur cette
machine sans changer de modèle** : ~1 tok/s stable et cohérent, pas de freeze.
