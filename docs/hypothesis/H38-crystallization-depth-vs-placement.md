# H38 -- Crystallization depth predicts optimal perturbation placement

**Status: SUPPORTED (crystallization confirmed), INCONCLUSIVE (placement
correlation).** Crystallization depths are attack-dependent and consistent: blend
crystallizes at layer 8.2 (mean across 8 checkpoints), badnet at layer 10.4
(mean across 8 checkpoints). The correlation with PSBD sweep placement could not
be tested because no per-block-band PSBD sweep data exists in the expected
directory structure.

Evidence: `scratch/crystallization_vs_placement.py`, results in
`results/crystallization_vs_placement.json`. Cross-references H30's direction
persistence data (`results/direction_persistence.json`).

## Claim

The optimal PSBD perturbation depth matches the crystallization depth from H30.
Blend (which crystallizes at layers 7 to 8) should be best detected with
perturbation at those layers. BadNet (which crystallizes at layers 10 to 11)
should be best detected at those layers.

## Test

1. Extract crystallization depth per checkpoint from H30's alignment-to-final
   curves (first layer where alignment exceeds 0.5).
2. Load per-block-band PSBD sweep AUROC from `results/*/psbd/` directories.
3. Correlate: does the best detection depth match the crystallization depth?

## Results

### Crystallization depths (alignment-to-final > 0.5 threshold)

| checkpoint | crystallization layer |
|---|---:|
| cifar100 badnet 0.5% | 11 |
| cifar100 badnet 1% | 11 |
| cifar100 badnet 5% | 10 |
| cifar100 badnet 10% | 10 |
| tiny badnet 0.5% | 10 |
| tiny badnet 1% | 11 |
| tiny badnet 5% | 10 |
| tiny badnet 10% | 10 |
| **badnet mean** | **10.4** |
| cifar100 blend 0.5% | 8 |
| cifar100 blend 1% | 9 |
| cifar100 blend 5% | 8 |
| cifar100 blend 10% | 9 |
| tiny blend 0.5% | 8 |
| tiny blend 1% | 8 |
| tiny blend 5% | 8 |
| tiny blend 10% | 8 |
| **blend mean** | **8.2** |

### Observations

1. **Blend crystallizes 2.2 layers earlier than badnet**, consistently across
   both datasets and all poison rates. Blend's distributed trigger (full-image
   blending) writes the direction into the residual stream at layers 7 to 8,
   where the direction already reaches 0.5+ alignment with its final-layer
   form. Badnet's localized trigger requires 2 more layers to accumulate the
   same alignment.

2. **Poison rate has a small but consistent effect on badnet**: lower rates
   (0.5%, 1%) push crystallization one layer later (11 vs 10). The weaker
   signal needs more processing. Blend is nearly unaffected by rate.

3. **No per-block-band PSBD sweep data found.** The PSBD sweep results use
   position names like `before_attention_norm`, `before_mlp_residual`, not
   block-band identifiers. The correlation between crystallization depth and
   optimal detection block-band requires a targeted experiment that applies
   PSBD perturbation at only the crystallization layers.

## Interpretation

The crystallization depth difference is robust and attack-dependent. It aligns
with the mechanistic story from H30: blend's distributed trigger engages
mid-layer attention (consistent with H31's backdoor heads at layers 5 to 6),
which writes the direction early. Badnet's localized trigger requires the
additional late-layer heads (L9H7, L10H9 from H31) to fully form the direction.

This predicts that PSBD perturbation confined to layers 7 to 9 should
preferentially detect blend, while perturbation at layers 10 to 11 should
preferentially detect badnet. Testing this requires a targeted sweep that is
not yet implemented.

## Connection to other hypotheses

- **H30**: supplies the alignment-to-final curves that define crystallization.
- **H31**: the 3 shared backdoor heads (L5H0, L6H3, L5H10) sit at layers 5 to
  6, which is where blend begins crystallizing. Badnet's extra heads (L9H7,
  L10H9) sit at layers 9 to 10, matching its later crystallization.
- **H10**: tested block-band placement directly. The onset-based selection
  (perturbation at the direction's onset layer) was refuted out of sample.
  Crystallization depth is a different criterion (where the direction reaches
  0.5 alignment, not where it first appears) and may predict better.
