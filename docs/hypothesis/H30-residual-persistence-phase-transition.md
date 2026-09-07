# H30 — Backdoor direction shows a phase transition, not uniform persistence

**Status: SUPPORTED with a qualification.** The direction is NOT uniformly
persistent. It crystallizes around layers 8 to 10, with a clear S-curve in
alignment to the final layer.

Evidence: `experiments/backdoor_direction/direction_persistence.py`, results in
`results/direction_persistence.json`.

## Claim

The backdoor direction is carried through the residual stream with minimal
rotation. Cosine similarity between directions at consecutive layers (L, L+1)
should be high (> 0.8), and the direction should be stable from early layers
onward.

## What actually happened

The direction does NOT persist uniformly from the start. Instead it shows a
phase transition:

1. **Early layers (1 to 6):** consecutive cosine is 0.3 to 0.7, and alignment to
   the final layer direction is near zero (0.02 to 0.19). The backdoor direction
   at these layers is nearly orthogonal to the final direction.

2. **Transition zone (7 to 9):** rapid increase. Alignment to final jumps from
   0.19 to 0.49. The direction is being assembled.

3. **Late layers (10 to 12):** consecutive cosine stabilizes at 0.82 to 0.95.
   Alignment to final reaches 0.73 to 0.95. The direction is now persistent.

## Representative numbers (CIFAR-100 badnet_a2o at 10%)

| layer pair | consecutive cosine | alignment to layer 12 |
|---|---:|---:|
| 1 to 2 | 0.625 | 0.060 |
| 3 to 4 | 0.724 | 0.090 |
| 5 to 6 | 0.509 | 0.158 |
| 7 to 8 | 0.866 | 0.406 |
| 9 to 10 | 0.832 | 0.674 |
| 10 to 11 | 0.943 | 0.833 |
| 11 to 12 | 0.855 | 0.855 |

## Attack-specific patterns

**Blend crystallizes earlier than badnet.** Blend's alignment to final at layer
8 is 0.40 to 0.56 across poison rates, while badnet's is 0.27 to 0.41. This
matches the trigger geometry: blend's global trigger affects all tokens from the
first convolution projection, giving it a head start.

**Poison rate does not affect crystallization depth.** Even at 0.5% poison rate,
the phase transition occurs at the same layers (8 to 10). The direction norm is
smaller at lower rates but the angular structure is the same.

## Implications

1. **Pre-residual dropout works because it disrupts the skip path in the
   crystallization zone.** The direction is being written into the residual
   stream by late-layer branches, and once written it persists through skip
   connections. Dropout before the residual add disrupts this writing process.

2. **The 159x norm growth is cumulative.** Each late-layer block adds a small
   aligned increment to the direction (consecutive cosine > 0.8), and these
   accumulate through the skip connections. This is the residual stream acting as
   an information highway (Raghu et al. CKA homogeneity), specialized for the
   backdoor signal.

3. **Weight orthogonalization (H34) fails on strong attacks because the
   direction rides the skip path.** Once written into the residual stream, the
   direction persists through layers even after the branch weights are
   orthogonalized. Only inference-time direction removal (H16) works, because it
   strips the direction from the stream itself.

4. **Detection perturbations should target layers 8 to 12.** This is where the
   direction crystallizes. Earlier perturbations are wasted because the direction
   has not formed yet.
