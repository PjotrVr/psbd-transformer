# H10 — Aiming dropout at the depth where an attack's direction lives beats spreading it over all blocks

**Status: the band premise is SUPPORTED (6/6). The onset-based selection rule is
REFUTED by a pre-registered out-of-sample test, 0/2.**

Restricting pre-residual dropout to a band of blocks always beats applying it to all
12. But the onset layer does **not** predict which band, in either direction, and the
rule that appeared to work was fitted to four points.

## What survives

**A band beats all-blocks on every attack tested, 6 of 6:**

| attack | set | onset | blocks 1-4 | blocks 5-8 | blocks 9-12 | all 12 | best |
|---|---|---|---|---|---|---|---|
| `blend` | derivation | 5 | 0.969 | 0.981 | **0.997** | 0.978 | 9-12 |
| `bpp` | derivation | 6 | 0.982 | 0.985 | **0.996** | 0.977 | 9-12 |
| `wanet` | **held out** | 7 | 0.931 | **0.976** | 0.975 | 0.932 | 5-8 |
| `lf` | derivation | 8 | 0.925 | 0.988 | **0.998** | 0.947 | 9-12 |
| `badnet_a2o` | derivation | 9 | **0.912** | 0.907 | 0.829 | 0.889 | 1-4 |
| `adaptive_blend` | **held out** | 12 | 0.762 | **0.920** | 0.569 | 0.784 | 5-8 |

Gains over all-blocks range from +0.019 to +0.136. It is also cheaper: dropout in 4
blocks instead of 12.

## What was refuted, and how

After the four derivation attacks, the pattern looked clean: attacks whose direction
is written early were best caught late, and the one written late was best caught
early. The inverted rule "perturb as far as possible from the onset layer" fitted all
four.

It was pre-registered against two held-out attacks. Onsets were measured first, the
predictions written to `docs/runs/2026-08-13-h10-out-of-sample.md`, and only then were
the jobs run.

| checkpoint | onset | predicted | actual best | verdict |
|---|---|---|---|---|
| `wanet` | 7 | blocks 9-12 (0.975) | blocks 5-8 (0.976) | FAIL by 0.001 |
| `adaptive_blend` | 12 | blocks 1-4 (0.762) | blocks 5-8 (0.920) | **FAIL by 0.158** |

**0 of 2.** The `wanet` miss is a tie and means little on its own. The
`adaptive_blend` miss does not: the rule predicted the *worst* useful band and missed
the best by 0.158. That checkpoint was the sharper test by design, because its onset
of 12 forced the rule to extrapolate past the derivation range.

So the onset-to-band mapping was an artifact of four data points. Four attacks,
three bands, and a rule with a free direction is not enough evidence for a
mechanism, and it took a deliberate held-out test to see it.

## What to use instead

**Blocks 5-8 is the robust default.** Over all six attacks it has the best mean rank
(1.67 against 1.83 for blocks 9-12 and 2.50 for blocks 1-4) and, unlike either
alternative, it is **never worse than second**. Blocks 9-12 wins more often (3 of 6)
but collapses badly when it loses: 0.829 on `badnet_a2o` and 0.569 on
`adaptive_blend`, both below all-blocks.

For a defender who does not know the attack, "never worse than second" is the property
that matters, so the recommendation is blocks 5-8 rather than the higher-variance
blocks 9-12.

## What is still unexplained

Why a middle band should be robust across attacks with onsets from 5 to 12 is not
accounted for by any mechanism proposed here. The honest position is that the band
effect is real and reproducible, and its cause is not yet known. Candidates worth
testing, none of them tested:

- Blocks 5-8 may simply be where clean class evidence is most concentrated,
  independent of where any backdoor is written.
- Late-block perturbation may be too close to the readout: whatever it damages, the
  model has no depth left to recover from, so clean and backdoor collapse together.
- The band effect may be about *how much* of the network is perturbed rather than
  *where*, with 4 of 12 blocks being near an optimum that the specific band only
  weakly modulates.

## Reproduce

```bash
PYTHONPATH=. python pbs/generate_psbd_jobs.py --phase depth_bands
python psbd_analyze.py --all
```

## Subquestions

1. The third candidate above is directly testable and would be decisive: sweep band
   *width* (2, 4, 6, 8 blocks) at a fixed location. If width explains most of the
   effect, location is a red herring.
2. Does blocks 5-8 stay best under SAM and at other poison rates?
3. `badnet_a2a` at blocks 9-12 scores 0.159, a detector running backwards. That is
   explained in [H5](H5-all-to-all-breaks-psbd.md), and its two-sided score of 0.966
   makes blocks 9-12 that attack's *best* band, which the table above does not show
   because it reports one-sided numbers.
