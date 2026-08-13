# H8 — Detection improves with poison rate

**Status: INCONCLUSIVE. The trend is attack-dependent and reverses sign.**

## Claim

PSBD AUROC increases with the poison rate used at training time, across
{0.01, 0.05, 0.1}, because more poisoned samples means a more strongly reinforced
trigger-to-target path.

## Evidence

AUROC at the 25th-percentile threshold, best rate, pre-residual, CIFAR-10 ViT:

| attack | 1% | 5% | 10% | trend |
|---|---|---|---|---|
| `badnet_a2o` | 0.686 | 0.872 | 0.889 | **rises**, strongly |
| `lf` | 0.887 | 0.945 | 0.947 | rises |
| `bpp` | 0.964 | 0.995 | 0.977 | non-monotone |
| `blend` | 0.986 | 0.980 | 0.978 | **falls** slightly |
| `badnet_a2a` | 0.601 | 0.539 | 0.510 | **falls**, toward chance |

Two attacks rise, one is flat-to-falling, one is non-monotone, one falls
substantially. There is no single trend.

## Reading

The rise for `badnet_a2o` is the largest effect (+0.20) and it is also the attack
with the *lowest* baseline detectability, so the pattern is better described as
**regression toward a ceiling** than as a poison-rate law. `blend` and `bpp` are
already at 0.96 to 0.99 at 1% poisoning and have nowhere to go.

That reframing matters for the deployment claim: the hardest case for a defender is
a low poison rate, and at 1% poisoning three of four attacks are already detected at
0.89 or better. The exception is `badnet_a2o` at 0.686, which is the case worth
reporting rather than the 10% column.

`badnet_a2a` falling toward chance as poisoning increases is a real inversion and is
explained separately in [H5](H5-all-to-all-breaks-psbd.md): more all-to-all
poisoning means more source classes with well-learned but mutually cancelling
mappings, so the mean backdoor direction cancels harder.

## Confound worth stating

ASR is already at or near 1.0 for `badnet_a2o`, `blend`, `bpp` and `lf` at *every*
rate on CIFAR-10, so poison rate here varies the reinforcement of an already
saturated backdoor rather than whether the backdoor exists. A grid where ASR itself
varies (WaNet spans 0.12 to 0.96 across rates) would separate "more poison" from
"stronger attack", and this one cannot.

## Separate finding, discovered while checking this

Poison rate does **not** mean what the folder name says for clean-label attacks.
`poison.choose_poison_indices` caps the count at the number of eligible samples, and
clean-label eligibility is the target class only. On CIFAR-100 that is 500 images,
so 1%, 5% and 10% all resolve to the same 500 poisoned samples: three folders, one
experiment. `args.json` records the *requested* rate with no warning. Confirmed by
ASR: `vit_cifar100_sig_0_01` 0.250 against `_0_05` 0.252.

`sig` and `lc` are excluded from this grid for unrelated reasons (weak ASR on ViT),
so no result here is affected. It does invalidate any poison-rate trend anyone draws
for clean-label attacks on CIFAR-100, and it should be fixed by recording a
`realized_poison_rate` alongside the requested one.

## Subquestions

1. Rerun on WaNet, whose ASR genuinely varies with poison rate, to separate poison
   rate from attack strength.
2. Is the apparent trend just AUROC ceiling compression? Plotting against
   `logit(AUROC)` would say.
3. Does 0.005 (trained, excluded here for weak ASR) extend the `badnet_a2o` trend
   downward? It is the only attack with room to move.
