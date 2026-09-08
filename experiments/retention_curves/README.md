# Is the backdoor the robust path in every perturbation modality? No. (H46)

## Question

Every detector in the PSBD family rests on one asymmetry: under perturbation the backdoor
pathway survives and the clean pathway degrades. H28 shows PSBD, STRIP, SCALE-UP and
IBD-PSC are one family in which the operator only selects a Jacobian, and this project has
measured that asymmetry under activation dropout, gradient-guided head-weight ablation,
weight interpolation toward the pretrained initialization, and prediction depth.

Four probes, all pointing the same way, invites the general claim: **the backdoor is the
robust path in every modality**. That claim is false, and this measures where it breaks.

## The common statistic

Modalities are not comparable at a shared nominal strength, so everything is read at
**matched clean damage**. For each modality sweep the disturbance and record

    ASR retention = ASR(d) / ASR(0)
    CA retention  = clean accuracy(d) / clean accuracy(0)

then report ASR retention at the disturbance where CA retention first falls to 0.75. The
backdoor is the more robust path exactly when ASR retention exceeds CA retention there.

## Result

GTSRB, 5% and 10%, attacks that actually implanted (ASR >= 0.5).

| modality | what is perturbed | ASR retention | CA retention | backdoor more robust |
|---|---|---|---|---|
| **activation dropout** | token activations at `before_attention_norm` | **0.910** | 0.527 | **12 / 14** |
| **Fourier amplitude** | the input's texture, phase preserved | 0.627 | 0.718 | **2 / 5** |

**The asymmetry is not universal. It is specific to activation-space perturbation, and it
reverses in the input-frequency modality.**

The 2 cells that break the dropout column are both `badnet_a2a`, at ASR retention 0.190 and
0.151 against clean retention 0.470 and 0.709. That is H5's inversion appearing as an
internal control: the one attack family whose trigger-to-target map is content-dependent is
the one whose backdoor is the *fragile* path.

## The sharpest sub-result: adaptive_blend is anti-fragile

| cell | modality | ASR retention |
|---|---|---|
| `vit_gtsrb_adaptive_blend_0_05` | dropout | **1.282** |
| `vit_gtsrb_adaptive_blend_0_1` | dropout | **1.192** |
| `vit_gtsrb_adaptive_blend_0_05` | Fourier amplitude | **1.125** |

Retention above 1 means the attack gets **more** effective as the model is damaged, in 2
mechanistically unrelated modalities. Adaptive-Blend exists to suppress latent separability
and it is the hardest attack in this panel for every detector measured here; this says why.
Damaging the model damages the carrier's own class evidence faster than the trigger's, so
the trigger wins more often, and a detector that reads "how much did the prediction move"
is reading a quantity that moves the wrong way.

## Consequences

1. A defence built on perturbation consistency is exploiting an asymmetry that exists in
   one modality. That is not a flaw in itself, but it bounds what the family can do and it
   predicts the failure on Adaptive-Blend rather than discovering it.
2. Input-space purification is not simply a weaker version of activation perturbation. The
   sign of the asymmetry differs, which is a mechanism for the failure recorded in
   [H45](../style_content_split/README.md).
3. The generalisation "the backdoor is always the robust path" was proposed by the
   orchestrator and is refuted here by its own measurement. It survives only for
   activation-space probes.

## What is not established

- GTSRB only, 2 modalities, 1 architecture, single seed. The weight-space modalities
  (interpolation toward pretrained, gradient-guided ablation) were measured on single cells
  and are not in the table because they were not swept to matched clean damage.
- The Fourier column is 5 cells, of which 2 support and 3 do not. It establishes that the
  asymmetry is not universal; it does not establish its sign in that modality.

## Reproduce

    PYTHONPATH=. python experiments/retention_curves/measure.py
