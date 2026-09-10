# Confidence, the max-softmax null model

Confidence is the maximum softmax probability the model assigns to any class, negated so low means poisoned. There is no paper behind it and no released code to port. It exists as the floor every other detector in this registry has to clear, because a method that cannot separate itself from a single forward pass and a max is reading calibration rather than detecting backdoors. This page records the statistic, why it earns a place in the registry despite citing nothing, what the port under `detectors/confidence.py` does, and where it currently stands.

## Why the null model

A backdoored model is typically more confident on a triggered input than on a clean one, since the trigger drives the logits toward the target class through a shortcut the ordinary class boundary does not offer. That gap is real, and it is also the first thing every published detector implicitly assumes is not enough on its own. Section 2.2 of `docs/attack-design/cross-defence.md` measured it directly against this project's own PSBD score: the null model beats PSBD on the benign control, so a detector that reports an AUROC below confidence's has not shown that its extra machinery buys anything, and an attacker who defeats the null model has already removed a baseline the comparison table depends on.

Keeping it in the registry is a discipline rather than an oversight. Every table `cli.compare_detectors` produces carries the `confidence` column beside the others, so a reader can see at a glance whether STRIP, SCALE-UP, IBD-PSC, TeCo, CD-L or Beatrix earned their extra forward passes on a given cell or merely reproduced what a max and a comparison already gave for 1 forward pass.

## Mechanism

    original form
        s(x) = max_c P(c | x, theta)

    descriptive form
        confidence = the largest softmax probability the model assigns to any class

| Symbol | Meaning |
|---|---|
| $x$ | the input image |
| $\theta$ | the model's parameters |
| $c$ | a class index |
| $P(c \mid x, \theta)$ | the softmax probability the model assigns to class $c$, given $x$ and parameterized by $\theta$ |
| $s(x)$ | the confidence statistic, the largest of those probabilities |

A well trained classifier is confident on inputs whose class evidence is clear and uncertain on inputs near a decision boundary. A backdoor is engineered to be unambiguous once its trigger is present, so it usually pushes the target-class logit far above the rest and the prediction is confident by construction. Nothing about $s(x)$ asks why the prediction is confident, so any legitimate source of high confidence, an easy image, a large object, a well separated class, reads identically to a trigger.

## What this port does on ViT

There is no paper to deviate from and no released code to compare against, so this section is short by construction rather than by omission. `confidence_scores` in `detectors/confidence.py` runs 1 forward pass per input under the shared autocast policy, takes the row-wise maximum of the softmax output and negates it. The negation is the only place a sign error could hide, and `experiments/preflight/check_signs.py` exists to catch exactly that class of mistake across every detector in the registry, confidence included.

## Hyperparameters

None. The statistic has no scaling set, no ensemble, no threshold of its own and no clean-data budget beyond the shared quantile rule every detector's report is read at.

## Cost

1 forward pass per input, the cheapest entry in `FORWARD_PASSES_PER_INPUT` and the reference every other detector's cost is stated relative to. Fitting is a no-op, so `confidence` is absent from `NEEDS_FITTING` and its record's `runtime_seconds["fit"]` is always near 0.

## How to run

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors confidence --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, beside STRIP, SCALE-UP, IBD-PSC and Beatrix, since a 1-forward-pass detector never sizes a job on its own.

## Where results land

`results/<folder>/detectors/confidence_metrics.json` holds the detection report at every quantile plus the provenance record. The raw per-sample scores sit beside it as `confidence_scores_validation.pt`, `confidence_scores_clean.pt` and `confidence_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order and already negated.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low score says the model was not very sure of its own prediction, which is what a hard clean image looks like just as much as what a poisoned one can look like on a model whose backdoor was trained at low strength or is competing against a confident clean prediction. On the synthetic fixture in `experiments/preflight/synthetic.py`, where the trigger drives the target logit far above the rest by construction, `python -m experiments.preflight.check_signs` reads confidence at AUROC 1.0000. That number reflects an unmissable backdoor and says nothing about a real checkpoint, where the gap between a confident clean prediction and a confident triggered one can be small.

The mechanism-level failure is a backdoor placed deliberately near the decision boundary instead of far from it. `docs/attack-design/A5-low-confidence-backdoor.md` works through the shape of this on the project's own perturbation-based score and, as a byproduct, states the same point for confidence directly: a poisoned sample capped at $p_c = 0.3$ reads as less confident than the median clean sample, so the raw statistic ranks it as more benign than most of the clean population. That analysis is this project's own theoretical prediction rather than a measurement on a trained checkpoint, and its own PSBD conclusion carries a later correction (A21), so the confidence row should be read as a mechanism argument, not a verified number. A published attack in the same family, Peng et al.'s under-confidence backdoor (LSBA, arXiv:2202.11203), caps the target posterior at about 0.6 by construction and reports its measured effect against STRIP rather than confidence directly, so the confidence-specific claim above remains a prediction until it is run.

## Direction

Low is poisoned. The raw statistic, the largest softmax probability, is usually high for a triggered input, so `confidence_scores` negates it once at the return boundary. A second negation anywhere downstream would produce a well-formed, exactly inverted detector, which is the failure `experiments/preflight/check_signs.py` and the `auroc_two_sided` diagnostic in `defences.decision.detection_report` both exist to catch.
