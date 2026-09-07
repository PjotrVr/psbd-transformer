# Adaptive attacks and dropout interaction: design

Three experiments that the perturbation study opens but does not answer. Written
before implementation so the evaluation contract is fixed in advance.

All three obey the project's standing rules: one-sided scoring (low PSU means
poisoned, a value below 0.5 is a failure and is never re-signed), comparison at
matched clean-validation shift ratio rather than matched rate, and a benign
control in every table.

---

## E1. Defender rate grids that actually bracket the usable window

**Status: partly done.** The calibration run
(`docs/runs/2026-08-14-perturbation-calibration.md`) showed disturbance per rate
varies 11x across operators at a shared p = 0.5, and that `channel_mask` on
residual-stream positions saturates below p = 0.1. The fine grid for those
positions is submitted (`Efine_001`).

**Remaining.** Two operators still have unbracketed ends:

- `head_mask` reaches only sigma 0.762 at p = 0.6 and the grid stops at 0.9.
  If p = 0.9 does not reach sigma 0.8, head masking **cannot** be compared to
  other operators at the study's usual matched target, and that limitation has to
  be reported rather than worked around. Extend to 0.95 and 0.98 to find out.
- `gaussian` jumps from sigma 0.072 at rate 0.2 to 0.749 at 0.3 on
  `post_residual`. Add 0.22, 0.25, 0.27 so the interesting region has resolution.

Cheap: 2 operators x a few rates, well under one job.

---

## E2. What happens when the model already has dropout

### Why this is not a detail

PSBD's protocol trains **without** dropout. The paper states it plainly
(`papers/PSBD/sec/4_method.tex`): "the model is trained on the poisoned training
set following the standard training procedure, which excludes the use of dropout,
data augmentation, and data normalization. After that, we apply dropout ... during
model inference."

This repo's checkpoints match that: `models.build_vit` calls `vit_b_16` with
torchvision's defaults, so `dropout` and `attention_dropout` are both 0.0, and
`defences/dropout.py` deliberately leaves every existing dropout at its eval
identity and injects fresh modules instead.

So the entire study rests on an assumption nobody has tested: **that PSBD needs a
dropout-free model.** If the "neuron bias effect" is a real mechanism, a model
trained with dropout has already been regularized against exactly the
single-path dependence PSBD exploits, and detection should degrade. If detection
is unaffected, the neuron-bias story is not what is doing the work, which lines up
with what [H23](../hypothesis/H23-gaussian-noise-control.md) is separately testing.

It is also the practical question. Real deployed ViTs are trained with dropout or
stochastic depth. A defence that only works on models trained without either is
much weaker than one that does not care.

### E2a. Inference-time stacking (cheap, no retraining)

Turn the model's own dropout modules on at inference **in addition to** the
injected probe, and measure detection as a function of the model's rate.

    p_model in {0.0, 0.05, 0.1, 0.2}   x   the probe's usual rate grid

Effective removal compounds, so a probe rate p_probe on top of p_model leaves
`(1 - p_model)(1 - p_probe)` alive. This must be reported as measured sigma, not
as the nominal probe rate, or the two knobs are confounded. That is exactly the
compounding bug `sweep_rates`' `try/finally` exists to prevent, appearing here
deliberately rather than accidentally.

**Implementation.** A `--model-dropout` flag on `psbd_dropout_sweep.py` that sets
`p` and calls `.train()` on the model's own `nn.Dropout` modules, excluding
`*.encoder.dropout` (the embedding dropout, applied once before the stack, which
`configure_pre_residual_dropout` already skips for the same reason). Cache name
gains a `_pmodel<rate>` suffix so it cannot collide.

**Caveat that must be stated.** These checkpoints were *trained* at
`dropout = 0.0`, so switching their dropout on at inference perturbs a model that
never saw that noise during training. That is a different experiment from E2b and
is not a substitute for it. E2a measures "extra noise on top", E2b measures "a
model that learned under noise".

### E2b. Trained-with-dropout checkpoints (needs retraining)

Retrain a small matrix with `dropout > 0` and run the full PSBD stack on the
result.

    architecture  vit
    dataset       cifar10
    attacks       badnet_a2o, blend, wanet          (patch, blended, warping)
    poison rates  0.01, 0.1                          (the failing case and the easy one)
    model dropout 0.1                                (plus 0.0 already on disk as control)
    epochs        15, seed 0                         (unchanged, for comparability)

12 new checkpoints. Everything else in the recipe is held fixed so the only moving
part is training dropout.

**Prediction.** If the neuron-bias mechanism is real, PSBD AUROC falls on the
`dropout = 0.1` models, most on `badnet_a2o` (the attack whose detection is most
placement-sensitive). If AUROC is unchanged, the mechanism is not load-bearing on
ViT and the method is measuring prediction margin.

**Also worth recording.** Whether training dropout changes ASR or clean accuracy
at all. If it suppresses the backdoor itself, that is a (weak, expensive)
*defence* rather than a detection result, and the two must not be confused.

---

## E3. An adaptive attacker that trains against PSBD

The most important of the three, and the one a reviewer will ask for.

### Threat model

The attacker controls the poisoned training set **and the training loss**. This is
the standard adaptive-attack setting: a defence is only interesting if it survives
an adversary who knows it. Stated assumptions:

- The attacker knows PSBD's statistic (PSU) and its decision direction.
- **White-box variant:** the attacker also knows the defender's exact placement
  and rate. This is the strongest attack and gives the upper bound on evasion.
- **Transfer variant:** the attacker optimizes against one (placement, operator)
  and the defender probes with a different one. This is the realistic case and the
  one the operator study is built to inform.

The attacker does *not* control the defender's clean validation set or threshold.

### Objective

PSBD flags low PSU. The attacker therefore wants poisoned samples to lose as much
confidence under perturbation as clean ones do, while keeping the backdoor
working.

    original form
        L = L_CE(f(x), y) + lambda * L_evade
        L_evade = ReLU( mean_{x in C} PSU(x) - mean_{x in P} PSU(x) )

    restated
        train as usual, plus a hinge penalty that is zero once the poisoned
        samples' average prediction shift is at least as large as the clean
        samples' average. It pushes the poisoned distribution up onto the clean
        one and stops, rather than pushing it arbitrarily far, which would create
        a detectable signal in the other direction.

PSU per sample is exactly PSBD's Equation 2, computed inside the training step:

    PSU(x) = P_c(x; theta) - (1/k) sum_i P_c(x; p, theta'_i),  c = argmax_j P_j(x; theta)

`c` is an index and is detached, so it contributes no gradient. Everything else is
differentiable: dropout masks are constants in the backward pass, so gradients
flow into `theta` through both terms.

**Why a hinge and not `|mean_C - mean_P|`.** A symmetric penalty would also punish
the attacker for poisoned PSU going *above* clean, which costs capacity for no
evasion benefit and risks overshooting into the opposite tail. The hinge encodes
"be at least as fragile as clean data, no more".

**Variant worth testing if the hinge is not enough:** add a second-moment term
`|std_C - std_P|`, since matching means alone still leaves a shape difference an
AUROC can exploit.

### Algorithm

```
for each batch (images, labels, is_poisoned):
    logits    = model(images)                       # no probe attached
    loss_ce   = cross_entropy(logits, labels)

    c         = logits.argmax(dim=1).detach()       # index only, no gradient
    p_base    = softmax(logits).gather(1, c)

    plug probe at (position, rate)                  # defences.dropout.plug_dropout
    p_drop    = mean over k passes of softmax(model(images)).gather(1, c)
    unplug probe                                    # try/finally, always

    psu       = p_base - p_drop
    loss_ev   = relu(psu[~is_poisoned].mean() - psu[is_poisoned].mean())

    (loss_ce + lambda * loss_ev).backward()
    optimizer.step()
```

The unplug must be in a `finally`. A probe left attached across steps compounds
into an effective rate of `1 - (1-p)^n` and every subsequent number is quietly
wrong, which is the failure `sweep_rates` documents at inference time and which is
worse here because it would silently corrupt the trained weights.

### Cost

`k + 1` forward passes and one backward through the perturbed graph, so roughly
**4x a normal training step** at k = 3. A 15-epoch ViT-B/16 CIFAR-10 run becomes
roughly 4 times its current wallclock. Sizing the first matrix at 6 checkpoints
keeps this to a handful of jobs; it is not cheap but it is not prohibitive.

### First matrix

    attacks        badnet_a2o, blend          (one localized, one distributed)
    poison rates   0.01, 0.1
    lambda         0.0 (control, = existing checkpoints), 1.0, 10.0
    probe          before_attention_norm at the sigma 0.6 rate

`lambda = 0` must reproduce the existing checkpoints' ASR and detection within
noise. If it does not, the adaptive training path differs from the normal one in
some way other than the penalty, and nothing else in the experiment is trustworthy.

### What counts as a successful attack

All three must hold together:

| quantity | requirement |
|---|---|
| ASR | stays above 0.9 |
| clean accuracy | within 2 points of the lambda = 0 control |
| PSBD AUROC at the targeted probe | falls toward 0.5 |

An attack that destroys ASR or clean accuracy has not evaded the defence, it has
removed the backdoor, and reporting it as evasion would be wrong.

### The transfer test, which is the actual point

Having trained against one probe, evaluate detection under **every** operator and
position in the study.

- If evasion transfers everywhere, PSBD is broken by an adaptive attacker and the
  honest conclusion is that it should not be deployed alone against one.
- If evasion is specific to the targeted probe and detection survives under other
  operators, then **operator diversity is itself the defence**, and the
  perturbation study has produced something more useful than a single better
  placement: a probe family an attacker must evade all of at once.

That second outcome is the one worth designing the paper around, and it is
falsifiable. [H16](../hypothesis/H16-where-the-backdoor-neurons-are.md)'s finding
that different attacks use disjoint dimensions is weak prior evidence for it:
if the backdoor's location is that plastic, an attacker suppressing one probe's
view may simply relocate into another's.

### Failure modes to watch

- **The penalty degenerates.** The cheapest way to equalize PSU is to make clean
  predictions as robust as poisoned ones, rather than the reverse. Monitor mean
  PSU on both groups separately, not just the gap. If clean PSU falls to meet
  poisoned PSU, the attacker has not hidden the backdoor, it has changed the
  model's overall robustness, and the clean-accuracy check may not catch it.
- **The probe collapses training.** At a high rate the perturbed forward pass is
  near-random and its gradient is noise. Start at the sigma 0.6 rate, not higher.
- **Poisoned-sample identity leaks into the loss.** `is_poisoned` is legitimate
  for the attacker (it controls the poisoning) but must never reach the defender's
  side of any evaluation.

---

## Order of work

1. E1 grid extension, smallest and unblocks clean comparison. Under 1 job.
2. E2a inference-time stacking, no retraining. 1 to 2 jobs.
3. E3 implementation and smoke tests on CPU, then the `lambda = 0` control, which
   must reproduce existing numbers before any real attack run is launched.
4. E3 first matrix, then the transfer table.
5. E2b retraining, last, because it is the most expensive and the least likely to
   change the paper's direction.
