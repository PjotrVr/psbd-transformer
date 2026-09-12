# Training-time ideas for perturbation-consistency detection at low poisoning rates, 2026-09-12

PSBD (Li, Chen, Liu, Wang, CVPR 2025, [arXiv:2406.05826](https://arxiv.org/abs/2406.05826)) reads a
model twice, once with dropout off and once with dropout on, and flags an input whose prediction
barely moves under that perturbation. On our ViT adaptation it works at 5% and 10% poisoning and
degrades at 1%, where the backdoor shortcut is weak and a poisoned input's prediction moves under
the perturbation almost as much as a clean one's does. We control training and know PSBD will run
at deployment, so the question is what to do at training time, without knowing which samples are
poisoned, to widen that gap. This note reads the PSBD paper itself, its citing and neighboring
literature on amplifying backdoors for detection, training-time detectors that split or purify
during training, consistency detectors related to PSBD's own mechanism, and augmentation tricks
with a known effect on trigger strength.

## Papers read

**PSBD itself.** [arXiv:2406.05826](https://arxiv.org/abs/2406.05826), Li, Chen, Liu, Wang, CVPR
2025 (peer reviewed). The method computes Prediction Shift Uncertainty, the variance in output
probabilities between a dropout-off and a dropout-on pass, and attributes the shift to a neuron
bias effect that pulls clean features toward other classes under dropout while backdoor features
resist the pull. The abstract and the arXiv listing carry no explicit discussion of a low poisoning
rate failure mode. Our own project record already establishes that gap empirically on ViT, so the
literature below stands in for what the source paper does not itself say.

**IBD-PSC.** Hou et al., ICML 2024, [arXiv:2405.09786](https://arxiv.org/abs/2405.09786), peer
reviewed. A test-time detector close in spirit to PSBD, it scales up the learned parameters of
batch normalization layers rather than toggling dropout, and reads the resulting prediction
consistency. Poisoned inputs stay consistent under that amplification while clean inputs do not,
the same asymmetry PSBD reads under dropout. It reports AUROC near 0.99 on CIFAR-10 across 7
attacks, but at rates the paper's own tables keep at 5% and above, so it offers a second consistency
signal rather than a low-rate answer.

**SCALE-UP.** Guo et al., ICLR 2023, [arXiv:2302.03251](https://arxiv.org/abs/2302.03251), peer
reviewed. A black-box input detector that scales pixel values up rather than perturbing the
network, and reads how often the scaled image's label agrees with the original. The mechanism is
the same asymmetry again, a poisoned input's label survives a perturbation that moves a clean
input's label. It needs no access to the network's internals, only hard labels, which makes it a
candidate second signal to combine with PSBD rather than a training-time lever on its own.

**TeCo.** Liu et al., CVPR 2023, code at
[github.com/CGCL-codes/TeCo](https://github.com/CGCL-codes/TeCo), peer reviewed. Perturbs the
input with image corruptions of increasing severity instead of perturbing the network, and reads
the severity at which the prediction flips. A backdoored model's clean images flip at similar
severities across corruption types, poisoned images flip at very different severities across types.
The paper itself flags a limitation relevant to our own work, degraded separation on some
architectures (VGG), which is a caution that a perturbation-consistency signal is architecture
dependent, exactly the finding that motivated moving PSBD off ConvNets in the first place.

**SAM-enhanced poisoned sample detection.** Zhang, Zhu, Zhu, Wu, accepted ICLR 2026,
[arXiv:2411.11525](https://arxiv.org/abs/2411.11525), peer reviewed (accepted, not yet at the
venue). This is the paper closest to our own question, and the one our SAM experiments already
test directly. It trains the model itself with Sharpness-Aware Minimization rather than plain SGD
or Adam, then runs an existing training-set poisoned-sample detector (Spectral Signature,
Activation Clustering and others) on the resulting features. SAM's flatter minimum amplifies the
activation of the top Trigger Activation Change neurons and suppresses the rest, and the paper
reports the gain is largest exactly where detection is weakest, at low poisoning ratio or weak
trigger strength, +34.38 points average true positive rate over the same detectors on a vanilla
model. Our own `docs/sam-findings-2026-09-11.md` reruns this claim on ViT-B/16 CIFAR-100 with
Spectral Signature and Activation Clustering at their own decision rules and reads at most 2.0
points of TPR movement at 1% poisoning, essentially flat, against the paper's double-digit ResNet18
CIFAR-10 gain. The most likely reason on record is a ceiling effect, ViT's target class features
are already 73 to 79% separable under plain Adam at 1% poisoning where the paper's ResNet18 baseline
for a weak trigger sits at 32.9%, leaving SAM little room to add. Our own PSBD-side reading (not
training-set detection, PSBD's inference-time consistency score) shows the same rate dependence,
gains of +0.065 to +0.074 at 10% poisoning collapsing to noise at 1%.

**Feature Shift Tuning (FST).** Min, Qin, Shen, Cheng, NeurIPS 2023,
[arXiv:2310.01875](https://arxiv.org/abs/2310.01875), peer reviewed. This is a post-hoc purification
method, tuning an already-poisoned checkpoint on a small clean set, not a training-time or
detection method, but its diagnosis is the one we need. It states directly that vanilla fine-tuning
purification fails specifically at low poisoning rates because backdoor and clean features are
entangled in the same feature directions, and the entanglement is worse the fewer poisoned examples
shaped that direction. FST's fix is to actively push the classifier's weight direction away from
the compromised one during fine-tuning, which stabilizes purification across poisoning rates at a
cost of only 10 epochs of tuning. The relevant lesson for us is diagnostic, not directly portable,
because FST assumes a fixed poisoned checkpoint and a labeled clean set, neither of which our
training-time setting has, but the entanglement account explains why any low-rate defense needs to
act on feature geometry, not just on a score threshold.

**Backdoor Defense via Decoupling the Training Process (DBD).** Huang, Li, Wu, Qin, Ren, ICLR 2022,
[arXiv:2202.03423](https://arxiv.org/abs/2202.03423), peer reviewed. Poisoned samples cluster in
feature space because end-to-end supervised training lets the label signal pull them there. DBD
breaks the pipeline into 3 stages, a label-free self-supervised backbone, a label-noise-robust
head trained on the frozen backbone, and a semi-supervised fine-tune that treats low-confidence
samples as unlabeled. Because self-supervised pretraining never sees labels, it cannot learn the
label-trigger shortcut in the first stage, so poisoned and clean features stay closer together in
that space and the resulting classifier is harder to backdoor at all. It is a full retraining
recipe, expensive, and it changes what gets learned rather than making an already-learned backdoor
more visible to a probe.

**Backdoor Defense via Adaptively Splitting Poisoned Dataset (ASD).** Gao et al., CVPR 2023,
[arXiv:2303.12993](https://arxiv.org/abs/2303.12993), peer reviewed. Trains with a running clean
pool and polluted pool, initialized from a small clean seed and updated online by a per-sample
loss-guided split, then trains the clean pool supervised and the polluted pool unsupervised
(semi-supervised consistency loss). This is close in shape to a training-time detector in the loop.
Its splitting signal is loss magnitude and speed of loss decrease, poisoned samples fit faster
early in training, which is an orthogonal signal to PSBD's dropout-shift signal and could in
principle be logged during our own training runs at no extra inference cost, though its published
results are again reported at 5 to 10% poisoning rather than at 1%.

**Backdoor Defense via Deconfounded Representation Learning (CBD).** Zhang, Jin, Wang, Qi, Wu, Yang,
Wang, CVPR 2023, [arXiv:2303.06818](https://arxiv.org/abs/2303.06818), peer reviewed. Casts the
backdoor as a confounder in a causal graph linking image and label, trains one deliberately
under-fit model (via early stopping) to capture the confounded, trigger-driven association, and a
second model that is penalized for mutual information with the first model's confounded features
and reweighted per sample. The mechanism, deliberately encouraging one copy of the network to
overfit the shortcut early so a second copy can be pushed away from it, is the closest published
idea in this survey to a training-time hinge that separates clean from backdoor behavior without
sample labels, though it targets robust classification rather than a downstream detector's score
gap.

**CBD, a different acronym: A Certified Backdoor Detector Based on Local Dominant Probability.**
Xiang, Xiong, Li, NeurIPS 2023, [arXiv:2310.17498](https://arxiv.org/abs/2310.17498), peer reviewed.
An inference-time detector built on randomized smoothing, certifying that a flagged input's
prediction under noise is dominated by a single class with a provable margin. It is a test-time
detector, not a training-time lever, and it needs no clean validation set, but its certification
guarantee is a property of the noise distribution and the frozen model, so it offers a template for
what a certified version of PSBD's own dropout perturbation could look like, at the cost of the
usual smoothing overhead (many noisy forward passes per input).

**On Certifying Robustness against Backdoor Attacks via Randomized Smoothing.** Weber, Xu, Karlas,
Zhang, Li, 2020, [arXiv:2002.11750](https://arxiv.org/abs/2002.11750), workshop/preprint, the
foundational link between randomized smoothing and backdoor robustness that the certified detector
above builds on. It shows smoothing bounds the effect any bounded-norm trigger can have on the
output distribution, which is a robustness certificate rather than a detection score, and is
included here because it is the theoretical ancestor of every "read the model under noise and
compare" detector in this list including PSBD.

**Backdoor Smoothing: Demystifying Backdoor Attacks on Deep Neural Networks.** Grosse, Lee, Biggio,
Park, Backes, Pendlebury, [arXiv:2006.06721](https://arxiv.org/abs/2006.06721), peer reviewed
(published in Computers and Security). This is the mechanistic paper behind PSBD's whole premise,
though PSBD does not cite it directly by this name. It shows that a successful backdoor attack
makes the decision function measurably smoother in a neighborhood of the trigger than in a
neighborhood of a clean point, and that the smoothing grows with attack success. Framed against our
question, this says the training-time quantity we would want to widen at 1% poisoning is exactly
this local smoothness gap, and that PSBD's dropout probe and any consistency detector are reading a
downstream symptom of it.

**Strong Data Augmentation Sanitizes Poisoning and Backdoor Attacks Without an Accuracy Tradeoff.**
Borgnia, Cherepanova, Fowl, Ghiasi, Geiping, Goldblatt, Goldstein, Gupta, ICASSP 2021,
[arXiv:2011.09527](https://arxiv.org/abs/2011.09527), peer reviewed. Tests mixup, CutMix and related
strong augmentations as a training-time backdoor sanitizer rather than a detector. CutMix reduces
attack success rate from 100% to 36% while raising clean validation accuracy by 9%, but plain mixup
does not defend at all because the base image class can still be re-associated with the patch under
mixing. The relevant lesson is negative for the naive version of the idea, augmentations that blend
whole images can suppress a strong trigger's implantation altogether rather than widening a
detection gap around it, which is a different outcome than what we want (we want the trigger to
still implant, just more separably).

**Revisiting the Assumption of Latent Separability for Backdoor Defenses.** Qi, Xie, Li,
Mahloujifar, Mittal, ICLR 2023, [arXiv:2205.13613](https://arxiv.org/abs/2205.13613), peer reviewed.
Not a proposal but a warning directly on point for category (c) below. It constructs an adaptive
attacker that regularizes its own poisoned samples toward diverse, non-clustered latent
representations and shows every latent-clustering defense in its comparison set collapses. The
lesson for a defender-side hinge is symmetric and unavoidable, any training-time signal we add to
widen the clean-poisoned gap is visible to an adaptive attacker who also controls poisoning and can
be optimized against directly, the same asymmetry our own evasion hinge already exploits from the
attacker's side.

**Variance-Based Defense Against Blended Backdoor Attacks (VaB).** Aseervatham, Kerzazi, Bennani,
ECML PKDD 2025, [arXiv:2506.01444](https://arxiv.org/abs/2506.01444), peer reviewed. A recent
training-set filter that separates poisoned from clean samples by the variance of per-sample
gradients or activations rather than by class-conditional clustering, aimed specifically at blended
triggers, which are harder to cluster than local patch triggers. It needs no held-out clean set. It
targets training-set filtering rather than inference-time detection, but its use of a second-order
statistic (variance across augmented or repeated passes of the same sample) rather than a first
moment is close in spirit to what PSBD reads and is a candidate additional training-time signal to
log for free during our own runs.

## Ranked training-time ideas for our setting

1. **A defender-side hinge that pulls clean PSU up rather than poisoned PSU down.** Our existing
evasion hinge (`cli/train_backdoor.py`) already pushes poisoned PSU toward clean PSU under a probe,
optimized by an attacker who knows the poison labels. The defender-side mirror trains on the whole
batch without labels and adds a term that penalizes low PSU on the batch as a whole, computed the
same way PSBD itself computes it (a second forward pass with the training-time perturbation turned
on, compared to the pass without it), added to the classification loss with a small weight. The
mechanism this targets is exactly Grosse et al.'s local-smoothness gap, if clean predictions are
made to move more under the same perturbation PSBD will later apply at inference, the ratio between
clean and poisoned movement widens even when the poisoned images do not move at all. The risk this
carries is the one Qi et al. raise directly, this term is visible in the loss function to anyone
who can see the training script, so an adaptive attacker who knows a defender-side hinge is present
could shape the trigger to move under the same perturbation, closing the gap back down. It costs 1
extra forward pass per batch (the same computational shape as the evasion hinge already coded), no
new data or labels, and is the most direct test of the training-aware hypothesis this survey was
commissioned to check. Testable immediately with the pipeline we have, add a `--defender-hinge`
flag mirroring the existing `--evasion` flag but penalizing low clean-batch PSU unconditionally,
sweep its weight, and re-run our existing 1% poisoning matrix on GTSRB and CIFAR-100 with
`cli.sweep` and `cli.analyze`.

2. **A consistency-regularization term borrowed from ASD, logged rather than acted on first.**
ASD's per-sample loss trajectory (how fast a sample's own loss falls early in training) is a signal
orthogonal to PSBD's dropout-shift signal and free to compute from logs we already write, since
`training.loop` already tracks per-epoch loss. Before touching the training objective, log
per-sample early-training loss for every checkpoint we already have at 1% poisoning and correlate
it against the eventual PSU gap that same sample gets from PSBD, to see whether the two signals
disagree in the cases PSBD misses. If they are correlated at 1% where PSBD alone is weak, a fused
score (their loss-speed feature plus PSBD's PSU) is a 0-cost win with no training change at all.
This is the lowest-risk item on this list because it changes nothing about training, only what we
read from the runs we already have, and can be tested this week on the checkpoints already on disk.

3. **A CBD-style paired-model early-stop probe as a training-time confounder detector.** Train the
same architecture with a deliberately truncated schedule (early-stopped at 3 to 5 epochs, well
short of our uniform 15) alongside the full run, and treat the truncated model's predictions on a
held-out clean split as an estimate of which directions in feature space the trigger, if present,
would have grabbed first, since Zhang et al.'s CBD shows early stopping isolates the confounded
shortcut before the causal signal catches up. The candidate use is not certification here but
feature reweighting, downweighting the loss on directions the early-stopped model already fits well
(likely spurious, trigger-driven) during the remainder of the full run, which should leave the
class-token trajectory of a real trigger less entangled with clean features and easier for PSBD's
token-mask probe to separate at low rates. Cost is 1 extra short training run per configuration (3
to 5 epochs, cheap relative to the 15-epoch full run) plus a reweighting step in the main loop. Test
on GTSRB at 1% first, since it is our default pilot test bed and clean-label rate caps do not bind
BadNet or WaNet there.

4. **SAM at rho tuned per dataset rather than a single fixed rho, revisited only if item 1 fails.**
Our own `docs/sam-findings-2026-09-11.md` already shows SAM's PSBD-TM gain is real but rate-gated,
present and significant at 10%, flat or sign-flipping at 1%, and the training-set detection version
of the same idea (arXiv:2411.11525) shows the identical rate gate on ViT independently. This is not
a new idea to test, it is a documented negative for our exact question at the exact rate we care
about, kept on the list only because a rho sweep finer than the 4 values already tried (a value
between 0.05 and 0.1, since that is where BPP's gain concentrated) is a cheap 1-parameter check
before writing SAM off at 1% poisoning for good. Cost is training reruns at 2 or 3 new rho values
on the datasets already covered, no new code.

5. **CutMix, not mixup, as a training-time regularizer, tested for its effect on the detection gap
rather than on attack success alone.** Borgnia et al. found CutMix suppresses BadNet-style implants
outright at typical rates, which is the wrong effect for us if it happens at 1% poisoning too,
since a defense that prevents implantation removes the ASR signal PSBD is supposed to catch, not
just the false negatives. The candidate worth testing is narrower than their claim, run CutMix at
our existing `--augment standard` alongside a CutMix variant only on the hard low-rate triggers
already prioritized in this project (adaptive-blend, WaNet, LC, SIG at 1% and 5%) and check ASR
first. If ASR survives above the 0.85 clearing bar under CutMix, check whether PSBD's PSU gap
widens; if ASR collapses, this idea is disqualified for our setting the same way it was for
Borgnia et al.'s BadNet. Cost is 1 augmentation flag change and a rerun of the existing 1%/5% hard
attack grid, cheap and already gated by an ASR check we run anyway.

6. **DBD-style label-free backbone pretraining, as a longer-horizon item.** Full decoupled training
changes what the network can learn to shortcut on at all, which is the strongest lever in this list
for widening a low-rate gap, but it also breaks our 15-epoch uniform training protocol
(`.claude/CLAUDE.md`'s constant-recipe rule) and is a full retraining recipe rather than an addition
to the existing loop, so it is a bigger commitment than items 1 through 5 and should only be taken
up if a lighter intervention fails outright. Cost is a full pipeline change (self-supervised stage,
frozen backbone, semi-supervised fine-tune), 1 to 2 orders of magnitude more compute per checkpoint
than the current recipe, and it would need its own experiment directory under `experiments/` before
touching the canonical training loop.

7. **FST-style weight-direction regularization ported to a from-scratch training term.** FST's
purification-time mechanism, actively steering the classifier head's weight vector away from the
direction a compromised model would settle on, assumes a poisoned checkpoint already exists and a
labeled clean set to steer against, neither of which the from-scratch, label-free setting has. A
speculative from-scratch version would need a proxy for "the compromised direction" built without
labels, plausibly the same early-stopped confounder model from item 3, which makes this item
redundant with item 3 rather than independent, and is listed only to record that FST's diagnosis
(feature entanglement grows as poisoning rate falls) is the clearest statement in this survey of why
1% is hard, even though its fix does not transfer directly.

8. **A second detector, IBD-PSC's batch-norm scaling probe, fused with PSBD rather than trained
against.** ViT-B/16 as we use it has few or no batch-norm layers (LayerNorm dominates), so IBD-PSC's
literal mechanism does not port, but its abstraction, scale a well-chosen internal parameter up and
read whether the prediction moves, is the same family as PSBD's token masking and SCALE-UP's pixel
scaling. This is not a training-time idea at all, it is a second inference-time score to fuse with
PSBD's, listed last because it does not answer the question this survey was assigned (what to do at
training time) but is worth 1 cheap sanity check, whether scaling LayerNorm's learned gain at
inference produces a PSBD-like PSU signal on our own checkpoints with zero training change, before
any training-time effort is spent.

## Training-aware defenses as a category

The literature treats a defender who controls training and knows a detector will run afterward as
a legitimate and common setting, DBD, ASD, CBD, VaB and the SAM-enhanced detector paper all assume
exactly this, training a model with the eventual defense in mind rather than defending a model
handed over by someone else. Reviewers do push back on 1 specific point, not the assumption itself
but whether the proposed training-time signal survives an attacker who is aware of it, the way Qi
et al.'s ICLR 2023 paper shows latent-clustering defenses collapse under an attacker optimized
against exactly that assumption, so any defender-side hinge we build should be evaluated against an
attacker aware of its existence and not only against the fixed attacks in our current suite. The
category's accepted evidence bar is also rate-sensitive across every training-aware paper read here,
methods report strong numbers at 5% and 10% poisoning and none of them report a clean win at 1%,
which is consistent with our own finding rather than a gap unique to PSBD.
