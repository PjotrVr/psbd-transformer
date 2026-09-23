# Where a backdoor lives in PreAct ResNet, ViT and Swin

This is a map of the 3 architectures in this repository for a security reader who
already knows how they compute. For each component it says what an attacker can
write there, what a defender can read there, how the training recipe moves a
backdoor toward or away from it, and what this project measured about it. Every
literature claim carries a source, and every claim of ours names the macro or file
that produces its number.

Claims are tagged **measured**, **argued** or **unknown**. The literature is thin
on transformers and the unknowns are as useful as the rest, so they are kept.

The sources are the local copies under `literature/`, cited as a path and a line
where there is one, and arXiv identifiers otherwise. The perturbation sites this
project injects at are the entries of `models.positions.POSITION_REGISTRY`, and
`docs/perturbations.md` explains each operator.

## 1. The short version

A backdoor is a shortcut the model learns because it is cheaper than the task. On
a ConvNet the literature finds it in a small set of late convolutional channels.
On a ViT it is 1 direction in the residual stream, carried by the trigger's own
tokens for most of the network and routed to the class token late through
attention. On Swin the picture is the same with 2 complications: windowed
attention delays the routing, and there is no class token to route to.

The consequence for a defender is that the right place to look depends on the
architecture's routing, not on where the ConvNet literature looked. The
consequence for an attacker is that every component a defender reads is a
component the attacker can train against, and the literature already has a
published attack on most of them.

## 2. PreAct ResNet

### The residual stream

He et al. move normalization and activation inside the residual branch, so a
unit computes $x_{l+1} = x_l + \mathcal{F}(x_l)$ with an identity shortcut
(arXiv:1603.05027). The stream therefore carries an unnormalized tensor that
nothing rescales between blocks.

- **Write.** An attacker's feature can be added to the stream by any block and
  survives unchanged to the head. The multi-trigger work installs up to 10
  coexisting triggers that sum on the same stream
  (`literature/multi_trigger/source/main.tex:82`), which breaks every defense that
  assumes 1 shortcut. **measured**
- **Read.** A probe on the stream sees the accumulated sum of every block's
  contribution rather than any single block. PSBD's published placement is
  dropout here, after each residual add and before the activation
  (`literature/psbd-li-arxiv2024/source/sec/4_method.tex:107`). **measured**
- **This project.** `models/positions.py` registers `post_residual` for
  `resnet18`, and `pbs/generate_resnet_control_jobs.py` runs it as the ConvNet
  control.

### The last convolutional stage and the channels before the pool

The founding localization result is that triggered inputs fire channels that are
dormant on clean inputs, in the last convolutional layer (BadNets,
arXiv:1708.06733, and Fine-Pruning, arXiv:1805.12185). Pruning those channels in
order of clean activation collapses the attack success rate once 0.68 to 0.82 of
the layer is gone, while clean accuracy holds (`fp/methodology.tex:52` in the
Fine-Pruning source). **measured**

- **Write.** The pruning-aware attack confines clean and backdoor activations to
  the same few channels, so pruning the backdoor costs clean accuracy
  (`fp/methodology.tex:118`). Concentration in the last layer is a property of
  ordinary training, not of backdoors, so an attacker who controls training can
  move it. **measured**
- **Read.** Every channel-pruning defense reads statistics at the pooled vector,
  where a trigger's response becomes 1 scalar per channel: ANP
  (arXiv:2110.14430), CLP's channel Lipschitz bound (arXiv:2208.03111) and
  FT-SAM's weight norms (arXiv:2304.11823). Layer-wise feature analysis finds the
  divergence peaks inside the last stage rather than just before the head, at
  layer 8 to 9 of a 10-layer ResNet-18 on CIFAR-10 (arXiv:2302.12758).
  **measured**
- **Hole.** Fine-Pruning names an attack that suppresses rather than activates
  neurons as out of scope (`fp/methodology.tex:141`). Nothing in that family
  closes it. **measured absence**

### BatchNorm

BatchNorm is where the ConvNet and transformer security surfaces differ most,
because it keeps running statistics and LayerNorm does not.

- **Read.** A backdoor neuron's recorded population statistics disagree with the
  statistics of benign data alone, which separates it from benign neurons with
  10 to 500 clean samples and no poisoned data (Zheng et al., NeurIPS 2022).
  IBD-PSC scales the learned parameters of the last BN layers and reads which
  predictions survive (`literature/ibd-psc-hou-icml2024.pdf`). **measured**
- **Write.** DeferBad chooses its concealment by whether BN exists. With BN it
  edits the first layers and freezes the running statistics, so later benign
  fine-tuning shifts the statistics and reactivates the backdoor
  (`literature/deferbad/source/iclr2025_conference.tex:320`). **measured**
- **This project.** IBD-PSC is defined on BN layers and has no published
  LayerNorm version. `detectors/` ports it by amplifying LayerNorm gains, which
  is an adaptation rather than the published method, and the uncalibrated port
  saturates on ViT while the calibrated one is the strongest competitor
  (`\DetectorsAurocIbdPsc` against `\DetectorsAurocBestCompetitor` in
  `paper/headline.tex`).

### The stem

No paper localizes a backdoor to the stem. Shallow layers show similar features
for clean and poisoned inputs (arXiv:2302.12758). A high-frequency trigger is cheap
for the stem's small receptive field to pick up, and the same property lets a
defender read the input spectrum before the stem (Zeng et al., arXiv:2104.03413).
**measured**

## 3. ViT-B/16

### Patch embedding and positional embedding

A 16 by 16 patch at 224 by 224 gives a 14 by 14 token grid, so a trigger smaller
than a patch still occupies a whole token
(`literature/megatron/source/main.tex:801`).

- **Write.** TrojViT ranks patch salience and places its trigger on the most
  salient patches (arXiv:2208.13049). **measured**
- **Read.** Doan et al. find that clean accuracy and attack success respond
  differently to patch transformations applied before the positional encoding,
  and build a defense for patch and blend triggers on it (AAAI 2023,
  arXiv:2206.12381). This is the closest prior ViT-specific input defense and the
  paper does not yet cite it. **measured**
- **Unknown.** No paper localizes a backdoor to the positional embedding itself.
- **This project.** `after_embedding` perturbs the embedding output. It reads
  `\StaircaseDropoutSitesAfterEmbeddingAuroc` with dropout, below the attention
  input.

### The class token

- **Read.** Karayalcin et al. find the trigger direction reaches the class token
  at block 5 to 6 for SSBA and only in the last few blocks for BadNets
  (`literature/backdoor-directions-karayalcin/source/main.tex:271`). **measured**
- **Write.** Megatron's attention-diffusion loss acts on the class-token row of
  the last layer's attention rollout, so it trains against the exact readout a
  rollout-based defender uses (`literature/megatron/source/main.tex:792`).
  **measured**
- **This project.** Activation patching finds the class token recovers nothing
  until the last quarter of the network on local triggers, and on BadNets it
  recovers `\PatchingBadnetFinalCls` at the last block. The class token's
  attention on the trigger tokens rises from `\RoutingBadnetClsEarly` in blocks 1
  to 4 to `\RoutingBadnetClsLate` in blocks 9 to 12. `docs/hypothesis/` holds the
  runs, `scripts/paper/mech_routing.py` and `mech_activation_patching.py` the
  numbers.

### The attention input and its LayerNorm

A ViT block is pre-norm: $\hat z = z + \mathrm{MSA}(\mathrm{LN}(z))$, then
$z' = \hat z + \mathrm{MLP}(\mathrm{LN}(\hat z))$. A LayerNorm rescales each token
to unit variance, so a perturbation injected before it is partly divided away and
a perturbation injected after it is not.

- **Unknown in the literature.** No paper compares the attention input, the MLP
  input, the normalization output and the residual stream as probe sites for
  backdoor detection, in either direction. This project's placement result is the
  first measurement of that contrast.
- **This project.** It is the result the paper rests on. Token masking at
  `before_attention_norm` reads `\HeadlineAurocAdaptive` against
  `\PublishedAurocAdaptive` for dropout on the stream, a paired gain of
  `\HeadlineGainAdaptiveAuroc`. The attention input's LayerNorm passes
  `\AbsorptionAttentionInputGaussianSurvival` of a Gaussian disturbance and
  `\AbsorptionAttentionInputTokenMaskSurvival` of a token mask, which is why noise
  there loses to masking (`scripts/paper/mech_layernorm.py`, 2 models).
- **Write.** An attacker who knows the defender masks tokens at this site can
  spread the trigger over many tokens so that no mask removes enough of it. That
  is the adaptive attack of `paper/sections/robustness.tex`, and it defeats a
  single probe.

### Attention heads and the attention map

- **Read.** 2 published ViT defenses detect outliers in the attention map, and
  both work only for patch triggers (arXiv:2206.12381, and Subramanya et al.
  cited at `literature/backdoor-directions-karayalcin/source/main.tex:110`).
  Blacking out the highest-attention region takes ViT-Base from 61.4 to 16.4
  attack success and fails on ResNets, because attention localizes the trigger
  at an IoU of 0.47 on ViT-Base against 0.04 on ResNet-50 (arXiv:2206.08477).
  **measured**
- **Write.** Every attention-map defense has a published attack. BadViT
  maximizes attention on the trigger, Megatron shapes the rollout while keeping
  it quiet and PASTA optimizes attention stealth and spreads the trigger across
  neighboring patches (`literature/pasta/source/1_abstract.tex`). HPMI swaps a
  whole head for a malicious one with no retraining (arXiv:2508.10243).
  **measured**
- **This project.** `attention_heads` masks whole heads inside
  `F.multi_head_attention_forward`. It reads `\AttentionHeadMaskAuroc` on
  `\AttentionHeadMaskN` models, strong but below token masking at the attention
  input, and it is outside the declared basis (`paper/tables/attention_probes.tex`).

### The residual stream

- **Read.** 1 rank-1 direction removed from the embedding and every attention and
  MLP output projection takes mean attack success from 97.7 to 6.7 over 33
  ViT-B/16 models, with Blend on CIFAR-100 the single survivor
  (`literature/backdoor-directions-karayalcin/source/main.tex:180`). **measured**
- **This project.** The removal reproduces on our models: attack success falls
  from `\ErasureAsrBeforeMean` to `\ErasureAsrAfterAllWritesMean`, editing only
  blocks 10 and 11 leaves `\ErasureAsrAfterBlocksOneZeroOneOneMean`, and Blend on
  CIFAR-100 survives. Dropout on the stream is the published PSBD placement and
  it ranks 5 of 13 defenses on ViT (`\DetectorsAurocRankPublished`), because it
  perturbs the class token itself on clean and triggered inputs alike.

### MLP and its hidden units

- **Argued.** Subramanya et al. report that attention layers hurt backdoor
  robustness while feed-forward layers help (WACV 2024). Their numbers were not
  verified here.
- **This project.** Token masking at the MLP input reads
  `\StaircaseOperatorsBeforeMlpNormTokenMaskAuroc`, below the attention input,
  because the MLP acts after attention has already mixed the trigger into the
  class token. Channel masking of the MLP hidden units reads
  `\StaircaseOperatorsMlpNeuronsChannelMaskAuroc`.

### The final norm and the head

Karayalcin et al.'s data-free detector reads the head's class rows against the
early output projections and catches WaNet and BPP but fails on patch triggers
(`main.tex:423`). Our reproduction names the WaNet target on all
`\WeightDetectorWanetModels` WaNet models and the BPP target on
`\WeightDetectorBppNamed` of `\WeightDetectorBppModels`
(`scripts/paper/app_weight_detector.py`). No paper reads the final LayerNorm
output itself. **unknown**

## 4. Swin-S

Swin is a hierarchical transformer, and every difference from ViT changes where a
backdoor can hide. The source is Liu et al., arXiv:2103.14030.

### The stem and patch merging

The stem splits the image into 4 by 4 patches, so a trigger covers 16 times more
tokens in the first stage than it does in ViT-B/16. Each patch-merging layer
concatenates a 2 by 2 group of tokens and projects it, halving resolution
(`swin/main.tex:105`). A trigger on 1 first-stage token has been averaged with 3
neighbors by stage 2 and with 63 by stage 4.

- **Consequence for a defender.** Token statistics are not comparable across
  stages, so a per-token probe has to be defined per stage. A depth band given
  as a range of block indices mixes stages with different token counts and
  window geometry, and the Swin band results in this repository should be read
  per stage.

### Window attention and the shift

Attention is restricted to 7 by 7 windows, and consecutive blocks alternate
between the regular partition and 1 displaced by 3 tokens (`swin/main.tex:144`).
A trigger token can only influence its own window in 1 block, and its information
spreads by about 1 window per pair of blocks. There is no direct global path.

- **Consequence.** PASTA's radiating-trigger argument assumes global attention
  and does not transfer unchanged. The depth at which a backdoor becomes global is
  stage-dependent rather than block-dependent. **argued**
- **This project.** The routing account predicts that token masking at the
  attention input should still work under windowing, because every route still
  passes through a windowed attention. It does: `\SwinRecommendedAurocAdaptive`
  over `\SwinCells` models against `\SwinPublishedAurocAdaptive` for dropout on
  the stream. Dropout at the same site gains `\SwinGainDropoutInputMinusPublished`
  over the stream, an interval containing 0, so on Swin too it is the operator
  that the attention input rewards.

### No class token

Swin classifies from a global average pool over the last stage
(`swin/main.tex:503`). Every class-token analysis needs redefinition, and
Karayalcin et al. use the token mean (`main.tex:287`), which is a choice rather
than a derivation. An attention-capture attack has to dominate the mean instead
of 1 token, which is a harder objective at a fixed trigger area. **argued**

### The relative position bias

Each head adds a learned $(2M-1)^2$ table, 169 entries at $M = 7$, to its
attention logits, shared across every window and every input. A weight-level edit
to it changes the attention prior for a whole stage at once. No published attack
or defense touches it. It is the cheapest unexamined weight surface in Swin.
**unknown**

### What this project cannot yet probe on Swin

`POSITION_REGISTRY["swin"]` has no `attention_heads` entry, because torchvision's
`ShiftedWindowAttention` does not expose per-head outputs the way
`F.multi_head_attention_forward` does. Head masking is therefore measured on ViT
only.

## 5. How the training recipe moves a backdoor

| Knob | What is measured | Direction | Source |
|---|---|---|---|
| Poison rate | Strength of the backdoor, not its depth. The ViT steering profile barely moves between 1% and 10% | rate changes magnitude, not location | Karayalcin et al. `main.tex:271`. **measured** |
| Epochs | Poisoned data is fitted much faster than clean data | a late model has the firmest shortcut | Anti-Backdoor Learning, arXiv:2110.11571. **measured** |
| Augmentation | Strong augmentation suppresses the backdoor. PSBD finds augmentation helps its detection | weaker backdoor, stronger clean bias | Borgnia et al. arXiv:2011.09527, PSBD `sec/4_method.tex:193`. **measured** |
| SAM on poisoned data | Amplifies the trigger-activated change of the already strongest neurons | concentrates the backdoor | Zhang et al., ResNet and VGG only. **measured**, no transformer |
| SAM on clean fine-tuning | Shrinks the same large-norm neurons | removes the backdoor | FT-SAM, arXiv:2304.11823. **measured** |
| Frozen backbone | Does not prevent a backdoor, it moves the whole of it into the head | head only | Subramanya et al. arXiv:2206.08477, 61.4 ASR on ViT-Base. **measured** |
| Label smoothing | Attacker-side smoothing defeats trigger inversion. Defender-side is unstudied | | LSP, `literature/label_smoothing_poisoning/`. **measured** attacker side only |
| Weight decay, LR schedule | Only federated numbers exist | | arXiv:2509.05192. **unknown** for centralized training |

The 2 SAM results look contradictory and are not: SAM pushes hardest on the
parameters the loss depends on most sharply, and the sign follows the data the
loss is computed on. In this repository Adam is the default optimizer and SAM
appears only on the `_sam_rho_` checkpoints, which the paper declares out of scope.

To place a backdoor deliberately at a component, the literature offers 3 levers:
choose the trigger's spatial extent (1 token or all of them), add a loss on the
component a defender reads (Megatron on the rollout, the adaptive attacker of this
project on PSU), or freeze everything but the target component during poisoning
(the frozen-backbone result).

## 6. What transfers between the 3

- A defense that reads attention gains 45 attack-success points on ViT and loses 2
  on ResNet-50, because attention localizes the trigger on ViT and nothing
  localizes it on a ResNet (arXiv:2206.08477). **measured**
- BackdoorBench reports that ConvNet defenses degrade on ViT
  (`main.tex:107`). A counter-paper, "Rethink Backdoor Robustness in Vision
  Transformers" (ECCV 2026), argues the degradation is a tuning artifact. It was
  not obtained, so the claim that defenses transfer worse to ViT is not settled.
- Karayalcin et al. exclude LF because it stays under 5% attack success on
  BackdoorBench's ViTs (`main.tex:136`). On our recipe LF implants at
  `\AttackLfAsrMin` to `\AttackLfAsrMax`, so implantability depends on the
  training recipe and not on the attack alone.
- PSBD itself transfers once the perturbation moves. The published placement is
  mid-pack on ViT, and token masking at the attention input ranks
  `\DetectorsAurocRankOurs` of `\DetectorsAurocDefensesRanked`.

## 7. Open questions this map leaves

1. **Rate against depth.** Nobody measures whether a lower poison rate writes the
   backdoor at a different depth. The only evidence is incidental and says no.
2. **Probe sites inside a pre-norm block.** This project is the only measurement.
   It has not been checked on a post-norm transformer.
3. **Attention sinks and register tokens.** The benign phenomenon is measured
   (Darcet et al., arXiv:2309.16588) and the security link is untested.
4. **LayerNorm gain and bias, and Swin's position bias, as backdoor surfaces.**
   Nothing localizes a backdoor to either.
5. **Head masking on Swin.** Needs a wrapper around `ShiftedWindowAttention`.
