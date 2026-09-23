# Evaluation standards for a security machine learning paper

What the methodology literature asks of an empirical detection paper, and where
PSBD-ViT currently stands against each item. The subject is experimental
validity rather than paper structure. Every rule below carries the URL it came
from, and every verdict about this project names the file or the record it was
read from.

Read `docs/open-questions.md` first. It lists 35 gaps found on 2026-09-23 and
this document cross-references them by Q number rather than restating them.
Where a row here says the same thing as a Q entry, the Q entry is the canonical
statement and this one exists to place it inside an external standard.

## The 10 pitfalls of Arp et al.

Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX
Security 2022, is the central source
(https://www.usenix.org/system/files/sec22-arp.pdf,
https://arxiv.org/abs/2010.09470). The authors rate 30 papers from CCS, IEEE
S&P, USENIX Security and NDSS over 10 years against 10 pitfalls grouped by
workflow stage, find that every paper carries at least 3 of them and that only
22% of the instances come with any discussion in the text. The prevalence
figures below are the "present" percentages printed beside each pitfall in the
paper, and the 2 highest at-least-partly-present rates are sampling bias at 90%
and data snooping at 73%.

**P1, sampling bias, 60% present.** The collected data does not represent the
true distribution of the security problem. Arp et al. accept that the bias can
often be mitigated rather than removed and ask for 2 things: build several
different estimates of the true distribution and analyze them separately, and
state the limitations of the dataset openly so a reader can judge the security
implications. Mixing data from incompatible sources is named as a common cause
of extra bias.

This project is exposed. The panel is 6 datasets of small images upscaled to 224
pixels, every model fine-tuned 15 epochs from ImageNet weights at a constant
learning rate, so the population of backdoored models is 1 recipe rather than a
sample of deployable models (`paper/sections/setup.tex`). The 2 incompatible
sources are this project's own `checkpoints/` and BackdoorBench's downloaded
reference models, and the repository keeps them apart by a documented rule in
`CLAUDE.md`, which is the right handling. `paper/sections/limitations.tex` states
the boundary in the first paragraph, which is what the remedy asks for.

**P2, label inaccuracy, 10% present.** Ground truth is noisy, unstable or
shifting. The remedy is to verify labels where possible and to model or cleanse
the noise, with 1 hard prohibition: instances with uncertain labels must not be
removed from the test data, because that is a variant of P1 and P3 together.

This project's analogue of a label is the poison membership of a test input,
which is exact by construction, so the pitfall does not transfer directly. The
prohibition does transfer to the attack-success bar. Removing 31 of 105 cells
because the attack failed and 3 because training diverged is a defensible
population restriction. `paper/sections/setup.tex` argues it, and it becomes a P2
violation the moment the removed cells are not counted and characterized in
the paper. The counts are there, the behavior of the detector on a weak or
absent backdoor is not.

**P3, data snooping, 57% present, 73% at least partly.** Arp et al. split it into
3 kinds. Test snooping uses the test set for preparatory work such as picking
features, parameters or algorithms. Temporal snooping ignores time dependence in
the data. Selective snooping cleanses data using statistics of the whole dataset
that would not exist at training time. The remedy is to split the test data early
and store it separately until the final evaluation, and to complement experiments
on well-known public datasets with more recent data from the domain, because the
characteristics of a public benchmark leak into every method developed on it.

This project is exposed on test snooping and clean on the other 2, since there is
no time axis in the data. `configs/psbd_basis.json` declares a
`selection_protocol` with `select_on_datasets` of cifar10 and gtsrb and
`report_on_datasets` of cifar100 and tiny, and freezes the 24 probe combinations
before any fused AUROC is read, which is the correct mechanism. Q15 and Q20
record that the mechanism was not honored for the headline placement, which is
test snooping in the exact form the paper describes.

**P4, spurious correlations, 20% present.** An artifact unrelated to the task
gives the model a shortcut. The remedy is to apply explanation techniques,
define the system's objective in advance and check that what the model learned
complies with that objective, since a correlation that is spurious in 1 setting
is a valid signal in another.

This project is exposed in an unusual direction, because the shortcut is the
phenomenon under study rather than a nuisance. The risk is that the prediction
shift statistic separates classes for a reason other than the backdoor. The
repository already ran the 2 obvious controls and 1 of them has no artifact on
disk: the benign reference per dataset (`configs/psbd_basis.json`,
`benign_reference`) and the confidence null, whose record
`results/psu_vs_confidence.json` is missing (Q8). Q35 settles the related
concern for the fractional statistic by deriving that the margin cancels.

**P5, biased parameter selection, 10% present.** The final parameters are not
fixed at training time and depend indirectly on the test set. Arp et al. single
out calibrating a detection threshold on test data, and observe that a threshold
picked from a ROC curve on the test set can be impossible to reach in
deployment. The remedy is strict data isolation with a separate validation set,
which they call sufficient here even though general snooping is hard.

This project satisfies the mechanical part and fails the protocol part.
`defenses.decision.threshold_at_quantile` reads the threshold from
`validation_psu` only, `data.splits.build_psbd_loaders_from_checkpoint` splits
2000 held-out images by a fixed permutation before anything else and the
analysis pool is the complement, so no threshold sees the analysis pool. The
disturbance rate is chosen by `select_rate_adaptively` from the clean-validation
shift ratio, which is label-free. `select_rate_by_oracle` exists and is computed
by `cli/analyze.py` as a diagnostic, and the only oracle number in the paper is
`\RouterOracle` at 0.883, labeled as an oracle. What remains is the placement
choice itself, which Q15 and Q20 show was not made on the declared selection
half.

**P6, inappropriate baseline, 20% present.** Evaluation without baselines, or
against mostly identical models, cannot show an improvement. The remedy asks for
simple and well-understood models beside the complex ones, for automated
baseline search as a lower bar and for a check on whether a non-learning method
also solves the problem.

This project is exposed today and has the material to close it. The published
ConvNet placement is swept on every cell as its own baseline, which is the right
form of same-method comparison. The 11 registered competitor detectors are
currently a smoke test on 3 GTSRB models with 500 validation images
(`paper/sections/appendix.tex`), so the comparison against the state of the art
does not exist yet at panel scale. The simple baseline in the sense Arp et al.
mean is the confidence or entropy of the unperturbed prediction, whose record is
missing (Q8).

**P7, inappropriate performance measures, 33% present, more than 50% at least
partly.** The measure does not account for the constraints of the application,
such as imbalance or a low false-positive requirement. Arp et al. print a ROC
curve and a precision-recall curve computed from the same scores on a 1 to 100
imbalanced dataset and observe that only the precision-recall curve conveys the
true performance. They decline to give a general rule and instead ask the author
to name the deployment and pick measures a practitioner could act on.

This project is exposed. The reported measures are AUROC, TPR at a 10%
false-positive budget and TPR at a 20% budget (`paper/sections/method.tex`), all
3 of which are insensitive to prevalence, and `defenses.decision.detection_report`
returns no precision field at all. `configs/psbd_basis.json` lists `auprc` and
`achieved_fpr` among its own `required_metrics` and neither is computed anywhere
in `defenses/` or reported anywhere in `paper/`.

**P8, base rate fallacy, 10% present.** A large class imbalance is ignored when
the numbers are interpreted. Arp et al. separate this from P7: P7 is the wrong
description of performance, P8 is the wrong reading of a correct description.
Their worked example is 99% true positives at 1% false positives, which at a 1
to 100 class ratio is 100 false positives for every 99 true positives. The
recommendations are precision and recall with their curves, the Matthews
correlation coefficient where the minority prevalence is itself inflated by
sampling bias, ROC curves read only up to tractable false-positive rates with a
bounded AUC, and a discussion of false positives against the base rate of the
negative class so a reader can see the induced workload.

This project is exposed in the strongest possible form, because the evaluation
population is balanced by construction. `defenses.decision.pair_clean_to_backdoor`
pairs 1 clean input to 1 triggered input so that the 2 populations differ only by
the trigger, which is the right experimental control and also fixes the
prevalence at 0.5. The arithmetic below works out what that costs.

**P9, lab-only evaluation, 47% present, more than 50% at least partly.** The
system is evaluated only in the laboratory with no discussion of practical
limits. The remedy asks for temporal and spatial relations of real data, for
runtime and storage measured under practical conditions and ideally for a
deployment.

This project is partly exposed. Runtime is reported, which is rarer than it
should be: the detector table in `paper/sections/appendix.tex` carries forward
passes per input and measured seconds per input on 1 A100, and
`paper/sections/setup.tex` gives training wall time per model. What is absent is
the deployment arithmetic of the previous pitfall and the query distribution a
real service would see.

**P10, inappropriate threat model, 17% present, more than 50% at least partly.**
The security of the learning system itself is not considered. The remedy asks
for a precise threat model, an adaptive adversary that targets the proposed
system, attention to every stage of the workflow, a preference for white-box
attacks by Kerckhoffs's principle, and it closes by calling an adversarial
evaluation a mandatory component rather than an add-on.

This project satisfies the core of it and has 1 gap left.
`paper/sections/background.tex` states attacker capability, defender knowledge
and the deployment decision point in 4 sentences.
`paper/sections/robustness.tex` reports a white-box adaptive attacker that
knows the probe and drives PSBD-TM from 0.966 to 0.233, which is the honest
result and better than what most defense papers report. The gap is that the
adaptive attacker was run against 1 probe and the recommended deployment is a
union of 3, so the deployed configuration is the 1 thing never attacked.

## Sampling bias, snooping and spurious correlation in detail

Pendlebury et al., "TESSERACT: Eliminating Experimental Bias in Malware
Classification across Space and Time", USENIX Security 2019, names the 2 biases
separately and gives constraints for each
(https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury,
https://www.usenix.org/system/files/sec19fall_pendlebury_prepub.pdf). Spatial
bias is a train and test distribution that does not match a real deployment,
temporal bias is an incorrect time split, and the paper shows both inflate
results on 3 Android classifiers over 129000 applications spanning 3
years. The transferable idea is the spatial constraint: the proportion of
positives in the test set must match the proportion expected in deployment, and
if it cannot, the paper's metrics must be stated as conditional on the test
proportion.

For this project the temporal half is vacuous and the spatial half is live. There
is no time axis in CIFAR or GTSRB, so nothing corresponds to temporal snooping.
The spatial constraint is violated on purpose by the 1 to 1 pairing, which is a
legitimate design for measuring a score's separating power and an illegitimate
one for reading a deployment number off it. The fix is to state the pairing as a
conditional and to convert once, explicitly, to a deployment prevalence.

Arp et al.'s selective snooping has 1 concrete analogue in this repository, and
it has already been refuted. Q34 tested whether dropping the modal perturbation
attractor from the clean validation split before reading the quantile threshold
is a free gain, and found the whole effect was threshold loosening worth -0.0004
on average over 706 cells. That is the correct outcome for exactly the kind of
whole-dataset cleansing the pitfall names.

The label-leakage question has a specific shape here. The threshold, the
disturbance rate and the placement are the 3 quantities that could leak, and the
first 2 are provably clean: `threshold_at_quantile` sees only `validation_psu`
and `select_rate_adaptively` sees only the clean-validation shift ratio, neither
of which contains a poison label. The third leaks through the human, since 27
placements were swept on the reported models before 1 was recommended, which is
Q15, Q16, Q20 and Q30 in 4 different forms.

## Base rate and class imbalance

Axelsson, "The base-rate fallacy and the difficulty of intrusion detection", ACM
TISSEC 3(3):186 to 205, August 2000, is the origin
(https://dl.acm.org/doi/10.1145/357830.357849), with an earlier version at ACM
CCS 1999 (https://dl.acm.org/doi/10.1145/319709.319710). The argument is that the
quantity a defender cares about is the Bayesian detection rate, the posterior
probability of an intrusion given an alarm, and that under a realistic base rate
this posterior forces a false alarm rate so low it may be unattainable. Axelsson
concludes that for a reasonable set of assumptions the false alarm rate, rather
than the detection rate, is the limiting factor for an intrusion detection
system.

Erbacher, "Base-Rate Fallacy Redux and a Deep Dive Review in Cybersecurity",
arXiv 2203.08801, re-examines the argument 20 years later and is useful for 2
things (https://arxiv.org/abs/2203.08801). It records that Axelsson defines the
false alarm rate as the false-positive rate, so the vocabulary matches this
project's. It also observes that machine learning detectors of 2022 sit mostly
at a false-positive rate of 1e-2 with rare exceptions at 1e-3, a factor of 10
better than 2000 while data volume grew by much more, and it argues that the
analyst cost of each false positive is what makes this binding.

The arithmetic for this project follows from 1 identity. With prevalence of
triggered inputs $\pi$, true-positive rate $TPR$ and false-positive rate $FPR$,
the precision is

$$\text{precision} = \frac{\pi \cdot TPR}{\pi \cdot TPR + (1 - \pi) \cdot FPR}$$

and the number of false alarms per true catch is $(1 - \pi) FPR / (\pi \cdot
TPR)$. The 3 operating points below are this project's own measured values at
the recommended placement under the adaptive 0.8 rule, read from
`paper/headline.tex`: TPR 0.602 at the 1% budget
(`\HeadlineTprAtOnePercent`), 0.766 at the 10% budget and 0.835 at the 20%
budget (`\StaircaseOperatorsBeforeAttentionNormTokenMaskTprAtOneZeroPercent` and
its 20% sibling).

| prevalence of triggered inputs | precision at 1% budget | precision at 10% budget | precision at 20% budget | false alarms per catch at 20% |
|---|---:|---:|---:|---:|
| 0.5, the paired evaluation | 0.984 | 0.885 | 0.807 | 0.2 |
| 0.1 | 0.870 | 0.460 | 0.317 | 2.2 |
| 0.01 | 0.378 | 0.072 | 0.041 | 23.7 |
| 0.001 | 0.057 | 0.008 | 0.004 | 239.3 |
| 0.0001 | 0.006 | 0.001 | 0.000 | 2395.0 |

The headline operating point is the third column. At the 1 to 1 prevalence the
evaluation uses it reads a precision of 0.807, and at a prevalence of 1 triggered
query in 1000 the same detector with the same TPR delivers a precision of 0.004,
which is 239 clean inputs rejected for every triggered input caught. The drop is
not a property of this detector. It is the identity above, and it is why
Axelsson's conclusion inverts the usual ranking of TPR against FPR.

The inverse direction is the number a deployer actually asks for. To reach a
precision of 0.5 at a prevalence of 0.001, the false-positive rate must fall to
8.4e-4 at the measured TPR of 0.835, which is 238 times tighter than the 20%
budget and 12 times tighter than the smallest budget the sweep currently reads.
To reach a precision of 0.9 at the same prevalence it must fall to 9.3e-5. A
1000-fold reduction in the false-positive rate is not a tuning exercise, and
saying so plainly is stronger than reporting a number that implies otherwise.

The workload framing that Arp et al. ask for is 1 sentence. At 100000
queries a day and a prevalence of 0.001 the 20% budget rejects 19980 clean
queries to catch 83 of the 100 triggered ones, while the 1% budget rejects 999
to catch 60. That comparison is what lets a reader decide whether the extra 23
catches are worth the extra 18981 rejections, and no reader can compute it from
AUROC.

## Metrics

Carlini et al., "Membership Inference Attacks From First Principles", IEEE S&P
2022, argues that aggregate metrics hide the thing a security evaluation is for
(https://arxiv.org/abs/2112.03570). The abstract's claim is that conventional
metrics fail to characterize whether the attack can confidently identify any
members of the training set, and the prescription is that attacks should instead
be evaluated by computing their true-positive rate at low, for example below
0.1%, false-positive rates. The companion device is the log-scale ROC curve,
which makes the low-FPR region readable at all.

The counter-arguments are real and 2 of them apply here. McDermott et al., "A
Closer Look at AUROC and AUPRC under Class Imbalance", NeurIPS 2024, shows that
the widespread claim that AUPRC is the better metric under imbalance is not
generally true, that AUPRC can favor improvements in subpopulations with more
frequent positives and so widen disparities, and that the claim itself is usually
made without citation or misattributed
(https://arxiv.org/abs/2401.06091,
https://proceedings.neurips.cc/paper_files/paper/2024/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html).
Their conclusion is that the metric follows from the use case rather than from a
blanket rule. Arp et al. themselves keep ROC and its AUC as useful for comparing
detection approaches, and warn that precision and recall mislead when the
minority prevalence is inflated by sampling bias, which is exactly the situation
a 1 to 1 pairing creates.

That gives the defensible position for this project's chosen 3 metrics. AUROC is
prevalence-free, which is the right property for the paper's actual question,
because the paper compares 27 placements of a probe across 69 cells and asks
which perturbation separates poisoned from clean inputs better. A
prevalence-dependent metric would make that comparison a function of an
arbitrary chosen prevalence, and Arp et al.'s own caution about inflated
minority prevalence argues against reading precision off a paired split. The 2
TPR points then carry the operational content AUROC cannot.

A paper reporting AUROC with TPR at 10% and 20% FPR owes a reader 6 things, all
of which come out of the sources above. State how the threshold is obtained, in
enough detail to reproduce it, which here is the quantile of 2000 clean
validation scores with the quantile equal to the budget. Report the realized
false-positive rate beside the nominal budget, per cell and as a spread, because
a quantile of 2000 samples applied to a different sample gives a random variable
and Q27 measures its range on this panel as 0.009 to 0.333. Say that the
evaluation population is 1 clean input to 1 triggered input and that no reported
number is a deployment precision. Give the base rate conversion once, as the
table above, so a reader can move from the reported TPR and FPR to a precision
at a prevalence of their choosing. Add the lowest budget the data already
supports, which is 1% and already computed. State what the detector is for at
these budgets, since a 20% rejection rate is an offline triage or a
human-in-the-loop filter rather than an inline firewall, and the paper should
say which.

There is a published precedent for a 10% budget in input-level backdoor
detection, and it is worth citing precisely because it is a comparison
convention rather than a deployment recommendation. Xie et al., "BaDExpert:
Extracting Backdoor Functionality for Accurate Backdoor Input Detection", ICLR
2024, fixes STRIP's FPR to 10% in its main comparison table, in its own words to
show its effectiveness, while fixing its own method's FPR to 1%
(https://arxiv.org/abs/2308.12439, https://arxiv.org/html/2308.12439v2). The
same paper states the threshold recipe this project uses, calculate scores on
the clean set and take the highest 1st percentile or any other FPR, and it says
that a deployer should anticipate robust defense at a low permissible FPR such
as 1% and that effectiveness improves as the permissible FPR rises. So a 10%
point is citable as a floor at which a weaker baseline is given its best
showing, and the 1% point is what the same literature treats as the deployment
number.

The surrounding convention is worth knowing. Gao et al.'s STRIP, ACSAC 2019,
reports a false acceptance rate under 1% at a preset false rejection rate of 1%
(https://arxiv.org/abs/1902.06531). BaDExpert reports 99.7% AUROC with more than
97% of backdoor inputs caught at under 1% FPR. SCALE-UP, ICLR 2023
(https://arxiv.org/abs/2302.03251) and IBD-PSC, ICML 2024
(https://arxiv.org/abs/2405.09786) report AUROC beside F1, so AUROC as a
headline is standard in this subfield and a 20% budget without a 1% companion is
not.

## Adaptive attackers and defense evaluation

Carlini et al., "On Evaluating Adversarial Robustness", arXiv 2019, is a living
checklist and its first instruction is to not follow the checklist mindlessly
(https://arxiv.org/abs/1902.06705, https://ar5iv.labs.arxiv.org/html/1902.06705).
The items that transfer to a backdoor input detector are these. Define a threat
model that fixes the attacker's goal, knowledge and capability, and for a
randomized defense make the threat model say what the attacker knows about the
randomness. Run adaptive attacks, against the complete end-to-end defense rather
than a component, with the loss adjusted so that success means what the defense
claims to prevent. Report clean accuracy with no attack present, and for a
defense that abstains or rejects, produce ROC curves. Apply the sanity checks:
verify that a stronger attack beats a weaker one, that more iterations do not
keep helping, that increasing the budget strictly increases attack success and
that the attack reaches 100% success somewhere on the curve. Ensemble properly
over randomness, and verify the attack succeeds when the randomness is fixed to
1 value. Run ablations that remove defense components, and attack a
similar-but-undefended model to prove the attack code works. Report per-example
results rather than only an average over attacks, and release models and code.

Tramer et al., "On Adaptive Attacks to Adversarial Example Defenses", NeurIPS
2020, is the empirical companion (https://arxiv.org/abs/2002.08347). They break
13 defenses published at ICLR, ICML and NeurIPS that already claimed adaptive
evaluations, and their conclusion is that typical adaptive evaluations are
incomplete rather than absent. The transferable lesson is that an adaptive
attack has to be designed per defense against the actual mechanism, and that
reusing a generic attack and reporting that it failed is the failure mode.

Zimmermann et al., "Increasing Confidence in Adversarial Robustness
Evaluations", NeurIPS 2022, adds the missing control
(https://arxiv.org/abs/2206.13991). The idea is an active test: construct inputs
that are known to be adversarial by construction, and check that the evaluation
flags them. An evaluation that cannot detect a defense known to be broken has
not measured anything, and the test is cheap relative to the evaluation it
validates.

For backdoor defenses specifically there are 3 sources. Veldanda and Garg, "On
Evaluating Neural Network Backdoor Defenses", arXiv 2010.12186, 2020, identifies
3 pitfalls and rates 5 defenses against them
(https://arxiv.org/abs/2010.12186). Pitfall 1 is insufficient evaluation over a
range of attack hyperparameters, and they find defenses that are sensitive to
something as incidental as the learning rate the BadNet was trained with.
Pitfall 2 is restrictive assumptions on backdoor structure and impact, meaning
triggers assumed small, of known shape, additive in pixel space or local, and
attacks assumed all-to-one. Pitfall 3 is failing to test adaptive attacks that
circumvent the defense's own explicit assumption. Their table marks fine-pruning
as carrying pitfall 2, Neural Cleanse as carrying 3, ABS and Generative
Modelling as carrying 1 and 3, and STRIP as carrying 1 and 3.

Qi et al., "Revisiting the Assumption of Latent Separability for Backdoor
Defenses", ICLR 2023, is the sharpest instance of pitfall 3 in this subfield
(https://arxiv.org/abs/2205.13613, https://iclr.cc/virtual/2023/poster/11430).
A family of defenses assumes poisoned and clean samples occupy separable latent
clusters, and the paper builds adaptive poisoning with correctly labeled
trigger-planted regularization samples and asymmetric trigger planting that
bypasses that family while keeping attack success and clean accuracy. The closing
line is a warning to defense designers who take a separability property as an
assumption, which is the position this project is in with respect to the
prediction shift.

Abad et al., "SoK: The Last Line of Defense: On Backdoor Defense Evaluation",
arXiv 2511.13143, November 2025, is the current survey of the field's evaluation
habits over 183 defense papers from 2018 to 2025
(https://arxiv.org/abs/2511.13143, https://arxiv.org/html/2511.13143). Its
findings are a list of the field's defaults. More than 73% of defenses are
evaluated only on dirty-label attacks and clean-label attacks are largely
ignored. Only 58% consider an adaptive attacker. More than 73% evaluate only on
MNIST or CIFAR-10, ImageNet-1K appears in about 20%, the architecture is
overwhelmingly a ResNet and transformers are largely unexplored. Most defenses
are tested only against visible patch triggers, 9 of 183 papers report any
stealthiness metric, execution time appears in about 5% and detection-based
defenses rarely report a false-positive rate at all. The recommended protocol
asks for diversity of model family and dataset complexity over sheer count,
consistent hyperparameter strategies across compared defenses, reporting that
includes execution time and clean accuracy impact, multiple runs with dispersion,
an adaptive adversary in every case, tuning validated across datasets and
stealthiness measured in input, feature and parameter space.

This project stands well against that survey and badly against 2 of its items.
It is a transformer paper in a field that is 73% CIFAR-10 ResNet, it includes
clean-label attacks, it reports execution time in seconds per input and it
reports a white-box adaptive attacker that beats the single probe. The 2 items it
fails are the ones the checklist below records: the adaptive attacker was never
run against the union that the paper recommends deploying, and there is no active
test proving the evaluation would notice a broken configuration.

## Statistical reporting over many comparisons

Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks",
MLSys 2021, models the whole benchmarking process and finds that data sampling,
parameter initialization and hyperparameter choice each move results materially
(https://arxiv.org/abs/2103.03098, https://arxiv.org/pdf/2103.03098). The
recommendation that matters here is that a comparison should randomize over the
sources of variation it wants to generalize over, and that adding more sources of
variation to an imperfect estimator approaches the ideal estimator more closely
than adding more repetitions of 1 source. Translated to this project, an interval
over models generalizes to new models and an interval over inputs generalizes
only to new inputs of the same models, so a claim about placements needs the
former.

Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 7,
2006, is the standard for the shape of this project's panel
(https://www.jmlr.org/papers/v7/demsar06a.html). The recommended device for
comparing several methods over several datasets is a rank-based omnibus test
followed by a post-hoc procedure that controls the family-wise error, rather than
a set of pairwise tests read independently. A paired difference with a bootstrap
interval per comparison, which is what this project does, is valid for each
comparison alone and says nothing about the family.

For the family, the 2 devices are Benjamini and Hochberg's false discovery rate
control (https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) and the replicability
framing of Dror et al., "Replicability Analysis for Natural Language Processing:
Testing Significance with Multiple Datasets", TACL 2017
(https://aclanthology.org/Q17-1033/), whose companion practical protocol is Dror
et al., "The Hitchhiker's Guide to Testing Statistical Significance in Natural
Language Processing", ACL 2018 (https://aclanthology.org/P18-1128/). The
replicability question is the one this panel actually asks: on how many of the 69
cells does the placement win, rather than what is the mean gain. Reporting a
count of wins with FDR control beside the mean paired difference answers both.

The reporting standard, as opposed to the mathematics, has 4 parts and they are
worth stating because the mathematics here is already correct. Declare which
single comparison is confirmatory and label the rest exploratory, which is Q16
and Q30 and which costs nothing because the winner's margin over the runner-up is
already indistinguishable from 0. Say what the interval is over, models or
inputs or seeds, which the NeurIPS checklist asks for in question 7
(https://neurips.cc/public/guides/PaperChecklist). Use independent randomness per
interval, since Q31 records that every bootstrap in the paper reuses seed 0 and
therefore shares resample indices at equal sample size, making Monte Carlo error
common-mode rather than independent evidence. Report the coverage each number is
read on, because Q26 records a table whose n column and AUROC column are over
each placement's own coverage while the gain column is over the intersection, so
differencing the printed AUROC column gives -0.052 against a true paired -0.063.

For a panel with cells excluded by a quality bar there is no single citation, and
the 3 sources above combine into 1 requirement. Arp et al.'s prohibition on
removing uncertain instances from test data, TESSERACT's spatial constraint and
the SoK's complaint about behavior when no backdoor is present all ask for the
same artifact: a flow from every cell trained to every cell reported, with the
reason for each exclusion, and a characterization of the excluded population.
This project has the flow, 105 cells to 71 clearing the bar to 69 comparable to
65 with the full basis, printed in `results/coverage/COVERAGE.md` and
`paper/sections/setup.tex`, and does not have the characterization. The 31
below-bar cells are models with a weak or absent backdoor, which is the false
alarm case a deployer meets most often, and Q28 records 2 clearing cells where
the deployable rule selects no rate at all.

## Reproducibility and artifact standards

ACM's "Artifact Review and Badging, Version 1.1"
(https://www.acm.org/publications/policies/artifact-review-and-badging-current)
defines 3 badge families. Artifacts Available means the artifacts are placed in a
publicly accessible archival repository with a DOI. Artifacts Evaluated comes in
Functional, meaning documented, consistent, complete and exercisable, and
Reusable, meaning above that norm and usable by others. Results Validated comes
in Reproduced, where the main results were obtained by a team other than the
authors using in part the authors' own artifacts, and Replicated, where a
different team obtained them without the authors' artifacts. Version 1.1 of
August 2020 swapped the definitions of Reproduced and Replicated to match the
NISO standard, so a paper citing an older badge means the opposite
(https://www.acm.org/publications/badging-terms,
https://sigir.org/general-information/acm-sigir-artifact-badging/).

USENIX Security's artifact evaluation uses 3 badges with concrete criteria
(https://secartifacts.github.io/usenixsec2024/badges). Artifacts Available
requires permanent public retrieval through a stable reference or DOI, rejects
personal websites, accepts Zenodo, FigShare, Dryad, Software Heritage, GitHub and
GitLab, and requires a commit hash or tag rather than a bare repository link for
an evolving repository. Artifacts Functional requires documentation sufficient
for a reader to exercise the artifact, inclusion of every key component the paper
mentions and the scripts and data needed to run the experiments. Results
Reproduced requires that a reviewer can obtain the paper's main results from the
submitted artifacts within an allowed tolerance, with the goal of validating the
main claims rather than bit-exact output.

The NeurIPS paper checklist adds the per-experiment requirements
(https://neurips.cc/public/guides/PaperChecklist). Question 6 asks for all
training details including data splits, hyperparameters and how they were chosen.
Question 7 asks for error bars, confidence intervals or significance tests on the
experiments that support the main claims, and specifically for a statement of
what variability the error bars capture and how they were computed. Question 8
asks for compute resources per experiment and in total, including the compute
spent on runs that did not make it into the paper. The reasoning behind the
program is in Pineau et al., "Improving Reproducibility in Machine Learning
Research", JMLR 22, 2021 (https://www.jmlr.org/papers/v22/20-303.html).

What a repository must ship to clear all 3 at once is short. A tag or commit hash
archived with a DOI, covering the code, the configurations and the seeds. A
documented path from a fresh checkout to each headline number, which this project
has in `scripts/paper/` reading `results/coverage/coverage.json`. Every published
number traceable to the commit that produced it, which is where Q9, Q10, Q11 and
Q14 bite: 546 of 1922 checkpoint sidecars carry a null commit, 271 more carry a
dirty tree, 2693 of 23703 sweep provenance records were written against a dirty
tree, 1 referenced commit is unreachable from any branch and all 54 tables and 10
figures were generated from dirty trees across 10 different commits. A total
compute figure, which the paper does not currently give. Every artifact a table
reads present on disk, which Q7 and Q8 fail, since `results/psu_vs_confidence.json`
and `results/backdoor_neuron_ablation.json` do not exist and 15 macros plus the
flagship mechanism table cannot be regenerated.

## Checklist

Verdicts are read from the repository on 2026-09-23. A Q number points at the
canonical entry in `docs/open-questions.md`, which this table does not restate.

| Requirement | Satisfied | Fix |
|---|---|---|
| Threat model fixes attacker capability, defender knowledge and the decision point (Arp P10, Carlini 2019) | yes, `paper/sections/background.tex` | none |
| Defender touches no poisoned data and no training set | yes, `data/splits.py` builds from the clean test set only | none |
| Threshold calibrated on a split disjoint from the evaluation pool (Arp P5) | yes, `threshold_at_quantile` reads `validation_psu`, split at `PSBD_HELDOUT_SIZE` before anything else | none |
| Disturbance rate chosen without poison labels (Arp P5) | yes, `select_rate_adaptively` reads the clean-validation shift ratio | none |
| Placement chosen on the declared selection half (Arp P3 test snooping) | **no**, the protocol in `configs/psbd_basis.json` selects a different placement | Q15, Q20. Add the Swin twin row and the re-selection experiment, both already in `results/` |
| 1 confirmatory comparison declared, the rest labeled exploratory (Demsar) | **no**, 27 placements ranked with no label | Q16, Q30. Label the ranking table exploratory. Costs no compute |
| Multiplicity control over the placement family (Benjamini and Hochberg, Dror et al.) | **no** | Q30. Report win counts with FDR control beside the mean paired difference |
| Interval says what it is over, and is over models for a claim about models (Bouthillier, NeurIPS Q7) | yes, paired over models with 5000 resamples, stated in `paper/sections/setup.tex` | none |
| Independent randomness per bootstrap interval | **no**, every interval reuses seed 0 | Q31. Seed per comparison. Costs no compute |
| Table columns a reader can difference (matched coverage) | **no** | Q26. Print the paired gain and its own n, or restrict the AUROC column to the intersection |
| The matched-shift comparison device is actually matched | **no**, off target by more than 0.10 in 34 of 71 cells | Q25. `interpolate_at_target_shift` exists and has no consumer outside the tests |
| Realized false-positive rate reported beside the nominal budget (Arp P7, Carlini 2022) | **no**, `detection_report` returns `fpr` and no table in `paper/` prints it | Q27. Add a realized-FPR column with its spread. Costs no compute |
| Tie artifacts at the threshold checked for the headline method | **no**, `threshold_diagnostics` is called only from `cli/baselines.py` and the tests | Call it for PSBD in `cli/analyze.py` and report `tie_share_at_threshold` |
| TPR at a low budget of 1% or below reported (Carlini 2022, BaDExpert, STRIP) | **no**, the paper reports 10% and 20% only | Q17. `\HeadlineTprAtOnePercent` already reads 0.602 and `fig_shift_ladder.py` already uses the `q0.01` block. Promote it into the headline table |
| Evaluation prevalence stated, and stated as not a deployment number (Arp P8, TESSERACT spatial) | **no** | Add 1 sentence naming the 1 to 1 pairing as a conditional |
| Base rate conversion given so a reader can reach a deployment precision (Arp P8, Axelsson) | **no** | Add the prevalence table of this document to the setup or the limitations section |
| Precision or AUPRC computed at all | **no**, `configs/psbd_basis.json` lists `auprc` in `required_metrics` and nothing in `defenses/` computes it | Add `auprc` and `achieved_fpr` to `detection_report`, or drop them from the declared protocol |
| Bounded AUC or log-scale ROC for the low-FPR region (Arp P8, Carlini 2022) | **no** | Bound the AUC at the largest budget reported, or plot 1 log-scale ROC. The cached per-pass probabilities support both |
| Deployment mode named, given a 20% rejection rate | **no** | Say whether the 20% point is offline triage or a human-in-the-loop filter |
| Published method reimplemented and run on every cell (Arp P6) | yes, `post_residual` is swept per cell as its own baseline | none |
| State-of-the-art competitors on the reported panel (Arp P6) | **no**, 3 GTSRB models with 500 validation images | Finish the `cli.baselines` run on the 69 comparable cells |
| A simple baseline of unperturbed confidence or entropy (Arp P6) | **no**, the artifact is missing | Q8. Regenerate `results/psu_vs_confidence.json` on a GPU |
| Benign-model control that can fail (Arp P4) | **no**, a benign model reads about 0.5 under either sign | Q22. Replace with the backdoored-model reading |
| Behavior on models with a weak or absent backdoor reported (SoK, Arp P2) | **no**, the 31 below-bar cells are counted and not characterized | Read the detector on the below-bar cells and report the false alarm rate there |
| Cells where the deployable rule selects nothing reported | **no** | Q28. 2 of 71 cells. 1 sentence in the limitations section |
| Adaptive white-box attacker against the defense (Arp P10, Carlini 2019, Tramer) | yes, `paper/sections/robustness.tex`, PSBD-TM falls to 0.233 and the paper says so | none |
| Adaptive attacker against the configuration the paper recommends deploying (Tramer, Veldanda pitfall 3) | **no**, the union of 3 probes is never attacked | Train the hinge-loss attacker against the min-rank union. Needs GPU |
| Non-adaptive evasion, a trigger built to survive token masking | **no** | Already the mentor's standing request, shortlist at `literature/README-attack-survey-2026-09-23.md` |
| Attack strength swept over hyperparameters (Veldanda pitfall 1) | partly, 3 poison rates per attack and 1 trigger configuration each | Vary trigger size or blend ratio on 1 attack to show the detector is not tuned to 1 setting |
| Attack families beyond local patches, including warping and global and source-specific (Veldanda pitfall 2, SoK) | yes, 9 attacks spanning patch, blend, low-frequency, warping, source-specific and clean-label | none |
| All-to-all label mode evaluated rather than assumed away (Veldanda pitfall 2) | yes, and reported as a failure at 0.411 in `paper/sections/limitations.tex` | none |
| Active test proving the evaluation would flag a broken configuration (Zimmermann) | **no** | Run the pipeline on a known-inverted placement and show it reads below 0.5, or on a benign model with a fake manifest |
| Attack success and clean accuracy of every evaluated model reported (Carlini 2019, SoK) | yes, `results/coverage/COVERAGE.md` and `app:results` | none |
| Detector runtime cost reported (SoK, Arp P9) | yes, forward passes and seconds per input in `paper/sections/appendix.tex` | none |
| Total compute for the project reported (NeurIPS Q8) | **no**, per-model training time only | Sum the PBS accounting into 1 figure |
| Dataset limits stated openly (Arp P1) | yes, first paragraph of `paper/sections/limitations.tex` | none |
| A dataset outside the small-image regime (SoK) | **no**, 6 datasets all upscaled to 224 pixels | Out of scope for this deadline. State it as a boundary rather than fixing it |
| Multiple training seeds with dispersion (SoK, Bouthillier) | partly, 3 seeds on 14 cells moving PSBD-TM by 0.012 | Report the 14-cell seed spread beside the headline rather than only in the robustness section |
| Every published number traceable to a clean commit (ACM, USENIX Available) | **no** | Q9, Q10, Q11, Q14. Regenerate all artifacts from 1 clean commit before submission |
| Every artifact a table reads present on disk (USENIX Functional) | **no**, 2 records missing and 15 macros unregenerable | Q7, Q8. Needs GPU |
| Archived release with a DOI or tag (ACM Available, USENIX Available) | unknown, not checked | Tag the submission commit and deposit on Zenodo |
| Splits, hyperparameters and how they were chosen documented (NeurIPS Q6) | yes, `configs/psbd_basis.json` plus `app:protocol` | none |
| Competitor ports cross-checked numerically against their reference (Arp P6) | yes, required by `CLAUDE.md` and present per detector | none |

## The 5 mistakes this project is most at risk of

**1. Reading a deployment claim off a balanced evaluation.** The pairing in
`pair_clean_to_backdoor` is the correct control for isolating the trigger and it
fixes the prevalence at 0.5, so every reported TPR and FPR is conditional on a
population no deployment has. A reviewer who computes the table in the base rate
section gets a precision of 0.004 at a prevalence of 0.001 and 239 false alarms
per catch, and the paper currently gives that reviewer no sentence to land on.
This is first because it is the cheapest to fix and the most damaging if a
reviewer finds it before the authors state it.

**2. Reporting a fixed-FPR number whose realized FPR is a 38-fold range.** The
threshold is a quantile of 2000 validation scores and the false-positive rate is
realized on a different sample, so Q27's measured span of 0.009 to 0.333 means
the headline TPR at a 10% budget averages true-positive rates measured at
false-positive rates from 1% to 33%. `detection_report` already returns the
realized `fpr` and no table prints it, so the paper asserts a matched comparison
it has the data to show is unmatched. This is second because the defect is in the
central number and the fix needs no compute.

**3. A placement chosen on the models it is reported on.** `configs/psbd_basis.json`
declares selection on CIFAR-10 and GTSRB and reporting on CIFAR-100 and Tiny, and
Q20 shows the recommended placement loses -0.029 to its twin on the selection
half and wins +0.028 on the reporting half, so the protocol applied literally
picks the twin. Arp et al. name this as test snooping and rate it the second most
prevalent pitfall in the field at 73%. It is third rather than first because the
gain over the published placement survives every re-derivation in the audit, so
what is at risk is the specific recommendation rather than the paper's result.

**4. Claiming adaptive robustness for a configuration that was never attacked.**
The paper honestly reports that a single probe falls to 0.233 under a white-box
attacker and then recommends a union of 3 probes that restores 0.967, with the
attacker on the union named as untested in `paper/sections/limitations.tex`.
Tramer et al. broke 13 defenses that already claimed adaptive evaluations, and Qi
et al. broke an entire defense family that took a separability property as an
assumption, which is the assumption the prediction shift is. The risk here is
specific: the adaptive section's structure invites a reader to treat the union as
evaluated, and the honest framing needs to be louder than the number.

**5. A paper whose numbers cannot be rebuilt.** All 54 tables and 10 figures came
from dirty trees across 10 different commits, 546 checkpoint sidecars carry no
commit at all, 2 result records that 15 macros and the flagship mechanism table
read are absent from `results/`, and the headline macros no longer reproduce from
the current tree because 4 new cells landed. USENIX's Results Reproduced badge
asks a reviewer to obtain the main results from the submitted artifacts, and this
repository would fail on the missing files before reaching the tolerance
question. It is last because it is bookkeeping rather than a claim, and it is on
the list because the deadline is the thing that makes bookkeeping slip.
