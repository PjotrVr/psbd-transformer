# Reading our draft against Picek's house style

## The house style, in 1 page

Sources read for this note. The local LaTeX source of "Backdoor Directions in Vision
Transformers" by Karayalcin, Krcek, Chen and Picek
(`/lustre/home/pstika/projects/PSBD-ViT/papers/backdoor_directions/main.tex`), the PSBD
paper by Li, Chen, Liu and Wang
(`/lustre/home/pstika/projects/PSBD-ViT/papers/PSBD/main.tex`, the method we adapt and a
Pin-Yu Chen paper in its own right), and 3 recent Picek-group papers fetched from arXiv:
"Towards Backdoor Stealthiness in Model Parameter Space" (Grond, CCS 2025,
https://arxiv.org/abs/2501.05928), "Context is the Key: Backdoor Attacks for In-Context
Learning with Vision Transformers" (https://arxiv.org/abs/2409.04142) and "Removing the
Trigger, Not the Backdoor: Alternative Triggers and Latent Backdoors"
(https://arxiv.org/abs/2603.09772).

The introduction runs 3 to 8 paragraphs and follows the same shape every time. It opens with
the general threat (what backdoors are, why they matter), narrows to what current defenses
miss, states the paper's angle in 1 or 2 sentences, and closes with a numbered contributions
list that names the mechanism ("we obtain the backdoor direction", "we propose Grond") and
points at a later section or table. The threat model gets its own paragraph or subsection,
stated as what the attacker controls and what the defender has access to, usually before any
method detail. Grond states it formally in a dedicated Section 3.1. The in-context paper
states it inline in the introduction, naming the 2 threats by number.

A table or figure is always introduced by a sentence that says what it will show before the
numbers appear. Grond: "as demonstrated in Tables 2 and 3." The in-context paper: "For the
backdoor performance, we observe an improvement compared to the task-specific attack even in
the training task, see Table 4." Paragraphs run 3 to 8 sentences, longer in method and result
sections, shorter in the introduction. Hedging is measured and specific. "We conjecture",
"we hypothesize" and "this is consistent with" appear where a mechanism is proposed but not
fully nailed down, and plain declaratives appear everywhere else. Nobody hedges a number, only
an explanation.

Related work is grouped by topic, never paper by paper, under paragraph headers close to ours:
Grond has "Backdoor Attacks" split into input-space, feature-space and supply-chain, and
"Backdoor Defenses" split into detection, mitigation and proactive defense. PSBD has
"Backdoor Attacks" and "Backdoor Defenses" as its 2 paragraphs, the second closing with a
paragraph naming the "consistency family" PSBD itself belongs to. Every paper read here
says plainly what is new relative to the nearest prior work, in the same paragraph that names
it, and reserves "first" for a claim actually checked against that nearest work.

Abstracts run 4 to 7 sentences. PSBD's names the phenomenon, the statistic and the state of
the art claim, no numbers. Grond's abstract carries 2 numbers (12 attacks, 17 defenses) and a
qualitative "outperforms all 12" claim. The in-context paper's abstract carries 4 numbers,
including a percentage of the training set touched, because that number is the paper's
central claim. Limitations get 1 short paragraph, near the end, phrased as a scope statement
rather than an apology, and often as the frankest sentence in the paper. Grond: "TAC can only
be used to analyze backdoor behaviors and cannot be deployed as a practical defense." The
in-context paper: "We found that these methods fall short." Methods and placements get short,
literal names (WaNet, BPP, Grond, FGA), used identically everywhere after their first
definition, never a description that grows or shrinks between sections.

## Concrete faults, ordered by importance

1. **The macro layer that is supposed to carry every headline number is not wired into the
   compiled paper, so nothing in `main.pdf` is actually generated from it.**
   `paper/headline.tex` defines 311 `\newcommand` macros (`\HeadlineAurocAdaptive` at 0.935,
   `\PublishedAurocAdaptive` at 0.832, and so on), each with a comment naming its source and
   generator, exactly as the project's own constitution promises ("Every headline number below
   is a macro in `paper/headline.tex`"). `main.tex` never inputs `headline.tex`, `preamble.tex`
   never inputs it either, and a search of every section file for any of those macro names
   (`grep -rno '\\\\[A-Z][A-Za-z]*' sections/*.tex`) returns nothing. Every number a reader sees
   in `sections/00-abstract.tex` through `12-appendix.tex` is a hand-typed literal. The 3
   numbers spot-checked against `headline.tex` still agree today, but nothing enforces that
   after the next sweep. Wire `\input{../headline}` into `preamble.tex` and replace the literal
   numbers in the headline table and the abstract with the macros, starting with
   `\HeadlineAurocAdaptive`, `\PublishedAurocAdaptive` and `\HeadlineTprAtOnePercent`, so a
   changed sweep result cannot silently leave the compiled paper behind.

2. **The auto-generated table captions still call the panel's unit "cell" and name placements
   by their code identifiers, in direct conflict with the prose that surrounds them.**
   `paper/tables/basis_ranking.tex`: "The floor is the worst single cell and an inversion is a
   cell below chance." `paper/tables/adaptive_attacker.tex`: "cells whose evasive checkpoint
   keeps attack success at or above 0.9." 30 of the 45 files under `paper/tables/` use "cell"
   at least once, while every section from `00-abstract.tex` to `12-appendix.tex` says "model"
   throughout ("65 backdoored ViT-B/16 models"). The worst instance is `paper/tables/headline.tex`
   itself, whose caption names the 2 compared placements as `` `before_attention_norm_token_mask` ``
   and `` `post_residual` ``, the raw config identifiers, when every section calls them
   attention-input token masking and post-residual dropout. Rewrite each caption to say "model"
   and to use the placement's prose name, the same rename the sections already carry.

3. **`tab:headline` is defined twice, and only luck keeps the compiled paper from breaking.**
   `sections/05-results.tex` writes its own `\begin{table}...\label{tab:headline}...\end{table}`
   with 2 columns. `paper/tables/headline.tex` also writes `\label{tab:headline}`, on an
   unrelated 25-column quantile sweep, and is never `\input`. Nothing enforces that it stays
   that way, and the file's own name is the first thing anyone finishes the appendix tables and
   reaches for. Either delete the orphaned `tables/headline.tex` and its 29 siblings that are
   never referenced from `main.tex`, or give it a distinct label (`tab:headline-quantiles`)
   before someone inputs it and gets 2 Table 1s.

4. **"Hard attacks" is defined once and then quietly narrowed, so every "hard attacks" number
   in the results rests on a set the reader can only reconstruct by cross-referencing the
   appendix.** `sections/04-setup.tex`: "We call BadNets, Blend and LF the easy attacks and
   BPP, WaNet, TaCT, SIG, Label-Consistent and Adaptive-Blend the hard ones." That is 6 named
   attacks. `sections/12-appendix.tex`: "The hard-attack subset of the main text is therefore
   BPP, WaNet, TaCT and SIG." That is 4, silently dropping Label-Consistent and Adaptive-Blend
   with no pointer back to the setup section's own list. A reader of `\cref{tab:gains}`'s "hard
   attacks, 29 models, +0.110" in `sections/05-results.tex` has no way to know, from that
   section, which of the 6 defined hard attacks the 29 models are drawn from. State the
   effective 4-attack set at the point of first use in `04-setup.tex`, or name Label-Consistent
   and Adaptive-Blend's absence explicitly wherever "hard attacks" is first computed on.

5. **Adaptive-Blend, the 1 attack this project selected specifically because it targets
   input-level detectors, never appears in a single result, and that gap is never named in
   Limitations.** `sections/10-related.tex` calls it "an attack designed against input-level
   detectors", precisely PSBD's own category. `sections/12-appendix.tex` reports, in a single
   clause inside a paragraph about a different table, that "Adaptive-Blend never reaches the
   bar." `sections/09-limitations.tex` lists all-to-all attacks, the 2 failing models,
   clean-label caps, seed coverage, SAM and adaptive attackers as scope boundaries, but never
   mentions that the panel's dedicated adversarial-attack baseline produced 0 usable models. A
   reviewer who reads only the abstract and limitations, as most do, would not learn this. Add
   1 sentence to `09-limitations.tex` naming Adaptive-Blend's absence as a boundary of the
   evidence, for example: "Adaptive-Blend, the attack in the panel built to evade input-level
   detectors, never reaches the attack-success bar at any poison rate we trained, so none of
   the numbers above include it."

6. **ASR is used as a bare column header in several tables with no expansion in that table or
   the running prose that precedes it.** `paper/tables/clean_label_sig_gtsrb.tex` and
   `paper/tables/clean_label_new_datasets.tex` both header a column "mean ASR" and neither file
   contains the phrase "attack success" anywhere to let a reader reconstruct the acronym.
   `paper/tables/clean_label_multitarget.tex`, `adaptive_attacker.tex`, `direction_ablation.tex`
   and `seeds.tex` do the same, relying on a caption elsewhere in the same file that says
   "attack success" without ever writing "(ASR)". No section in `00-abstract.tex` through
   `12-appendix.tex` writes "attack success rate (ASR)" either, so the acronym has no first
   definition anywhere a reader would meet it before a table. Add "(ASR)" once, in
   `sections/04-setup.tex` at "A backdoored model enters the evaluation only if its attack
   success rate is at least 0.85", and add it to the 2 captions that carry no expansion at all.

7. **A number is reported with no indication which placement it belongs to.**
   `sections/07-swin.tex`: "WaNet is detected worse on Swin than on ViT, at 0.26 to 0.29 against
   0.51 to 0.55 with dropout at the matched rate on CIFAR-100." The sentence reads as a
   Swin-versus-ViT comparison of the recommended placement, but "with dropout" then names a
   different placement for at least 1 side, and it never says which side. Rewrite, for
   example: "Token masking detects WaNet worse on Swin than on ViT, at 0.26 to 0.29 against
   0.51 to 0.55, both at the matched rate on CIFAR-100 with post-residual dropout as the
   reference."

8. **An ambiguous adverb placement in the introduction survives from an earlier pass.**
   `sections/01-introduction.tex`: "We show once, in \cref{sec:results:headline}, where that
   difference comes from." "Show once" reads as a slip before it reads as "this is the only
   place we make the comparison." Reword: "\Cref{sec:results:headline} is the only place we
   make this comparison, and it shows where the difference comes from."

9. **A repeated preposition forces a reread in the setup section.**
   `sections/04-setup.tex`: "Choosing the best of 18 placements on the same models it is then
   reported on would be an optimistic reading." Reword: "Choosing the best of 18 placements on
   the models it is later reported on would give an optimistic reading."

10. **A banned contrastive construction splits across 2 sentences instead of 1, which still
    reads as the same pattern.** `sections/06-mechanism.tex`: "...and it is not what makes the
    statistic work in general. What makes it work in general appears to be the margin." State
    the claim directly: "The margin appears to be what makes the statistic work in general. A
    triggered input's predicted class has a large margin over the rest because the shortcut is
    firm, and any perturbation that moves clean decisions more than firm ones separates the
    populations, whether or not the clean decisions move toward the target."

11. **A softer version of the same pattern opens the headline comparison.**
    `sections/05-results.tex`: "It works, but it is not the best site a ViT offers." State the
    finding directly: "A ViT offers a better site. Masking whole tokens at the input of every
    attention block, before its LayerNorm, is the placement we recommend."

12. **The winning placement carries no short name, unlike every named method in the papers we
    are measured against (WaNet, Grond, FGA, PSBD itself).** "Attention-input token masking"
    and "post-residual dropout" are precise but run 4 and 3 words and appear dozens of times
    across the draft. This is a defensible choice, since the project moved away from internal
    codenames on purpose, but it costs sentence economy everywhere the phrase repeats twice in
    1 sentence, for example `sections/07-swin.tex`: "Moving dropout, the paper's perturbation,
    from the residual stream to the attention input gains nothing... Moving to token masking at
    that same input gains +0.091." A single short name, even just "the winning placement" on
    second reference within a paragraph, would read faster without reintroducing a codename.

13. **"Logit lens" is used twice with no citation attached to either use.**
    `sections/08-robustness.tex`: "a logit lens that costs 1 pass." `sections/10-related.tex`:
    "Activation patching \cite{zhang2024patching} and the logit lens are the tools we use to
    locate the backdoor by depth." Activation patching gets a citation in the same sentence,
    the logit lens does not, though it is exactly as load-bearing a tool. Attach its source at
    first use in `08-robustness.tex`.

14. **The abstract's first sentence describes prior work, not this paper's contribution,
    which delays the reader's answer to "what is new here" past sentence 2.**
    `sections/00-abstract.tex`: "Prediction Shift Backdoor Detection (PSBD) flags a poisoned
    input by how little its prediction moves when the network is perturbed with dropout at
    inference." This is PSBD's claim, not ours, and only "We adapt it to Vision Transformers"
    in sentence 3 makes the paper's own contribution explicit. Every Picek-group abstract read
    for this note puts a first-person contribution verb in its first or second sentence.
    Grond states "we propose Grond" and the in-context paper opens with the general threat
    then names its own 2 new attacks by sentence 2. This is a smaller issue than the others
    above, since
    "It was designed for ResNets" already signals PSBD is prior work by sentence 2, but tightening
    it removes any ambiguity for a reader who does not already know PSBD.

## What a reviewer from this community would object to in the methodology

The competitor-detector comparison, the number every input-level detection paper leads with,
is a 3-model smoke test. `sections/08-robustness.tex`: "The full comparison over the 71
backdoored models, which is where the hard attacks decide the ranking, is running." A reviewer
would read this as "the paper has not yet run the comparison its own title promises," since
the smoke test's own text admits "On an easy patch attack PSBD has no edge", leaving the
question the hard attacks answer completely open at submission time.

The Swin transfer, 1 of the paper's 4 stated contributions, rests on 5 of the 18 placements
in the basis rather than a full sweep. `sections/07-swin.tex`: "the Swin models have not been
swept over the full 18-placement basis, so the Swin ranking rests on 5 placements rather than
18. That sweep is in progress." The 0.968 AUROC headlined for Swin in the abstract and
conclusion is therefore the best of 5 candidates, not the best of 18, and could move once the
sweep finishes. The abstract states 0.968 with no such qualification.

The adaptive-attacker result, which the conclusion generalizes to "an attacker who evades 1
probe does not evade the others", is measured on 1 architecture, 1 dataset and 5 attacks.
`sections/08-robustness.tex`: "trained jointly with the backdoor on CIFAR-100 for BadNets,
Blend, BPP, LF and Adaptive-Blend at 3 poison rates, 14 backdoored models in all." The
conclusion's claim in `sections/11-conclusion.tex`, "The union of probes costs a few forward
passes per input and restores detection above 0.97 on every attack we trained against a single
probe", is scoped correctly inside that sentence, but the abstract's "a union of probes
restores detection" carries no dataset or architecture qualifier at all, which is broader than
what was tested.

The forward-pass convergence claim that justifies keeping $k=3$ is checked on 6 models, all at
1% poisoning and all easy attacks (BadNets, Blend and LF on CIFAR-100 and Tiny ImageNet).
`sections/08-robustness.tex` says as much: "Whether the hard attacks at 1\% behave the same is
being measured, since their $k=20$ sweeps had not finished when this draft was compiled." A
reviewer would ask why the paper's own $k=3$ choice, carried over unchanged from PSBD, is
validated only on the subset where the method already works best.

The dead macro layer described as fault 1 above is also a methodology objection in its own
right, not just a prose issue. This project's own standard states that "a number in `docs/`
that cannot be traced to a commit is not a result", and the entire point of `paper/headline.tex`
and `paper/tables/macro_ledger.tex` (which grades every macro STRONG, SUPPORTING or
unregistered) is to make that traceability mechanical. A reviewer who noticed the macros were
unused in the compiled PDF would have grounds to ask whether every number in the paper still
matches the ledger, since nothing in the build enforces that they do.

## What the draft already does well

The threat model gets its own named subsection early, in `sections/02-background.tex`, stated
as what the attacker controls and what the defender has, exactly where Grond and the
in-context paper put theirs. The contributions list in `sections/01-introduction.tex` is 4
items, each naming a specific number and a section, matching the Picek-group convention of a
contribution that points at where it is proven rather than asserting it. Related work in
`sections/10-related.tex` is grouped by topic under paragraph headers (Attacks, Input-level
detectors, Training-time detection and removal, Mechanistic work on ViT backdoors), the same
shape as PSBD's and Grond's related-work sections, and it says plainly what is new relative to
Karayalcin et al. rather than leaving the reader to infer it.

Every gain in the results is a paired difference with a 95% bootstrap interval over models, and
seed variance is measured and reported separately from model-to-model variance
(`sections/08-robustness.tex`, `app:seeds`), which is more disciplined than most of the papers
read for this note, several of which report only point estimates. The limitations section
names negative and null results as their own items, the all-to-all failure, the 2 inverted
models, the clean-label rate caps and the SAM null result, rather than folding them into a
generic future-work paragraph, which is closer to what NeurIPS and ICLR reviewer guides ask
for than what the Discussion sections of the papers read here actually do. Finally, the
mechanism section states its account first in 1 paragraph and then tests it, including
against a closest competing account (Karayalcin et al.'s), reproducing their removal result
and their layer-ordering claim on 2 of 3 datasets while flagging where GTSRB reverses the
order, which is a genuine falsification exercise against prior work rather than a citation
that merely agrees with it.
