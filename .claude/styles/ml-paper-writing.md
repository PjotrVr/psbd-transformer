# Writing a machine learning research paper

Distilled from the sources below plus this project's `writing-style.md`. Read that file
first. Everything here inherits its rules: no semicolons, no em dashes, no arrows, no
hyphens as sentence punctuation, no Oxford comma, digits not words, noun-phrase headers.

## 1. Sources, in 1 sentence each

- Lipton and Steinhardt, "Troubling Trends in Machine Learning Scholarship"
  (https://arxiv.org/abs/1807.03341). Explanation dressed as speculation, unattributed
  sources of empirical gain, mathiness and misused language mislead readers even when no
  claim is technically false, and each has a 1-line fix, do not do it.
- Steinhardt, "Advice for Authors" (https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors).
  Local sentence-level precision, conciseness and consistency carry more of a paper's
  readability than its global structure does.
- "Highly Opinionated Advice on How to Write ML Papers"
  (https://www.lesswrong.com/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).
  Compress the paper into 1 to 3 falsifiable claims first, then build the abstract,
  introduction, experiments and limitations to support exactly those claims and nothing
  broader.
- Farquhar, "How to Write ML Papers"
  (https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/). Figure 1
  and the abstract are read by far more people than the rest of the paper, so budget
  writing time in proportion to readership, not in proportion to page count.
- Peyton Jones, "How to Write a Great Research Paper" (https://simon.peytonjones.org/great-research-paper/).
  Write the paper before the work is finished, since writing is what forces the idea into
  a shape clear enough to attack, and a paper exists to move 1 idea from your head to the
  reader's.
- Karpathy, "A Survival Guide to a PhD" (https://karpathy.github.io/2016/09/07/phd/).
  Reviewing bad papers teaches what to avoid faster than reading good ones does, and a
  paper should carry exactly 1 contribution, organized around it surgically.
- Nielsen, "Principles of Effective Research" and "Writing"
  (https://michaelnielsen.org/blog/principles-of-effective-research/,
  https://michaelnielsen.org/blog/writing/). Clarity about the goal, even an imperfect
  goal, builds the forward momentum that most research and most writing actually run on.
- Widom, "Tips for Writing Technical Papers" (http://infolab.stanford.edu/~widom/paper-writing.html).
  A reader should be able to state the paper's technical contribution by the end of page 3,
  and every section should read as its own small story.
- Black, "Writing a Good Scientific Paper" (https://perceiving-systems.blog/post/writing-a-good-scientific-paper).
  A paper is a goal, a problem that blocks the goal and a solution, repeated at every
  scale, and its nugget is the 1 insight that turns an unsolvable problem into a solvable
  one.
- Parikh, Batra and Lee on rebuttals and reviewing
  (https://deviparikh.medium.com/how-we-write-rebuttals-dc84742fece1,
  https://deviparikh.com/citizenofcvpr/). Write as if the reader remembers nothing about
  your paper and will not read it a second time, so re-establish notation and setup rather
  than assume it survived from an earlier read.
- The Heilmeier Catechism (https://www.darpa.mil/about/heilmeier-catechism). State the goal,
  the current limit, what is new, who cares, the risks and the test for success, in that
  order and in language with no jargon.
- Perez, "Easy Paper Writing Tips" (https://ethanperez.net/easy-paper-writing-tips/). Put
  the verb early, cut pronouns and hedge words, and make every sentence carry new
  information or cut it.
- Jason Wei on abstracts and introductions (https://www.jasonwei.net/blog). Most readers
  see only the abstract, the introduction and a screenshot of 1 figure, so write those 3
  for a reader who is not a native English speaker and will read nothing else.
- NeurIPS Reviewer Guidelines (https://neurips.cc/Conferences/2026/ReviewerGuidelines).
  Reviewers reward authors for stating limitations plainly and check whether the claims
  match what the theory or the experiments actually support.
- ICLR Reviewer Guide (https://iclr.cc/Conferences/2026/ReviewerGuide). A review should
  check the paper's central claims in enough detail to be convinced of them, not raise
  points that would not change the accept or reject decision.
- Kording and Mensh, "Ten Simple Rules for Structuring Papers"
  (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619). Every
  unit from sentence to abstract should run context, content, conclusion, and the title,
  abstract, figures and outline are where the writing time should go.
- Olah and Carter, "Research Debt" (https://distill.pub/2017/research-debt/). Unclear
  exposition is a debt the whole field pays interest on, and distilling an idea to its
  clear form is its own kind of research contribution.
- Whitesides, "Whitesides' Group: Writing a Paper"
  (https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper). Draft the
  figures and the outline before the experiments are finished, and expect the readable
  paper to be the product of 10 or more rewrites, not the first draft.

## 2. Section checklist

### Abstract
- State the achievement in 1 sentence, then why it is hard or why it matters, then the
  approach, then the evidence, in roughly that order.
- Every number in the abstract should also appear, with its interval or its comparison, in
  the body.
- No forward references, no citations, no undefined acronyms.

### Introduction
- Open with the problem and why a reader who does not already work on it should care.
- State what is hard about it and why it has not been solved, before stating the fix.
- Reveal the contribution by the end of page 1. Do not make the reader wait for Section 4.
- Close with a short, numbered list of contributions, each pointing at the section that
  supports it.

### Contributions
- 1 to 3 claims, each specific enough to be false. "We study X" is not a claim.
- Each claim should name the section and, where it exists, the number that supports it.
- Do not list a contribution the paper does not defend with evidence.

### Related work
- Group prior work by method or by question, not paper by paper.
- State what each line of prior work got right and what it left short, in present tense.
- Say plainly what is new here relative to the closest prior work, and be careful with
  "first", since a claim of priority is checked harder than any other sentence in the
  paper.

### Method
- Say what changes and what is held fixed, and why holding it fixed keeps the comparison
  fair.
- Define every symbol at first use. Prefer the source paper's own notation to a new one.
- State the decision rule and its direction (which score means what) once, plainly, and
  do not restate it with different wording later.

### Experiments
- State the question each experiment answers before the numbers, not after.
- Report strong baselines, not the weakest one that makes the method look good.
- Separate the variables that could each explain a result, and test them 1 at a time.
- State the protocol for choosing a hyperparameter, a split or a threshold before the
  result that protocol produced, so a reader can check the choice was not made after
  seeing the answer.

### Results tables and figures
- Introduce a table or figure with a sentence that states what it will show, before the
  numbers.
- Follow it with a reading, a sentence that says what pattern the reader should take away
  and why, not a restatement of the cells.
- A caption should be readable without the surrounding prose. State the axes, the units
  and the 1 thing the reader should notice.
- Figure 1 carries more weight than any other figure. Make it stand on its own.

### Mechanism or analysis sections
- State the account first, in 1 or 2 sentences, then the evidence for each part of it.
- Distinguish what the account predicts from what it merely describes after the fact.
- Report the controls (a random baseline, a benign model) beside the effect, not in a
  separate section a reader has to hunt for.

### Limitations
- State them as boundaries of the evidence, not as apologies.
- Every limitation a reader would find on their own by reading the experiments should be
  named here first.
- A limitation that changes the recommendation (not just a caveat) belongs in the abstract
  too.

### Conclusion
- Restate the finding in 1 or 2 sentences, without repeating the abstract word for word.
- Name the open questions as open questions, not as future work filler.
- No new numbers and no new claims that were not already supported in the body.

## 3. Sentence and paragraph rules

- 1 topic per paragraph. A paragraph that reports 3 unrelated numbers is 3 paragraphs.
- Topic sentence first, evidence in the middle, the takeaway last. A reader skimming only
  first sentences should get the paper's argument.
- State a number with its comparison and its uncertainty together, not as a bare figure.
  "0.935 against 0.832, a gain of 0.103" carries more than "0.935" on its own.
- Introduce a table or figure, then read it. Never paste a table with no sentence either
  side of it.
- Put the verb early in the sentence. A long noun phrase before the first verb is harder
  to parse than the same content split into 2 sentences.
- 1 idea per sentence. If a sentence needs "and" to hold 2 findings together, it is
  usually 2 sentences.
- Make the thing the paper cares about the grammatical subject. "The gain grows as the
  poison rate falls" reads better than "As the poison rate falls, an increase in the gain
  is observed."
- Active voice, with the actor named. "We measured" or "the placement scores" rather than
  a passive construction that hides who or what did the measuring.
- Cut hedge words that do not carry information. "May", "can" and "seems to" should
  survive only where the claim is genuinely uncertain, not as a reflex softener.
- Cut filler connectives: "note that", "it is worth noting", "essentially", "simply", "in
  order to". They add length and no content.
- Do not use the "it is not X, it is Y" construction, and do not pair 2 clauses that both
  open with "it is" to make a contrast. State the claim directly instead.
- Do not start every sentence in a paragraph with "we". Vary the subject.
- Define an acronym or a piece of jargon at its first use, in the section where it first
  appears, not only in an appendix or a table caption.
- Use the same name for the same thing everywhere. A placement, a variable or a method
  named 1 way in Section 3 should not become a different name or a vaguer description in
  Section 8.
- Avoiding AI-sounding prose is mostly a matter of cutting: cut the contrastive
  construction above, cut hedge-then-restate pairs, cut a summary sentence that only
  repeats the paragraph's first sentence, and cut a transition word that does no work
  ("moreover", "furthermore", "it is important to note").

## 4. Common reviewer complaints and how to pre-empt them

- "The claim is broader than the evidence." State the claim at the same scope as the
  experiments that support it. If the panel is 4 datasets, say 4 datasets, not "across
  vision transformers" without qualification.
- "I cannot tell what is new here versus prior work." Name the closest prior work in the
  same paragraph as the claim of novelty, and say what specifically differs.
- "The comparison is not apples to apples." State the protocol (the split, the
  hyperparameter search, the compute budget) for every method compared, not only for the
  proposed one.
- "The paper does not say what its numbers would look like if it failed." State the floor
  (chance, a random baseline, a benign control) beside every headline number.
- "I do not know why this works." A ranking or a table of numbers is not an explanation.
  State the mechanism, or state plainly that none is offered.
- "The limitations section is a list of things that could be improved, not a list of
  things the current evidence cannot support." Write limitations as scope statements.
- "Terminology drifts across sections." A rewrite of the early sections that renames a
  method or a unit of analysis must propagate to every later section, table caption and
  appendix that still uses the old name.
- "An acronym or symbol appears in a table with no definition in the prose." Define it
  before the table, not only in the table's own caption.
- "A forward reference points at content that is not there." Every "as Section N shows"
  needs Section N to actually show it. Check this by rereading the target section, not by
  trusting the sentence that wrote the pointer.
- "The result is a single run." State how many seeds, and how much a result moves across
  them, beside the mean.

## 5. Anti-patterns

- Overclaiming: a claim stated more strongly than the panel behind it supports, for
  example a mechanism claimed to be general when it was measured on 1 architecture or 1
  attack family.
- "Novel" and "significant" asserted rather than shown. State what is different and let
  the reader judge whether it is novel. State the effect size and its interval rather than
  the word "significant".
- Mathiness: notation that dresses up a simple idea, or a symbol introduced and never
  used again. Every symbol should earn its place by appearing in a claim the paper checks.
- Suggestive language: a result described as "suggesting" or "hinting at" a mechanism
  without the experiment that would test it, when a direct experiment was available.
- Undefined terms: an acronym, a named placement or a technical term used before its
  first definition, or defined once and then referred to inconsistently later.
- Results without a reading: a table or figure with no sentence stating what pattern the
  reader should take from it.
- A claim of priority ("the first to") without checking the closest related work for a
  prior instance.
- A dangling forward reference: a sentence that promises content in a later section that
  the later section does not contain.

## 6. This paper, specifically

Sections 00 through 07 have been rewritten since the checklist above was drafted, moving
to plain-named placements ("attention-input token masking", "post-residual dropout") and
"models" for the panel's unit. Sections 08 through 12 have not caught up, which is the
largest single issue below. Each item names the file and quotes the sentence it concerns.

1. **Terminology has not propagated past Section 7.** `paper/sections/05-results.tex` and
   `paper/sections/06-mechanism.tex` now name the 2 placements directly ("Attention-input
   token masking against post-residual dropout") and call the panel's unit a "model".
   `paper/sections/08-robustness.tex`, `09-limitations.tex`, `10-related.tex` and
   `12-appendix.tex` still say "the recommended placement", "the residual placement" and
   "cell", for example 09-limitations.tex: "the recommended placement moves by 0.012
   AUROC between seeds". Propagate the rename from 00-07 into 08-12 so a reader does not
   meet 2 vocabularies for the same panel.

2. **A forward reference points at content that is not there.**
   `paper/sections/10-related.tex`: "The SAM-enhanced detection paper trains the victim
   with sharpness-aware minimisation to amplify backdoor neurons; \cref{sec:limitations}
   notes what we found on our SAM checkpoints." `paper/sections/09-limitations.tex`
   contains no mention of SAM. Either add the promised paragraph to limitations or drop
   the pointer.

3. **CKA is never defined.** `paper/sections/12-appendix.tex`: "the final-layer linear CKA
   between clean and triggered activations (1.0 means the trigger leaves the
   representation unchanged)". The acronym (centered kernel alignment) is used here and in
   `paper/tables/tac_layers.tex` and nowhere expanded. Spell it out at first use, here or
   earlier in `02-background.tex` or `06-mechanism.tex`.

4. **TAC is never defined either.** `paper/tables/tac_layers.tex`: the caption and header
   both use "TAC" ("the last layer's clean-versus-triggered CKA and mean TAC") with no
   prose definition anywhere in the sections read. Name what it stands for on first use.

5. **ASR appears only as a bare column header.** `paper/sections/08-robustness.tex`,
   `tab:adaptive`: the column is headed "ASR" with no prior parenthetical ("attack success
   rate (ASR)") the way AUROC and TPR are introduced in
   `paper/sections/05-results.tex`: "the area under the ROC curve (AUROC) measures... The
   true-positive rate (TPR) at a false-positive budget is...". Give ASR the same treatment.

6. **Semicolons remain in the sections not yet rewritten**, against this project's own
   rule and against the cleanup already visible in 00-07. `paper/sections/12-appendix.tex`:
   "whose attack success rate is at least 0.85; the cells below that bar are marked with a
   dagger". `paper/sections/10-related.tex`: "remove the backdoor by orthogonalising it
   out of the weights; \cref{sec:mechanism:picek} states where we agree". Split each into
   2 sentences.

7. **A banned contrastive construction survives in Section 8.**
   `paper/sections/08-robustness.tex`: "It is not only weaker on average, it is also far
   less stable." State the claim directly, for example "The residual placement is weaker
   on average and far less stable across seeds."

8. **An AI-tell filler word survives the rewrite of Section 5.**
   `paper/sections/05-results.tex`: "so more blocks is simply more chances to remove
   them." Cut "simply". The project's own `scripts/prose_audit.py` already flags this
   word in code comments, the same rule reads naturally onto the prose here.

9. **An ambiguous adverb placement in the introduction.**
   `paper/sections/01-introduction.tex`: "We show once, in \cref{sec:results:headline},
   where that difference comes from." "Show once" reads as a tense error before it reads
   as "this is the only place we make the comparison". Reword, for example
   "\Cref{sec:results:headline} is the only place we make this comparison, and it shows
   where the difference comes from."

10. **An unchecked priority claim.** `paper/sections/10-related.tex`: "Our work is the
    first to study what happens to such a detector when the perturbation site is a design
    choice, as it is on a transformer." A claim of priority draws the hardest scrutiny of
    any sentence in a related-work section. Karayalcın et al., discussed 2 paragraphs
    later in the same section, also studies transformer backdoor mechanism, though not
    this exact question. Either narrow the claim to what is checked ("the first to treat
    the perturbation site as a design choice for this detector") or cite the nearest
    attempt and say what it did not do.

11. **A colon-heavy sentence.** `paper/sections/10-related.tex`: "PSBD
    \cite{li2025psbd} belongs to the consistency family with STRIP, SCALE-UP and TeCo:
    all perturb the model or its input and read how stable the prediction is, and differ
    in what they perturb." Split at the colon into 2 sentences.

12. **A repeated-preposition stumble in the setup section.**
    `paper/sections/04-setup.tex`: "Choosing the best of 18 placements on the same models
    it is then reported on would be an optimistic reading". The doubled "on... on" forces
    a reread. Reword, for example "Choosing the best of 18 placements on the models it is
    later reported on would give an optimistic reading."

13. **The abstract drops the deployment number the rest of the paper leans on.** The
    current `paper/sections/00-abstract.tex` states AUROC (0.935, 0.832) but not the true
    positive rate at 1% false positives (0.629 against 0.477) that
    `paper/sections/01-introduction.tex`'s contribution list and
    `paper/sections/05-results.tex`'s \cref{tab:headline} both lead with. A reader who
    stops at the abstract misses the number a deployer would ask for first.

14. **"cells" versus "checkpoints" versus "models" inside a single appendix.**
    `paper/sections/12-appendix.tex` alone uses "the checkpoints of \cref{tab:panel}",
    "Each row is 1 dataset and 1 attack" and "cell" in the same section ("a cell below the
    bar prints only its attack success rate"). Pick 1 noun for the panel's unit and use it
    throughout the appendix, consistent with whichever term 08-12 settle on after item 1.

15. **An underspecified comparison in the Swin section.**
    `paper/sections/07-swin.tex`: "WaNet is detected worse on Swin than on ViT, at 0.26 to
    0.29 against 0.51 to 0.55 with dropout at the matched rate on CIFAR-100." The sentence
    reads as a Swin-versus-ViT comparison for the recommended placement (token masking),
    but "with dropout" then names a different placement for at least 1 side of the
    numbers, and it is not stated which side. Name the placement for each of the 2
    numbers explicitly.
