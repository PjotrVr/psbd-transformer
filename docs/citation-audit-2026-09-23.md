# Citation audit, 2026-09-23

## Scope and method

The audit covers `paper/references.bib` against every `\cite`, `\citep` and `\citet` in
`paper/sections/*.tex`, `paper/tables/*.tex` and `paper/main.tex`. Usage counts come from a
key extraction over those files, so a key cited twice in 1 sentence counts twice. Every
bibliographic claim was checked against the arXiv API, the CVF open access proceedings, the
AAAI proceedings or the publisher record, and the local LaTeX source at
`literature/backdoor-directions-karayalcin/source/` for the 1 entry that has no public
metadata worth trusting. The bib holds 56 entries and the prose uses 48 of them.

## Inventory

Usage is the total count across sections and tables. Verdict `ok` means the entry matched the
record on authors, title, venue, year and preprint identifier. Verdict `fixed` means a
discrepancy was found and the entry was corrected in place. Verdict `unused` means the record
is right and nothing cites it.

| Key | Type | Usage | Verified venue and year | Verdict |
| --- | --- | --- | --- | --- |
| `li2025psbd` | inproceedings | 3 | CVPR 2025, pp. 10255 to 10264, arXiv 2406.05826 | fixed, pages added |
| `dosovitskiy2021an` | inproceedings | 2 | ICLR 2021 | ok |
| `liu2021swin` | inproceedings | 1 | ICCV 2021 | ok |
| `gal2016dropout` | inproceedings | 1 | ICML 2016, pp. 1050 to 1059 | fixed, title capital |
| `gu2017badnets` | article | 3 | IEEE Access 7:47230 to 47244, 2019 | ok |
| `chen2017targeted` | article | 3 | preprint, arXiv 1712.05526, never published | ok |
| `turner2019label` | article | 3 | preprint, arXiv 1912.02771, never published | ok |
| `nguyen2021wanet` | inproceedings | 3 | ICLR 2021 | ok |
| `tang2021demon` | inproceedings | 4 | USENIX Security 2021, pp. 1541 to 1558 | ok |
| `gao2019strip` | inproceedings | 2 | ACSAC 2019, pp. 113 to 125 | ok |
| `guo2023scale` | inproceedings | 2 | ICLR 2023, arXiv 2302.03251 | ok |
| `hou2024ibdpsc` | inproceedings | 2 | ICML 2024, arXiv 2405.09786 | ok |
| `huang2023distilling` | inproceedings | 1 | ICLR 2023, arXiv 2301.10908 | ok |
| `liu2023teco` | inproceedings | 2 | CVPR 2023 | ok |
| `ma2023beatrix` | inproceedings | 1 | NDSS 2023, arXiv 2209.11715 | ok |
| `mo2024ted` | inproceedings | 1 | IEEE S&P 2024, arXiv 2312.02673 | ok |
| `chou2020sentinet` | inproceedings | 1 | IEEE S&P Workshops, DLS 2020, arXiv 1812.00292 | ok |
| `karayalcin2026backdoor` | article | 4 | preprint, arXiv 2603.10806, 2026-03-11 | fixed, diacritics |
| `zhang2024patching` | inproceedings | 2 | ICLR 2024, arXiv 2309.16042 | ok |
| `baldock2021difficulty` | inproceedings | 1 | NeurIPS 2021, arXiv 2106.09647 | ok |
| `cohen2019smoothing` | inproceedings | 0 | ICML 2019, arXiv 1902.02918 | unused |
| `rajabi2023mdtd` | inproceedings | 0 | ACM CCS 2023, arXiv 2308.15673 | unused |
| `darcet2024registers` | inproceedings | 1 | ICLR 2024, arXiv 2309.16588 | ok |
| `sun2024massive` | inproceedings | 1 | COLM 2024, arXiv 2402.17762 | ok |
| `xiong2020layernorm` | inproceedings | 0 | ICML 2020, arXiv 2002.04745 | unused |
| `mitchell2023detectgpt` | inproceedings | 0 | ICML 2023, arXiv 2301.11305 | unused |
| `wang2025a2x` | article | 0 | AAAI 2026, arXiv 2511.13356 | fixed and unused |
| `kornblith2019cka` | inproceedings | 1 | ICML 2019 | ok |
| `wang2022bpp` | inproceedings | 2 | CVPR 2022, pp. 15074 to 15084 | ok |
| `zeng2021lf` | inproceedings | 3 | ICCV 2021, pp. 16473 to 16481 | ok |
| `barni2019sig` | inproceedings | 3 | ICIP 2019, pp. 101 to 105 | ok |
| `qi2023adaptive` | inproceedings | 3 | ICLR 2023 | ok |
| `tran2018spectral` | inproceedings | 1 | NeurIPS 2018, pp. 8011 to 8021 | ok |
| `chen2018activation` | inproceedings | 1 | SafeAI at AAAI 2019, arXiv 1811.03728 | ok |
| `hayase2021spectre` | inproceedings | 1 | ICML 2021, pp. 4129 to 4139 | ok |
| `liu2018finepruning` | inproceedings | 1 | RAID 2018, arXiv 1805.12185 | ok |
| `li2021nad` | inproceedings | 1 | ICLR 2021, arXiv 2101.05930 | ok |
| `wu2021anp` | inproceedings | 1 | NeurIPS 2021, pp. 16913 to 16925 | ok |
| `zeng2022ibau` | inproceedings | 1 | ICLR 2022, arXiv 2110.03735 | ok |
| `zheng2022clp` | inproceedings | 2 | ECCV 2022, arXiv 2208.03111 | fixed, title capital |
| `wu2022backdoorbench` | inproceedings | 0 | NeurIPS 2022 Datasets and Benchmarks | unused |
| `krizhevsky2009cifar` | techreport | 1 | University of Toronto, 2009 | ok |
| `stallkamp2012gtsrb` | article | 1 | Neural Networks 32:323 to 332, 2012 | ok |
| `le2015tiny` | techreport | 1 | Stanford CS231N, 2015 | ok |
| `netzer2011svhn` | inproceedings | 1 | NIPS Workshop 2011 | ok |
| `helber2019eurosat` | article | 1 | IEEE JSTARS 12(7):2217 to 2226, 2019 | ok |
| `kingma2015adam` | inproceedings | 1 | ICLR 2015, arXiv 1412.6980 | ok |
| `madry2018pgd` | inproceedings | 0 | ICLR 2018, arXiv 1706.06083 | unused |
| `selvaraju2017gradcam` | inproceedings | 1 | ICCV 2017, pp. 618 to 626 | ok |
| `efron1993bootstrap` | book | 1 | Chapman and Hall, 1993 | ok |
| `paszke2019pytorch` | inproceedings | 1 | NeurIPS 2019, arXiv 1912.01703 | ok |
| `foret2021sam` | inproceedings | 0 | ICLR 2021, arXiv 2010.01412 | unused |
| `li2022backdoorsurvey` | article | 1 | IEEE TNNLS 35(1):5 to 22, 2024 | fixed, year and pages |
| `nostalgebraist2020logitlens` | misc | 2 | LessWrong, 2020 | ok |
| `belrose2023tunedlens` | article | 2 | preprint, arXiv 2303.08112 v6 | fixed, author order |
| `zhang2024samdetection` | article | 1 | preprint, arXiv 2411.11525, never published | ok |

## Dead citations

No `\cite` key in the prose lacks a bib entry, so the build has no undefined reference and
nothing renders as a bold question mark. The check ran over all 14 section files, all 79 table
files and `main.tex`, and the current `main.blg` carries no warnings. That direction is clean.

8 entries are in the bib and nothing cites them, which confirms the count the earlier audit
reported. The recommendation differs per entry, because 3 of them mark a claim the paper makes
without support and 5 are residue from withdrawn text.

| Key | Recommendation | Where it belongs, or why it should go |
| --- | --- | --- |
| `wu2022backdoorbench` | cite it | `paper/sections/attacks.tex` names BackdoorBench 4 times as the source of the trigger conventions, the WaNet cross ratio and the BPP negative ratio, with no citation attached. Attach at the first mention, `attacks.tex:4`. |
| `foret2021sam` | cite it | `paper/tables/sam.tex` and the SAM rows of the robustness section rest on sharpness aware minimization, and the optimizer is described in `setup.tex` without its source. Attach where rho is first stated. |
| `madry2018pgd` | cite it | `attacks.tex:14` says the Label-Consistent bases come from an untargeted L-infinity PGD attack at 100 steps and step size 2.5 times epsilon over 100. That recipe is Madry et al. and the citation is missing. |
| `cohen2019smoothing` | delete | Randomized smoothing appears nowhere in the current text and the method makes no certification claim. It was a framing device for a paragraph that no longer exists. |
| `mitchell2023detectgpt` | delete | The probability curvature analogy is not made in the current draft. Keeping it invites a reviewer to ask which claim it supports. |
| `rajabi2023mdtd` | delete or cite | MDTD is an input-level trojan detector in the same threat model, so it could join the list at `related-work.tex:8`. Leaving it uncited is the worse of the 2 options. |
| `xiong2020layernorm` | cite it | The operator effect is attributed to which side of a LayerNorm the perturbation lands on, in `CLAUDE.md` and in the results section. Pre-norm against post-norm placement is exactly Xiong et al. and the claim reads better with it. |
| `wang2025a2x` | delete | The all-to-all results are a small ablation and do not engage with optimized target class mapping. The entry was corrected to its published form in case the user keeps it. |

Deleting an entry is the user's call, so nothing was removed from the bib. The 4 entries
marked `cite it` are the more serious finding, because each marks a place where the prose
asserts a method it borrowed and names no source.

## Overciting

The prose is disciplined and the density is low. 1 citation per named method is the dominant
pattern and the 3 stacked pairs that exist are defensible as literature pointers. 3 cuts are
worth making.

The attack list is cited 3 times over. `paper/sections/related-work.tex:5` introduces all 9
attacks with 1 citation each, `paper/sections/setup.tex:8` repeats all 9 keys in a single
sentence, and `paper/sections/attacks.tex` cites each a third time where it describes the
trigger. The cut is `setup.tex:8`, which should name the attacks and point to the appendix
with no keys at all, because the sentence already ends with `\Cref{app:attacks} describes
every attack`. The 6 dataset citations in the same sentence are the first use of those keys
and must stay.

The 4 test-time detectors are cited twice in adjacent subsections.
`paper/sections/background.tex:5` reads `This is the setting of STRIP \cite{gao2019strip},
SCALE-UP \cite{guo2023scale}, IBD-PSC \cite{hou2024ibdpsc}, TeCo \cite{liu2023teco} and PSBD`,
and `paper/sections/related-work.tex:8` cites the same 4 keys 1 page later where each method is
actually described. Cut the 4 keys from `background.tex:5` and keep the names, because the
threat-model sentence there is scoping the problem rather than crediting the work. The PSBD
citation in the same sentence is also redundant against `background.tex:8`, which cites it
again 3 lines down.

The logit lens carries 2 citations in a place where 1 fits. `paper/sections/robustness.tex:62`
reads `a logit lens \cite{nostalgebraist2020logitlens,belrose2023tunedlens} that costs 1 pass`,
and the described probe reads the class token through the frozen final LayerNorm and head,
which is the plain logit lens and not the tuned lens. Cut `belrose2023tunedlens` there and keep
the pair at `related-work.tex:16`, where naming the refinement alongside the original is the
right move. `karayalcin2026backdoor` at 4 uses and `tang2021demon` at 4 uses are both fine,
because each use is a distinct claim and the Tang paper contributes both an attack and a
detector that the paper treats separately.

## Missing work

This is the section a reviewer punishes hardest, and the gap is concentrated in 3 places. The
priority claim at `related-work.tex:10`, `To our knowledge no earlier work studies what happens
to such a detector when the perturbation site is a design choice, as it is on a ViT`, is stated
against a related-work section that cites no ViT-specific backdoor defense at all. The method
section carries 0 citations, so the perturbation primitive the whole paper rests on has no
source. The mechanistic tools are cited from blog posts and from secondary sources rather than
from the peer-reviewed work that established them.

### The 10 that matter most

| Rank | Work | Identifier | Claim it attaches to |
| --- | --- | --- | --- |
| 1 | Doan, Lao et al., Defending Backdoor Attacks on Vision Transformer via Patch Processing, AAAI 2023 | arXiv 2206.12381, DOI 10.1609/aaai.v37i1.25125 | The priority claim at `related-work.tex:10`. This is a ViT-specific test-time backdoor defense built on patch transformations before the positional encoding, evaluated on CIFAR-10, GTSRB and Tiny ImageNet, which is 3 of the paper's 6 datasets. It is the nearest published antecedent of token masking and its absence is the single most dangerous omission. |
| 2 | Yu, Yin et al., Fast and Lightweight Backdoor Detection via Head Random Probing, preprint 2026 | arXiv 2605.18908 | The same priority claim, and the `attention_heads` site in `method.tex:7`. Concurrent work that probes a transformer by randomly perturbing attention heads, which is the closest competing instance of the paper's own idea that the perturbation site inside the block is a design choice. |
| 3 | Subramanya, Koohpayegani et al., A Closer Look at Robustness of Vision Transformers to Backdoor Attacks, WACV 2024, pp. 3874 to 3883 | arXiv 2206.08477 in its earlier form | The token-masking operator in `method.tex:9` and the depth result at `results.tex:46`. Their test-time image blocking defense zeroes patches on a ViT and cuts attack success by a large margin, so it is the prior art for zeroing whole tokens and for the claim that ViTs route a patch trigger differently from a ConvNet. |
| 4 | Srivastava, Hinton et al., Dropout: a simple way to prevent neural networks from overfitting, JMLR 15:1929 to 1958, 2014 | JMLR v15 paper 313 | `method.tex:7` and `method.tex:9`, and the inverted-scaling rationale in `models/positions.py`. Dropout and its inference-time rescaling are the primitive the method injects, and the paper currently cites only the Bayesian reading of it. |
| 5 | Abad, Kr\v{c}ek et al., SoK: The Last Line of Defense: On Backdoor Defense Evaluation, preprint 2025 | arXiv 2511.13143 | The protocol paragraph at `setup.tex:17` and the limitations section. A systematization of how backdoor defense evaluation goes wrong, which is the standard the panel and the declared basis split are trying to meet. It is also from Picek's group, so a reviewer from that circle will look for it. |
| 6 | Xie, Qi et al., BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection, ICLR 2024 | arXiv 2308.12439 | The detector list at `related-work.tex:8`. A strong input-level detector in exactly the paper's threat model, by the authors of Adaptive-Blend, which the paper already cites as `qi2023adaptive`. Omitting it while citing the same group's attack looks selective. |
| 7 | Elhage, Nanda et al., A Mathematical Framework for Transformer Circuits, Transformer Circuits Thread, 2021 | transformer-circuits.pub/2021/framework | The residual-stream account at `background.tex:24` and the routing story in `mechanism.tex`. The claim that the residual stream is the sum of everything the network has written so far is theirs and is currently unsourced. |
| 8 | Geva, Caciularu et al., Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space, EMNLP 2022 | arXiv 2203.14680 | The logit lens at `robustness.tex:62`. A peer-reviewed grounding for reading intermediate activations through the output head, which a security venue will prefer to a LessWrong post as the load-bearing citation. |
| 9 | Meng, Bau et al., Locating and Editing Factual Associations in GPT, NeurIPS 2022, and Vig, Gehrmann et al., Causal Mediation Analysis for Interpreting Neural NLP, NeurIPS 2020 | arXiv 2202.05262 and arXiv 2004.12265 | Activation patching at `mechanism.tex:42`. The paper cites only Zhang and Nanda, which is a best-practices study of the technique rather than its origin, so the primary sources are missing. |
| 10 | Xiao, Tian et al., Efficient Streaming Language Models with Attention Sinks, ICLR 2024 | arXiv 2309.17453 | `related-work.tex:16`, which says `the register and attention-sink literature \cite{darcet2024registers,sun2024massive}`. Neither cited work is the attention-sink paper, so the phrase currently names a literature it does not cite. |

### Worth adding if space allows

| Work | Identifier | Claim it attaches to |
| --- | --- | --- |
| Abbasi, Zhang et al., Backdoor Attacks and Defenses in Computer Vision Domain: A Survey, 2025 | arXiv 2509.07504 | `introduction.tex:6`, which opens on a 2022 survey. A 2026 submission should point at a 2025 or later survey beside it. |
| Zheng, Lou and Jiang, TrojViT: Trojan Insertion in Vision Transformers, CVPR 2023 | arXiv 2208.13049 | `related-work.tex:5`. The ViT-specific patch-wise trigger, which shows that transplanting a ConvNet attack to a ViT gives a weak backdoor, the mirror of this paper's argument about transplanting a defense. |
| Peng, Fu et al., Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models, Neural Networks 193:108025, 2026 | arXiv 2509.05318 | `related-work.tex:10`, the consistency family. The nearest transformer analogue of PSBD's statistic, in the language domain, published after PSBD. |
| Huang, Li et al., BackdoorIDS: Zero-shot Backdoor Detection for Pretrained Vision Encoder, preprint 2026 | arXiv 2603.11664 | The detector list at `related-work.tex:8`. Recent ViT-encoder detection that a reviewer tracking 2026 arXiv will know. |
| Ahlers, Passon et al., Kill it with FIRE: On Leveraging Latent Space Directions for Runtime Backdoor Mitigation, preprint 2026 | arXiv 2602.10780 | The erasure experiment in `appendix.tex:72`. Concurrent work on latent-direction mitigation at runtime, which bears on the reproduction of the Karayalcin removal. |
| Liu, Qiao et al., PASTA: A Patch-Agnostic Twofold-Stealthy Backdoor Attack on Vision Transformers, preprint 2026 | arXiv 2604.20047 | The threat model in `limitations.tex`. A ViT-specific stealthy attack that the panel does not include, worth naming as out of scope rather than leaving unmentioned. |
| Ovadia, Fertig et al., Can You Trust Your Model's Uncertainty, NeurIPS 2019 | arXiv 1906.02530 | The margin account in `method.tex`. Evidence on how predictive uncertainty behaves under distribution shift, which is what the fractional PSU statistic is measuring. |

## Rendering problems

The style is `plainnat` with `natbib` in `numbers,sort&compress` mode, and `plain` family
styles pass titles through BibTeX's title-lowercasing, so any capital that is not inside braces
is lost. 2 entries lost a capital that carries meaning and both were fixed. `gal2016dropout`
had `Bayesian` unbraced and would have rendered as `bayesian`, and `zheng2022clp` had
`Lipschitzness` unbraced and would have rendered as `lipschitzness`. Every acronym in the
remaining titles is already braced, which was checked entry by entry.

No entry is missing a required field for its type. The 2 `@techreport` entries carry
`institution`, the 1 `@book` entry carries `publisher`, every `@inproceedings` entry carries
`booktitle` and every `@article` entry carries `journal`. The `@misc` entry for the logit lens
carries `howpublished` and `year`, which is what `plainnat` needs.

1 `@article` entry was a published paper filed as a preprint. `wang2025a2x` appeared at AAAI
2026 and was changed to `@inproceedings` with the AAAI booktitle and year 2026, keeping the
arXiv identifier in the note. The 4 remaining `journal={arXiv preprint ...}` entries are
`chen2017targeted`, `turner2019label`, `karayalcin2026backdoor` and `zhang2024samdetection`,
and all 4 are genuinely unpublished as of today, so the form is correct rather than stale. The
`belrose2023tunedlens` entry is also a genuine preprint at v6, though its author order had
drifted from the bib and was corrected to the current arXiv listing.

The `note={arXiv:...}` field on published entries prints under `plainnat` and is used
consistently across the bib, so it reads as a deliberate convention rather than an artifact.
1 cosmetic mismatch is worth knowing about and needs no action. 2 keys carry a year that is not
the publication year, `gu2017badnets` for a 2019 IEEE Access paper and `chen2018activation` for
a 2019 workshop paper, and both keys are named after the preprint year. A key is a label and
changing it would touch 4 prose sites for no gain.
