---
name: reference-scaleup-impls
description: Where the SCALE-UP (ICLR 2023) reference implementations live and the three paper-vs-code divergences that break reproduction
metadata:
  type: reference
---

SCALE-UP (Guo et al., ICLR 2023, arXiv 2302.03251) has three separate implementations that do NOT agree:

- Official author repo: `github.com/JunfengGo/SCALE-UP` (`torch_model_wrapper.py` produces the decision matrix, `test.py` reduces it to SPC, `utils.py` has AUROC).
- BackdoorBox: `THUYimingLi/BackdoorBox`, `core/defenses/SCALE_UP.py`.
- backdoor-toolbox: `vtu81/backdoor-toolbox`, `other_defenses_tool_box/scale_up.py`.

Known divergences to check before trusting any port:
1. Scaling set. Paper Sec 4.2 says `S = {3,5,7,9,11}` (5 elements, no n=1). Official repo uses `for h in range(1, 12)` = `{1,...,11}` (11 elements, n=1 included and trivially self-consistent, so SPC floor is 1/11). BackdoorBox and backdoor-toolbox follow the paper's 5-element set.
2. Data-limited normalisation. Paper Eq 3/4 uses per-predicted-class mean and std (mu_yhat, sigma_yhat). Both BackdoorBox and backdoor-toolbox compute a single GLOBAL mean/std.
3. Normalisation placement. Only backdoor-toolbox denormalises before the multiply and renormalises after; BackdoorBox multiplies whatever the dataloader emits. The multiply must happen in [0,1] pixel space with `clip(x * n, 0, 1)`.

Also: `init_spc_norm` in both framework ports compares scaled predictions to GROUND-TRUTH labels, while the test path compares to the model's own original prediction. Inconsistent by construction.

**Why:** This project ports attacks/defences from BackdoorBench and Backdoor-Toolbox and matches their semantics, so a SCALE-UP baseline would silently inherit whichever convention the source used, and the resulting AUROC is not comparable across conventions.

**How to apply:** If SCALE-UP is added as a baseline defence here, pin the scaling set and the normalisation scope explicitly in config rather than inheriting a framework default, and record which convention the reported number uses.
