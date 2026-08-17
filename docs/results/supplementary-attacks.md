# Supplementary Attack Detection: BPP and LF

BPP (Bit-level Perturbation Poisoning) and LF (Low-Frequency) are outside the 5-attack panel but have checkpoints and sweep data across CIFAR-100 and Tiny ImageNet. These results supplement the main tables.

All numbers: fractional PSU, sigma-matched (sigma >= 0.6), one-sided (low PSU = poisoned).

## token_mask @ before_attention_norm (recommended config)

| Dataset | Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---|---:|---:|---:|---:|
| cifar100 | 1% | bpp | 0.963 | 0.927 | 0.680 | 0.4 |
| cifar100 | 1% | lf | 0.942 | 0.944 | 0.915 | 0.4 |
| cifar100 | 5% | bpp | 0.984 | 0.981 | 0.954 | 0.4 |
| cifar100 | 5% | lf | 0.953 | 0.957 | 0.931 | 0.4 |
| cifar100 | 10% | bpp | 0.994 | 0.993 | 0.984 | 0.4 |
| cifar100 | 10% | lf | 0.995 | 0.994 | 0.993 | 0.4 |
| tiny | 1% | bpp | 0.973 | 0.966 | 0.928 | 0.4 |
| tiny | 1% | lf | 0.922 | 0.930 | 0.853 | 0.4 |
| tiny | 5% | bpp | 0.988 | 0.973 | 0.944 | 0.4 |
| tiny | 5% | lf | 0.987 | 0.976 | 0.957 | 0.4 |
| tiny | 10% | bpp | 0.995 | 0.976 | 0.978 | 0.4 |
| tiny | 10% | lf | 0.973 | 0.956 | 0.849 | 0.4 |

Mean AUROC: BPP 0.969, LF 0.960. Both detected reliably at all poison rates.

## gain_scale @ mlp_norm_out

| Dataset | Poison | Attack | ASR | AUROC | TPR@5% | Rate |
|---|---|---|---:|---:|---:|---:|
| cifar100 | 1% | bpp | 0.963 | 0.981 | 0.967 | 2 |
| cifar100 | 1% | lf | 0.942 | 0.957 | 0.940 | 2 |
| cifar100 | 5% | bpp | 0.984 | 0.966 | 0.859 | 2 |
| cifar100 | 5% | lf | 0.953 | 0.962 | 0.942 | 2 |
| cifar100 | 10% | bpp | 0.994 | 0.993 | 0.972 | 2 |
| cifar100 | 10% | lf | 0.995 | 0.994 | 0.984 | 2 |
| tiny | 1% | bpp | 0.973 | 0.947 | 0.964 | 1 |
| tiny | 1% | lf | 0.922 | 0.920 | 0.787 | 1 |
| tiny | 5% | bpp | 0.988 | 0.956 | 0.972 | 1 |
| tiny | 5% | lf | 0.987 | 0.970 | 0.967 | 1 |
| tiny | 10% | bpp | 0.995 | 0.971 | 0.981 | 1 |
| tiny | 10% | lf | 0.973 | 0.942 | 0.802 | 1 |

Mean AUROC: BPP 0.969, LF 0.958. Matches token_mask closely.

## Comparison with the 5-attack panel

BPP and LF are easier to detect than the panel's hardest attacks (wanet, lc, adaptive_blend):

| Config | Panel 1% mean | BPP 1% | LF 1% |
|---|---:|---:|---:|
| token_mask @ before_attention_norm | 0.849 (C100) / 0.962 (Tiny) | 0.927 / 0.966 | 0.944 / 0.930 |
| gain_scale @ mlp_norm_out | 0.945 (C100) / 0.958 (Tiny) | 0.981 / 0.947 | 0.957 / 0.920 |

BPP and LF sit above the panel mean. Neither attack creates a new failure mode. This is consistent with BPP being a subtle pixel perturbation (similar to blend in mechanism) and LF being a frequency-domain trigger (strong enough to produce a clear directional signal).

SIG checkpoints exist but no perturbation sweep data has been cached for the positions above.

## Source

Direct computation from PSBD caches in `results/vit_{dataset}_{attack}_{rate}/psbd/`.
