# Where clean samples move under dropout, in latent space (H7)

## Question

H7 rests on 1 measurement: a histogram of which class label a shifted clean
prediction lands on, counted from the cached per-pass argmax. That is entirely
prediction space. It says the label changed and where it went, and nothing about
whether the representation moved the way PSBD's mechanism requires.

PSBD's stated mechanism is a claim about representations. Under dropout the model
loses the clean class evidence and falls back on the strongest learned association,
the trigger-to-target path, so a clean sample should drift toward wherever the
target class lives. That is directly testable and had not been tested here.

## The 4 measurements

All at the final block's CLS features, all on the same samples.

| quantity | what it says |
| --- | --- |
| `toward_target` | cosine between the dropout-induced displacement and the direction from the sample's own class centroid to the target class centroid. Positive means clean samples drift toward the target |
| `toward_landed` | the same but toward the centroid of whichever class the sample actually shifted to. This is the control: if the displacement is simply toward wherever the prediction went, `toward_target` carries no extra information |
| `along_backdoor` | projection of the displacement onto the backdoor direction. The sharpest form of the claim: does dropout push a clean sample along the very direction the trigger uses? |
| `displacement_norm` | scale, so the cosines can be read as more than angles |

A benign model probed with the same trigger is measured alongside, because a
pretrained backbone has its own fallback behaviour under heavy perturbation and that
has to be subtracted from any claim about poisoning.

## Running it

    PYTHONPATH=. python experiments/shift_in_latent_space/measure.py \
        --checkpoint-folder vit_cifar10_blend_0_1 vit_cifar10_badnet_a2o_0_1

Writes `results/<folder>/shift_latent.json`, and with `--umap` a projection coloured
by where each sample landed. Needs a GPU.

## Finding

H7 is partially refuted, and the exception is the most important result in the
ledger. The target-class drift is real for `bpp`, `lf` and `badnet_a2o` at higher
poisoning, running at 2 to 5 times the chance rate of 0.10, and it strengthens with
poison rate for both `badnet` variants exactly as the neuron-bias story predicts.

It is absent for `blend`, which reads 0.052 at 10% poisoning, below chance, while
being the best-detected attack in the whole grid at AUROC 0.978 to 0.986. So on a
transformer the method and its published explanation come apart: PSBD works best
precisely where its stated mechanism is measurably not happening. Whatever produces
the confidence gap for `blend` is not clean samples collapsing onto the attacker's
target class.

Hypothesis doc: `docs/hypothesis/H7-clean-shifts-to-target.md`.
