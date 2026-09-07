# Monte Carlo pass count at low poison rate (H24)

## Question

PSU is an expectation over k stochastic passes, so k sets the estimator's variance.
If the 1% poisoning failure is estimator noise rather than a missing signal, raising
k should help more at 1% than at 10%. If it does not, the k = 20 sweep is not worth
the GPU time.

The free half of the test needs no new jobs. The cache stores per-pass tracked-class
probabilities as a `(k, N)` tensor, so k = 1 and k = 2 are subsets already on disk
and the k = 1 to k = 3 increment can be read off directly.

## Running it

    PYTHONPATH=. python experiments/monte_carlo_passes/k_sweep.py
    PLACEMENT=post_residual PYTHONPATH=. python experiments/monte_carlo_passes/k_sweep.py

`PLACEMENT` defaults to `before_attention_norm`, and the rate is matched at
clean-validation sigma >= 0.6 through `scratch/surface.json`, so the placement being
swept is compared at equal measured disturbance rather than at a shared rate.

## Finding

Both predictions held, and the paid half was then run. The gain from k = 3 to k = 20
is real and concentrated exactly where predicted: `badnet_a2o` at 1% gains 0.046
AUROC (0.839 to 0.885), while the same attack at 5% gains 0.007. Low poison rate is
where the estimator is noise limited, and k = 3 is the floor there.

Hypothesis doc: `docs/hypothesis/H24-monte-carlo-passes.md`.
