"""Verify the closed form the 3 predictions rest on, by autograd.

Claim (docs/theory-perturbation-consistency.md, objection 3):

    original form
        tr grad^2_z p_c = 2 p_c (||p||_2^2 - p_c),   p = softmax(z)

    descriptive form
        hessian_trace = 2 * prob_predicted * (sum_of_squared_probs - prob_predicted)

If this is wrong, prediction 1 (a head probe cannot beat confidence) and
prediction 2 (inversions concentrate at low confidence) both lose their basis, so
it is checked before either is measured rather than assumed.

The 2-class corollary is checked here too: with p_c = p >= 0.5 the trace is
2p(2p-1)(p-1), zero at both p = 0.5 and p = 1, with a single interior extremum.

Example
    PYTHONPATH=. .venv/bin/python experiments/theory_predictions/hessian_trace_identity.py
"""

import torch


def closed_form_trace(probs: torch.Tensor, predicted: int) -> float:
    """2 * p_c * (||p||^2 - p_c), the claimed trace of the Hessian at the logits."""
    p_c = probs[predicted]
    trace = 2.0 * p_c * (probs.pow(2).sum() - p_c)
    return float(trace)


def autograd_trace(logits: torch.Tensor, predicted: int) -> float:
    """Sum of the diagonal second derivatives of softmax(z)[predicted] wrt z."""
    z = logits.detach().clone().requires_grad_(True)
    probability = torch.softmax(z, dim=0)[predicted]

    first = torch.autograd.grad(probability, z, create_graph=True)[0]
    diagonal = torch.stack(
        [
            torch.autograd.grad(first[i], z, retain_graph=True)[0][i]
            for i in range(z.numel())
        ]
    )
    return float(diagonal.sum())


def check_random_softmaxes(trials: int, num_classes: int, temperature: float) -> float:
    """Largest absolute disagreement between the closed form and autograd."""
    worst = 0.0
    for trial in range(trials):
        torch.manual_seed(trial)
        logits = torch.randn(num_classes, dtype=torch.float64) * temperature
        probs = torch.softmax(logits, dim=0)
        predicted = int(probs.argmax())

        gap = abs(
            closed_form_trace(probs, predicted) - autograd_trace(logits, predicted)
        )
        worst = max(worst, gap)
    return worst


def two_class_profile(grid: torch.Tensor) -> torch.Tensor:
    """tr(p) = 2p(2p-1)(p-1) over p in [0.5, 1], the non-monotone corollary."""
    return 2.0 * grid * (2.0 * grid - 1.0) * (grid - 1.0)


def main() -> None:
    print("closed form 2*p_c*(||p||^2 - p_c) against autograd on random softmaxes\n")
    for num_classes, temperature in (
        (2, 1.0),
        (10, 1.0),
        (100, 1.0),
        (100, 3.0),
        (100, 0.3),
    ):
        worst = check_random_softmaxes(40, num_classes, temperature)
        print(
            f"  classes {num_classes:>4}  logit scale {temperature:>4}  max |gap| {worst:.3e}"
        )

    grid = torch.linspace(0.5, 1.0, 5001, dtype=torch.float64)
    profile = two_class_profile(grid)
    peak = int(profile.abs().argmax())
    print("\n2-class corollary tr(p) = 2p(2p-1)(p-1) on p in [0.5, 1]")
    print(
        f"  tr(0.5)          {float(two_class_profile(torch.tensor([0.5], dtype=torch.float64))[0]):+.6f}"
    )
    print(
        f"  tr(1.0)          {float(two_class_profile(torch.tensor([1.0], dtype=torch.float64))[0]):+.6f}"
    )
    print(
        f"  |tr| peaks at p  {float(grid[peak]):.4f}  value {float(profile[peak]):+.6f}"
    )
    print(f"  sign everywhere  {'<= 0' if bool((profile <= 1e-12).all()) else 'MIXED'}")
    print(
        f"  single interior extremum: {int((profile.diff().sign().diff() != 0).sum())} turning point(s)"
    )


if __name__ == "__main__":
    main()
