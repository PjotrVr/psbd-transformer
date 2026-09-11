"""The confidence null: maximum softmax probability, and nothing else.

There is no paper to cite. This is the null model every input-level detector has
to beat before its extra machinery has earned anything, and it beats PSBD on the
benign control. A method that cannot separate itself from a single forward pass
and a max is reading calibration rather than detecting backdoors.

    original form
        s(x) = max_c P(c | x; theta)

    descriptive form
        confidence = the largest softmax probability the model assigns to any class

Data requirement: none.
Forward-pass cost: 1 per input.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from defences.inference import forward_probs


@torch.inference_mode()
def confidence_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """Negated max softmax probability per sample, shape (N,), low meaning poisoned.

    Negated at the boundary so the shared convention holds. A backdoored model is
    usually more confident on a triggered input than on a clean one, so the raw
    statistic is high for poisoned, and returning it unnegated would read as a
    well-formed, exactly inverted result rather than as an error.
    """
    model.eval()

    batch_scores = []
    for images, _ in loader:
        probs = forward_probs(
            model, images, device, use_bfloat16
        )  # (batch, num_classes)
        batch_scores.append(-probs.max(dim=1).values.cpu())  # (batch,)

    if not batch_scores:
        return torch.empty(0)

    scores = torch.cat(batch_scores).float()  # (N,)
    return scores
