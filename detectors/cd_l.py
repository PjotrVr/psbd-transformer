"""CD-L: Cognitive Distillation on logits (Huang et al., ICLR 2023).

Paper: "Distilling Cognitive Backdoor Patterns within an Image", arXiv:2301.10908.
The objective is Section 3.1, Equations (1) and (2), the detection rule Section
3.2, Equation (4) and the optimiser settings Appendix B.3.

    original form
        argmin_m || f_theta(x) - f_theta(x_cp) ||_1
                 + alpha * || m ||_1 + beta * TV(m)                    Eq. (1)
        x_cp = x ⊙ m + (1 - m) ⊙ delta                                 Eq. (2)
        g(x) = 1 if || m ||_1 <= t, 0 otherwise                        Eq. (4)

    descriptive form
        distilled_input = pixels * mask + (1 - mask) * random_fill
        objective       = mean over classes of |logits(distilled_input) - logits(pixels)|
                          + l1_weight * sum(mask) + tv_weight * total_variation(mask)
        cd_l_score      = sum of the final mask, low meaning poisoned

f_theta is the model's logit map, m in [0, 1]^{h x w} a single-channel mask shared
by the colour channels, delta in [0, 1]^c a per-channel uniform fill redrawn at
every optimisation step, TV the total variation and t a threshold set on clean
data. The mask is parametrised through a scaled tanh, so the optimiser works on
an unconstrained parameter and the mask itself never leaves [0, 1].

Mechanism. The L1 term pushes the mask toward 0 everywhere, and only pixels whose
removal changes the logits can hold it up. A clean input's logits rest on the
object, spread over many pixels, so the mask stays large. A triggered input's
logits are dominated by the backdoor path, and the trigger's own pixels are all
the model needs to reproduce them, so the mask collapses onto the trigger and its
L1 norm is small. Eq. (4) flags a mask whose L1 norm falls below a threshold, the
same quantile-of-clean-validation rule defences.decision applies here.

Data requirement: none. The objective compares the model against itself, so no
label and no clean image enters the score. Clean data is used for the threshold
only, exactly as for every other detector.
Forward-pass cost: 1 reference pass plus num_steps forward-and-backward passes
per input, 251 forward-equivalents at the default 100 steps with a backward
counted as 1.5 forwards.

Deviations from the paper, each recorded in full in docs/detectors/cd_l.md:

  1. The TV weight is 1.0, the released code's default, where Appendix B.3
     states 10. The paper's own ablation finds detection insensitive to it.
  2. Both the reference pass and every distilled pass are normalised. The
     released class normalises the reference pass only, harmless there because
     its models consume [0, 1] input, wrong on a model that does not.
  3. The mask lives at the dataset's native resolution and the model's own
     Resize upsamples the distilled input. At 224 the L1 term would be 50 times
     the paper's magnitude and swamp the logit term.
  4. The fill is drawn on the model's device rather than on the CPU and moved,
     so on CUDA it comes from a different generator than the released code's.
  5. Gradients reach the mask parameter only. The model is frozen for the
     duration and no parameter ever holds a .grad.
  6. Forward and backward run under the shared autocast policy with the mask,
     the Adam state and the loss in float32. The released code is float32 end
     to end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader

from defences.inference import forward_logits, frozen_parameters

from .strip import normalization_buffers

# Adam's step size, Appendix B.3 "initial learning rate 0.1" and the reference
# class's lr default on its line 14.
DEFAULT_LEARNING_RATE = 0.1

# Appendix B.3 sets beta_1 = beta_2 = 0.1, the reference passes betas=(0.1, 0.1)
# on line 36. With this little memory Adam is close to sign descent, so each
# step moves every mask parameter by about the learning rate.
ADAM_BETAS = (0.1, 0.1)

# Appendix B.3 "a total of 100 steps", the reference default on line 14.
DEFAULT_NUM_STEPS = 100

# alpha in Eq. (1), which the released code names gamma. 0.01 is the CD-L value
# for CIFAR-10 and GTSRB in Appendix B.3 and the reference default on line 14.
DEFAULT_L1_WEIGHT = 0.01

# beta in Eq. (1). The reference default on line 14 is 1.0, Appendix B.3 says 10,
# and Figure 10 shows detection unchanged across 1 to 100. Deviation 1.
DEFAULT_TV_WEIGHT = 1.0

# p of the reference's torch.norm, line 14. Eq. (1) and Eq. (4) both use L1.
MASK_NORM = 1

# The reference initialises the parameter with torch.ones on line 35, so the
# effective mask starts at (tanh(1) + 1) / 2 = 0.8808 rather than at 1, where
# the tanh gradient would be too small to move it.
MASK_PARAMETER_INIT = 1.0

# mask_channel=1 on line 14, Eq. (2)'s m in [0, 1]^{w x h} shared by the channels.
MASK_CHANNELS = 1


def effective_mask(mask_parameter: torch.Tensor) -> torch.Tensor:
    """The mask in [0, 1] from its unconstrained parameter, same shape in.

    original form (reference get_raw_mask)
        m = (tanh(theta) + 1) / 2
    descriptive form
        mask = scaled tanh of the parameter, so every real value lands in [0, 1]

    Appendix B.3's "scaled tanh function to ensure the mask is between [0, 1]".
    """
    mask = (torch.tanh(mask_parameter) + 1) / 2  # same shape as mask_parameter
    return mask


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    """TV(m) of Eq. (1) per image, shape (batch,), in the reference's normalisation.

    original form (reference total_variation_loss)
        TV(m) = ( sum_{i,j} (m_{i+1,j} - m_{i,j})^2
                + sum_{i,j} (m_{i,j+1} - m_{i,j})^2 ) / (c * h * w)
    descriptive form
        total_variation = squared differences between vertically adjacent and
                          horizontally adjacent mask entries, summed, divided by
                          the mask's element count

    Squared rather than absolute differences, and divided by the mask's own
    element count, because that is what the released code computes. The paper
    writes TV(m) without defining it.
    """
    batch, channels, height, width = mask.shape

    # Squared differences between adjacent rows, (batch, 1, height - 1, width),
    # and between adjacent columns, (batch, 1, height, width - 1).
    vertical = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).pow(2)
    horizontal = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).pow(2)
    summed = vertical.sum(dim=(1, 2, 3)) + horizontal.sum(dim=(1, 2, 3))  # (batch,)

    normalized = summed / (channels * height * width)  # (batch,)
    return normalized


def mask_norms(masks: torch.Tensor) -> torch.Tensor:
    """|| m ||_1 per image, the Eq. (4) statistic, shape (batch,), low meaning poisoned.

    torch.linalg.vector_norm is the kernel torch.norm(p=1) dispatches to, and
    the reference calls torch.norm. abs().sum() reduces in a different order and
    lands about 2e-4 away on a 32 by 32 mask, which is enough to fail a bit-level
    comparison against the released class.
    """
    norms = torch.linalg.vector_norm(masks, ord=MASK_NORM, dim=(1, 2, 3))  # (batch,)
    return norms


def distill_masks(
    model: nn.Module,
    images: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    use_bfloat16: bool,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l1_weight: float = DEFAULT_L1_WEIGHT,
    tv_weight: float = DEFAULT_TV_WEIGHT,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> torch.Tensor:
    """Eq. (1) solved for 1 batch, the final effective masks, (batch, 1, height, width).

    images is the batch exactly as the loader serves it, normalised and at the
    dataset's native resolution. The reference logits come from that tensor. The
    mask is optimised in [0, 1] pixel space at the same resolution, the Eq. (2)
    blend is renormalised with the same statistics, and the model's own Resize
    takes the distilled input up to 224.

    The model is frozen for the whole loop and the gradient is taken against the
    mask parameter alone, so no model parameter ever receives a .grad. Nothing
    here runs under inference_mode, which would sever the graph the gradient
    needs. The forward runs under the shared autocast policy through
    forward_logits, the backward outside it, which is the order autocast
    documents. The mask, the Adam state and the objective stay float32.
    """
    model.eval()
    images = images.to(device)  # (batch, channels, height, width), normalised
    batch, channels, height, width = images.shape

    mean_tensor, std_tensor = normalization_buffers(
        mean, std, images.device, images.dtype
    )
    pixels = (images * std_tensor + mean_tensor).clamp(
        0.0, 1.0
    )  # (batch, channels, height, width)

    mask_parameter = torch.full(
        (batch, MASK_CHANNELS, height, width),
        MASK_PARAMETER_INIT,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )  # (batch, 1, height, width)
    optimizer = torch.optim.Adam([mask_parameter], lr=learning_rate, betas=ADAM_BETAS)

    with torch.enable_grad(), frozen_parameters(model):
        reference_logits = forward_logits(
            model, images, device, use_bfloat16
        ).detach()  # (batch, num_classes)

        for _ in range(num_steps):
            mask = effective_mask(mask_parameter)  # (batch, 1, height, width)

            # Redrawn every step, as Eq. (2) prescribes, so a pixel that is 0 in
            # the image is still distinguishable from a pixel the mask removed.
            fill = torch.rand(
                batch, channels, 1, 1, device=device
            )  # (batch, channels, 1, 1)

            assert mask.shape == (batch, MASK_CHANNELS, height, width), (
                f"mask {tuple(mask.shape)} does not match pixels {tuple(pixels.shape)}"
            )
            distilled = (
                pixels * mask + (1 - mask) * fill
            )  # (batch, channels, height, width), Eq. (2)
            renormalized = (
                distilled - mean_tensor
            ) / std_tensor  # (batch, channels, height, width)
            distilled_logits = forward_logits(
                model, renormalized, device, use_bfloat16
            )  # (batch, num_classes)

            logit_gap = F.l1_loss(
                distilled_logits, reference_logits, reduction="none"
            ).mean(dim=1)  # (batch,)
            sparsity = l1_weight * torch.linalg.vector_norm(
                mask, ord=MASK_NORM, dim=(1, 2, 3)
            )  # (batch,)
            smoothness = tv_weight * total_variation(mask)  # (batch,)
            objective = (logit_gap + sparsity + smoothness).mean()  # Eq. (1), scalar

            (gradient,) = torch.autograd.grad(objective, mask_parameter)
            mask_parameter.grad = gradient  # (batch, 1, height, width)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    masks = effective_mask(mask_parameter).detach()  # (batch, 1, height, width)
    return masks


def cd_l_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    use_bfloat16: bool = True,
    seed: int = 0,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> torch.Tensor:
    """CD-L score per sample, shape (N,), float32 on the CPU, low meaning poisoned.

    Scores come back in the loader's own order. The seed fixes the fill sequence,
    so the same model and loader reproduce the same masks.

    Returned as the raw mask L1 norm, not negated. Eq. (4) flags a sample whose
    norm falls at or below the threshold, so the paper's statistic is already low
    for poisoned and negating it would invert the detector.
    """
    model.eval()
    seed_everything(seed)

    batch_scores = []
    for images, _ in loader:
        masks = distill_masks(
            model, images, mean, std, device, use_bfloat16, num_steps=num_steps
        )  # (batch, 1, height, width)
        batch_scores.append(mask_norms(masks).cpu())  # (batch,)

    if not batch_scores:
        return torch.empty(0)

    scores = torch.cat(batch_scores).float()  # (N,)
    if not torch.isfinite(scores).all():
        raise ValueError(
            f"{int((~torch.isfinite(scores)).sum())} of {scores.numel()} CD-L scores "
            "are not finite, so the mask optimisation diverged"
        )
    return scores
