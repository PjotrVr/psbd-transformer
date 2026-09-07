"""The published input-level backdoor detectors PSBD is compared against.

Every detector here solves the same problem PSBD solves: given 1 suspicious input
and a deployed model, decide whether that input carries a trigger. Methods that
score a whole poisoned TRAINING set by clustering its representations (Spectral
Signatures, Activation Clustering, SCAn) are deliberately absent, because they
need the training pool and produce a partition rather than a per-input decision,
so their numbers would answer a different question in the same table.

One interface, 3 rules that make the comparison mean something.

  Direction. Every detector returns a float tensor of per-sample scores where LOW
    means poisoned, which is PSU's convention, so psbd.decision.detection_report
    applies to all of them unchanged. SCALE-UP, IBD-PSC and TeCo all define
    statistics that are HIGH for poisoned, and each is negated once, at its own
    scoring boundary, with a comment saying so. Getting this wrong is silent: it
    yields a confident, well-formed, exactly inverted result, which is what the
    two-sided field in detection_report exists to surface.

  Data budget. Every method that needs clean data gets the SAME clean validation
    split PSBD uses, the 2000-sample heldout slice from
    psbd.splits.build_psbd_loaders_from_checkpoint. No method sees more data than
    another, and each module's docstring states its requirement explicitly. This
    departs from 2 of the papers, which budget clean data differently (SCALE-UP
    asks for 100 samples per class, IBD-PSC for 100 in total), and those
    departures are recorded in the modules concerned.

  Cost. Each module's docstring states its forward-pass count per input, and
    FORWARD_PASSES_PER_INPUT below repeats it in machine-readable form for the
    comparison table. The counts are far from equal and that is a real
    deployment constraint, not an implementation detail.

| Detector | Paper | Statistic | Clean data | Forwards per input |
|---|---|---|---|---|
| confidence | none, the null model | max softmax | none | 1 |
| strip | Gao et al., ACSAC 2019 | entropy under superimposition | 8 samples | 8 |
| scale_up | Guo et al., ICLR 2023 | label consistency under pixel amplification | none | 6 |
| scale_up_data_limited | Guo et al., ICLR 2023 | the same, standardized per class | the shared split | 6 |
| ibd_psc | Hou et al., ICML 2024 | retained confidence under parameter amplification | the shared split | 6 |
| teco | Liu et al., CVPR 2023 | spread of corruption hardness thresholds | none | 71 |

PSBD itself is not in this registry. It is scored through psbd.inference and
psbd.scores, whose cost is k forward passes at the chosen probe rate.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import confidence as confidence_module
from . import ibd_psc as ibd_psc_module
from . import scale_up as scale_up_module
from . import strip as strip_module
from . import teco as teco_module

# N in STRIP Eq. (3). Matches the value every already-recorded baseline number in
# this repo was produced with.
STRIP_OVERLAYS = 8

DETECTOR_NAMES: tuple[str, ...] = (
    "confidence",
    "strip",
    "scale_up",
    "scale_up_data_limited",
    "ibd_psc",
    "teco",
)

# Model queries per scored input, the deployment cost of each method. Counts
# include the 1 unamplified or uncorrupted pass a method needs to fix its own
# reference label, since a defender pays for that too.
FORWARD_PASSES_PER_INPUT: dict[str, int] = {
    "confidence": 1,
    "strip": STRIP_OVERLAYS,
    "scale_up": len(scale_up_module.PAPER_SCALES) + 1,
    "scale_up_data_limited": len(scale_up_module.PAPER_SCALES) + 1,
    "ibd_psc": ibd_psc_module.DEFAULT_ENSEMBLE_SIZE + 1,
    "teco": len(teco_module.DEFAULT_CORRUPTIONS) * teco_module.MAX_SEVERITY + 1,
}

# What each method needs from the shared clean validation split. "none" means the
# method is data-free and would run against a model the defender holds no data for.
DATA_REQUIREMENT: dict[str, str] = {
    "confidence": "none",
    "strip": f"{STRIP_OVERLAYS} clean images, unlabelled",
    "scale_up": "none",
    "scale_up_data_limited": "the clean validation split, labelled",
    "ibd_psc": "the clean validation split, labelled",
    "teco": "none",
}

# Methods whose build step runs forward passes over the clean validation split
# before any input is scored. Listed so a caller can report that fixed cost
# separately from the per-input cost, which is what a deployment would care about.
NEEDS_FITTING: frozenset[str] = frozenset({"scale_up_data_limited", "ibd_psc"})

Detector = Callable[[nn.Module, DataLoader, torch.device], torch.Tensor]


@dataclass(frozen=True)
class DetectorContext:
    """Everything a detector may need beyond the (model, loader, device) call.

    model and device are here as well as in the call signature because 2 methods
    fit state against the model before scoring anything: IBD-PSC runs Algorithm 1
    to choose how many layers to amplify, and SCALE-UP's data-limited variant
    estimates per-class clean statistics. build_detector performs that fitting
    eagerly, so the returned callable is cheap and stateless from then on.

    validation_loader must be the clean validation split from
    psbd.splits.build_psbd_loaders_from_checkpoint, which is what keeps every
    method on the same data budget. It is required for strip, ibd_psc, and
    scale_up_data_limited, and unused by the rest.

    mean and std are the dataset's normalization statistics from
    psbd.config.DATASET_REGISTRY. SCALE-UP and TeCo both operate in [0, 1] pixel
    space and need them to undo what the loader did.

    teco_corruptions names which of TeCo's corruption types to use. It defaults
    to all 14 available, which is the setting every reported number should use.
    It is overridable only because TeCo costs 10 times what any other method here
    costs, so a smoke run or a test needs a way to buy a cheaper answer, and a
    reduced set has to be visible in the call rather than hidden in a flag.
    """

    model: nn.Module
    device: torch.device
    mean: tuple[float, ...]
    std: tuple[float, ...]
    validation_loader: DataLoader | None = None
    num_classes: int | None = None
    use_bfloat16: bool = True
    seed: int = 0
    teco_corruptions: tuple[str, ...] = teco_module.DEFAULT_CORRUPTIONS


def _require_validation_loader(name: str, context: DetectorContext) -> DataLoader:
    """The clean validation split, or a message naming what is missing and why."""
    if context.validation_loader is None:
        raise ValueError(
            f"detector {name!r} needs clean data ({DATA_REQUIREMENT[name]}) and "
            "context.validation_loader is None. Pass the 'validation' loader from "
            "psbd.splits.build_psbd_loaders_from_checkpoint, which is the split "
            "every method here shares."
        )
    return context.validation_loader


def _guard_same_model(name: str, fitted: nn.Module, given: nn.Module) -> None:
    """Refuse to score with a model the detector was not fitted against.

    IBD-PSC holds direct references to the fitted model's LayerNorm modules, so
    scoring a different model would amplify one model's parameters and read
    another model's logits. That produces a complete, plausible, meaningless
    number, so it is rejected rather than allowed.
    """
    if given is not fitted:
        raise ValueError(
            f"detector {name!r} was fitted against a different model instance than "
            "the one passed to score. Rebuild it with build_detector for this model."
        )


def _build_confidence(context: DetectorContext) -> Detector:
    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        return confidence_module.confidence_scores(
            model, loader, device, context.use_bfloat16
        )

    return score


def _build_strip(context: DetectorContext) -> Detector:
    validation_loader = _require_validation_loader("strip", context)
    overlays = strip_module.collect_overlay_batch(
        validation_loader, STRIP_OVERLAYS, context.seed
    )

    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        return strip_module.strip_scores(
            model,
            loader,
            device,
            overlays,
            context.use_bfloat16,
            context.seed,
            STRIP_OVERLAYS,
        )

    return score


def _build_scale_up(context: DetectorContext) -> Detector:
    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        return scale_up_module.scale_up_scores(
            model,
            loader,
            device,
            context.mean,
            context.std,
            scale_up_module.PAPER_SCALES,
            context.use_bfloat16,
        )

    return score


def _build_scale_up_data_limited(context: DetectorContext) -> Detector:
    validation_loader = _require_validation_loader("scale_up_data_limited", context)
    if context.num_classes is None:
        raise ValueError(
            "scale_up_data_limited needs context.num_classes to group Eq. (3) by class"
        )

    validation_spc, _, validation_labels = scale_up_module.spc_scores(
        context.model,
        validation_loader,
        context.device,
        context.mean,
        context.std,
        scale_up_module.PAPER_SCALES,
        context.use_bfloat16,
    )
    class_means, class_stds = scale_up_module.fit_class_spc_statistics(
        validation_spc, validation_labels, context.num_classes
    )

    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        return scale_up_module.scale_up_scores(
            model,
            loader,
            device,
            context.mean,
            context.std,
            scale_up_module.PAPER_SCALES,
            context.use_bfloat16,
            class_means,
            class_stds,
        )

    return score


def _build_ibd_psc(context: DetectorContext) -> Detector:
    validation_loader = _require_validation_loader("ibd_psc", context)

    ordered_layers = ibd_psc_module.amplifiable_norm_layers(context.model)
    start_layer_count, _ = ibd_psc_module.select_start_layer_count(
        context.model,
        validation_loader,
        context.device,
        ordered_layers,
        ibd_psc_module.DEFAULT_SCALING_FACTOR,
        ibd_psc_module.DEFAULT_ERROR_THRESHOLD,
        context.use_bfloat16,
    )

    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        _guard_same_model("ibd_psc", context.model, model)
        return ibd_psc_module.ibd_psc_scores(
            model,
            loader,
            device,
            ordered_layers,
            start_layer_count,
            ibd_psc_module.DEFAULT_SCALING_FACTOR,
            ibd_psc_module.DEFAULT_ENSEMBLE_SIZE,
            context.use_bfloat16,
        )

    return score


def _build_teco(context: DetectorContext) -> Detector:
    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        return teco_module.teco_scores(
            model,
            loader,
            device,
            context.mean,
            context.std,
            context.teco_corruptions,
            teco_module.MAX_SEVERITY,
            context.use_bfloat16,
            context.seed,
        )

    return score


DETECTOR_BUILDERS: dict[str, Callable[[DetectorContext], Detector]] = {
    "confidence": _build_confidence,
    "strip": _build_strip,
    "scale_up": _build_scale_up,
    "scale_up_data_limited": _build_scale_up_data_limited,
    "ibd_psc": _build_ibd_psc,
    "teco": _build_teco,
}


def build_detector(name: str, context: DetectorContext) -> Detector:
    """Look up a detector by name and fit whatever it needs, failing loudly on a typo.

    Returns a callable taking (model, loader, device) and returning a float tensor
    of per-sample scores in the loader's own order, where LOW means poisoned.

    Fitting happens here, not at scoring time, so the cost of reading the clean
    validation split is paid once per model rather than once per split scored. For
    a detector in NEEDS_FITTING this call runs forward passes and is not cheap.

    A silent fallback on an unknown name would produce a complete, plausible
    comparison table answering a different question than the one asked, so an
    unknown name raises.
    """
    if name not in DETECTOR_BUILDERS:
        raise KeyError(f"unknown detector {name!r}, known: {sorted(DETECTOR_BUILDERS)}")

    detector = DETECTOR_BUILDERS[name](context)
    return detector
