"""The published input-level backdoor detectors PSBD is compared against.

Every detector here solves the problem PSBD solves: given a suspicious input and a
deployed model, decide whether the input carries a trigger. Methods that score a
whole poisoned training set by clustering its representations (Spectral
Signatures, Activation Clustering, SCAn) are absent, because they need the
training pool and produce a partition rather than a per-input decision.

    context = DetectorContext(model, device, mean, std, validation_loader=loader)
    detector = build_detector("strip", context)
    scores = detector(model, loader, device)

1 interface and 3 rules make the comparison mean something.

  Direction. Every detector returns per-sample scores where low means poisoned,
    PSU's convention, so defences.decision.detection_report applies to all of
    them unchanged. SCALE-UP, IBD-PSC and TeCo define statistics that are high
    for poisoned, and each is negated once at its own scoring boundary. Getting
    this wrong is silent: it yields a well-formed, exactly inverted result, which
    the two-sided field in detection_report exists to surface.

  Data budget. Every method that needs clean data gets the same clean validation
    split PSBD uses, the 2000-sample heldout slice from
    data.splits.build_psbd_loaders_from_checkpoint. 2 papers budget clean data
    differently, and those departures are recorded in their modules.

  Cost. Each module states its forward-pass count per input, and
    FORWARD_PASSES_PER_INPUT repeats it in machine-readable form. The counts are
    far from equal, and that is a real deployment constraint.

| Detector | Paper | Statistic | Clean data | Forwards per input |
|---|---|---|---|---|
| confidence | none, the null model | max softmax | none | 1 |
| strip | Gao et al., ACSAC 2019 | entropy under superimposition | 8 samples | 8 |
| scale_up | Guo et al., ICLR 2023 | label consistency under pixel amplification | none | 6 |
| scale_up_data_limited | Guo et al., ICLR 2023 | the same, standardized per class | the shared split | 6 |
| ibd_psc | Hou et al., ICML 2024 | retained confidence under parameter amplification | the shared split | 6 |
| teco | Liu et al., CVPR 2023 | spread of corruption hardness thresholds | none | 71 |

PSBD itself is not in this registry. It is scored through defences.inference and
defences.scores at k forward passes per input.
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

# N in STRIP Eq. (3). Named once, in the module that implements it, so a run's
# provenance and the registry's cost table cannot quote different values.
STRIP_OVERLAYS = strip_module.DEFAULT_NUM_OVERLAYS

DETECTOR_NAMES: tuple[str, ...] = (
    "confidence",
    "strip",
    "scale_up",
    "scale_up_data_limited",
    "ibd_psc",
    "teco",
)

# Buildable by name but outside DETECTOR_NAMES, so no default run, job or sign
# gate includes them until a smoke test has fixed their settings.
EXPERIMENTAL_DETECTOR_NAMES: tuple[str, ...] = ()

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

# Methods that fit per-sample statistics on the validation split and therefore
# return out-of-fit scores for it (jackknife, leave-one-out or 2 folds) rather
# than in-sample ones. A threshold set on in-sample scores is too tight, since a
# sample deviates less from statistics it helped fit.
CROSS_FITTED: frozenset[str] = frozenset({"scale_up_data_limited"})

# "autocast" methods run under context.use_bfloat16 like PSBD's own passes. A
# "float32" method forces full precision for its gradient step whatever the
# context says, because bf16 gradient noise changes an optimisation trajectory
# where it only rounds a forward pass. The effective dtype of a run is recorded
# in its provenance and must be identical across every cell a table compares.
PRECISION_POLICY: dict[str, str] = {
    "confidence": "autocast",
    "strip": "autocast",
    "scale_up": "autocast",
    "scale_up_data_limited": "autocast",
    "ibd_psc": "autocast",
    "teco": "autocast",
}

# The settings a run's provenance records, so 2 records can be compared for
# whether they measured the same method.
DETECTOR_HYPERPARAMETERS: dict[str, dict] = {
    "confidence": {},
    "strip": {"overlays": STRIP_OVERLAYS},
    "scale_up": {"scales": list(scale_up_module.PAPER_SCALES)},
    "scale_up_data_limited": {
        "scales": list(scale_up_module.PAPER_SCALES),
        "min_class_samples": scale_up_module.MIN_CLASS_SAMPLES,
        "cross_fit_folds": scale_up_module.CROSS_FIT_FOLDS,
    },
    "ibd_psc": {
        "scaling_factor": ibd_psc_module.DEFAULT_SCALING_FACTOR,
        "ensemble_size": ibd_psc_module.DEFAULT_ENSEMBLE_SIZE,
        "error_threshold": ibd_psc_module.DEFAULT_ERROR_THRESHOLD,
    },
    "teco": {
        "corruptions": list(teco_module.DEFAULT_CORRUPTIONS),
        "max_severity": teco_module.MAX_SEVERITY,
    },
}

Detector = Callable[[nn.Module, DataLoader, torch.device], torch.Tensor]


@dataclass(frozen=True)
class DetectorContext:
    """Everything a detector may need beyond the (model, loader, device) call.

    model and device are here as well as in the call because 2 methods fit state
    against the model before scoring: IBD-PSC runs Algorithm 1 to choose how many
    layers to amplify and SCALE-UP's data-limited variant estimates per-class
    clean statistics. build_detector fits eagerly, so the returned callable is
    cheap and stateless.

    validation_loader must be the clean validation split from
    data.splits.build_psbd_loaders_from_checkpoint, which keeps every method on
    the same data budget. mean and std are the dataset's normalization statistics,
    which SCALE-UP, TeCo and STRIP need to get back to pixel space.
    teco_corruptions defaults to all 14 and is overridable only because TeCo costs
    10 times what any other method costs, so a smoke run needs a cheaper answer
    that is visible in the call.
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


def effective_precision(name: str, context: DetectorContext) -> str:
    """The dtype a detector's model queries actually run in under this context."""
    if PRECISION_POLICY[name] == "float32":
        return "float32"
    autocast_applies = context.use_bfloat16 and context.device.type == "cuda"
    precision = "bfloat16" if autocast_applies else "float32"
    return precision


def _require_validation_loader(name: str, context: DetectorContext) -> DataLoader:
    """The clean validation split, or a message naming what is missing and why."""
    if context.validation_loader is None:
        raise ValueError(
            f"detector {name!r} needs clean data ({DATA_REQUIREMENT[name]}) and "
            "context.validation_loader is None. Pass the 'validation' loader from "
            "data.splits.build_psbd_loaders_from_checkpoint, which is the split "
            "every method here shares."
        )
    return context.validation_loader


def _guard_same_model(name: str, fitted: nn.Module, given: nn.Module) -> None:
    """Refuse to score with a model the detector was not fitted against.

    IBD-PSC holds direct references to the fitted model's LayerNorm modules, so
    scoring a different model would amplify the fitted model's parameters and read
    the other model's logits. That produces a complete, plausible, meaningless
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
            context.mean,
            context.std,
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

    validation_spc, validation_predicted, validation_labels = (
        scale_up_module.spc_scores(
            context.model,
            validation_loader,
            context.device,
            context.mean,
            context.std,
            scale_up_module.PAPER_SCALES,
            context.use_bfloat16,
        )
    )
    class_means, class_stds = scale_up_module.fit_class_spc_statistics(
        validation_spc, validation_labels, context.num_classes
    )
    # The threshold set is scored out of fit, see cross_fitted_validation_scores.
    validation_scores = scale_up_module.cross_fitted_validation_scores(
        validation_spc, validation_predicted, validation_labels, context.num_classes
    )

    def score(model: nn.Module, loader: DataLoader, device: torch.device):
        if loader is context.validation_loader:
            return validation_scores
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
    """The named detector, fitted against context, as a (model, loader, device) callable.

    It returns per-sample scores in the loader's own order, low meaning poisoned.
    Fitting happens here rather than at scoring time, so the clean validation
    split is read once per model, and a detector in NEEDS_FITTING runs forward
    passes here. An unknown name raises, since a silent fallback would produce a
    plausible comparison table answering a different question.
    """
    if name not in DETECTOR_BUILDERS:
        raise KeyError(f"unknown detector {name!r}, known: {sorted(DETECTOR_BUILDERS)}")

    detector = DETECTOR_BUILDERS[name](context)
    return detector
