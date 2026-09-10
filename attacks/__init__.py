"""The attack registry: a name resolves to a builder and its paper defaults.

    attack = build_attack("wanet", default_config("wanet"), image_size=32, target_label=0)

Each attack lives in its own module and defines only its trigger and config. The
shared machinery (the Attack record, label policy, poisoned datasets) is in
poisoning.py. Deterministic pixel-space attacks are implemented from their papers.
Learned triggers (SSBA, TrojanNN, ISSBA) arrive as pregenerated images through
generated.py. Attacks that need a co-trained generator (Input-aware, LIRA) are not
registered, since they need a bespoke training loop.

This is the only package that re-exports its submodules' names, because it is a
dispatcher: a caller should never have to know which file an attack lives in.
"""

from dataclasses import asdict, replace
from dataclasses import fields as dataclass_fields
from typing import Callable

from attacks.poisoning import Attack

from . import (
    adaptive_blend,
    badnet,
    blend,
    bpp,
    generated,
    lc,
    lf,
    sig,
    tact,
    wanet,
)
from .bases import adversarial_config_error as adversarial_config_error
from .bases import missing_adversarial_bases as missing_adversarial_bases


def _badnet_all_to_one() -> badnet.BadNetConfig:
    return badnet.BadNetConfig(label_mode="all_to_one")


def _badnet_all_to_all() -> badnet.BadNetConfig:
    return badnet.BadNetConfig(label_mode="all_to_all")


# The all_to_m family interpolates between all_to_one (m = 1) and all_to_all
# (m = num_classes). m is how many distinct classes the trigger maps onto, so the
# backdoor has to encode log2(m) bits about the image, which is why the registered
# values are powers of 2. The 2 poles keep their own names rather than being
# duplicated here. An m above the dataset's class count is rejected at build time.
def _badnet_all_to_m(num_targets: int) -> Callable[[], badnet.BadNetConfig]:
    def factory() -> badnet.BadNetConfig:
        return badnet.BadNetConfig(label_mode="all_to_m", num_targets=num_targets)

    return factory


# name maps to (builder, config factory). A factory of None means the attack needs
# arguments with no sensible default, so its config must be built directly.
_ATTACKS = {
    "badnet": (badnet.build, _badnet_all_to_one),
    "badnet_a2o": (badnet.build, _badnet_all_to_one),
    "badnet_a2a": (badnet.build, _badnet_all_to_all),
    "badnet_a2m2": (badnet.build, _badnet_all_to_m(2)),
    "badnet_a2m4": (badnet.build, _badnet_all_to_m(4)),
    "badnet_a2m8": (badnet.build, _badnet_all_to_m(8)),
    "badnet_a2m16": (badnet.build, _badnet_all_to_m(16)),
    "badnet_a2m32": (badnet.build, _badnet_all_to_m(32)),
    "badnet_a2m64": (badnet.build, _badnet_all_to_m(64)),
    "badnet_a2m128": (badnet.build, _badnet_all_to_m(128)),
    "blend": (blend.build, blend.BlendConfig),
    "sig": (sig.build, sig.SigConfig),
    "wanet": (wanet.build, wanet.WaNetConfig),
    "lf": (lf.build, lf.LowFrequencyConfig),
    "lc": (lc.build, lc.LabelConsistentConfig),
    "bpp": (bpp.build, bpp.BppConfig),
    "adaptive_blend": (adaptive_blend.build, adaptive_blend.AdaptiveBlendConfig),
    "tact": (tact.build, tact.TactConfig),
    "generated": (generated.build, None),
}

ATTACK_NAMES = tuple(_ATTACKS)


def default_config(attack_name: str):
    """A fresh config dataclass carrying the attack's paper defaults."""
    factory = _ATTACKS[attack_name][1]
    if factory is None:
        raise ValueError(
            f"{attack_name} has no default config, build its config directly"
        )

    config = factory()
    return config


def apply_config_overrides(config, overrides: dict | None):
    """The config with the given fields replaced, each cast to its declared type.

    Training and evaluation both go through this, so a checkpoint is always rebuilt
    with the trigger it was trained with. Rebuilding from the defaults instead would
    hand the attack-success set a trigger the model never saw and read a near-zero
    ASR that has nothing to do with the attack. JSON turns tuples into lists, so a
    tuple-valued field is restored as a tuple.
    """
    if not overrides:
        return config
    declared = {field.name: field for field in dataclass_fields(config)}
    updates = {}
    for key, value in overrides.items():
        if key not in declared:
            raise ValueError(
                f"{type(config).__name__} has no field {key!r}; "
                f"available: {sorted(declared)}"
            )
        annotation = declared[key].type
        if annotation is float:
            updates[key] = float(value)
        elif annotation is int:
            updates[key] = int(value)
        elif isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return replace(config, **updates)


def config_overrides(config, attack_name: str) -> dict:
    """The fields of config that differ from the attack's defaults.

    Only the difference is recorded, so provenance stays small and a field nobody
    touched cannot be corrupted by a JSON type round-trip.
    """
    try:
        default = default_config(attack_name)
    except ValueError:
        return {}
    current, baseline = asdict(config), asdict(default)
    return {key: value for key, value in current.items() if baseline.get(key) != value}


def build_attack(
    attack_name: str, config, image_size: int, target_label: int
) -> Attack:
    """The named attack, built for this image size and target label."""
    builder = _ATTACKS[attack_name][0]

    attack = builder(config, image_size, target_label)
    return attack
