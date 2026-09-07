"""Registry mapping an attack name to its builder and default config.

Deterministic pixel-space attacks are implemented from their papers. Learned or
optimized attacks (SSBA, TrojanNN, ISSBA) are served through the generated
adapter from pregenerated triggers. Adaptive-Blend and TaCT are standard-training
attacks whose specialization is cover samples, which the training entrypoint reads
from their config. Generator-coupled attacks (Input-aware, LIRA) need a co-trained
generator and a bespoke loop, so they are not registered here.

This is the one module in psbd that re-exports its submodules' names, because it
is a genuine dispatcher: a caller writes build_attack("wanet", ...) and never has
to know which file wanet lives in.
"""

from psbd.poisoning import Attack

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


def _badnet_all_to_one() -> badnet.BadNetConfig:
    return badnet.BadNetConfig(label_mode="all_to_one")


def _badnet_all_to_all() -> badnet.BadNetConfig:
    return badnet.BadNetConfig(label_mode="all_to_all")


# name maps to (builder, config factory). A factory of None means the attack needs
# arguments with no sensible default, so its config must be built directly.
_ATTACKS = {
    "badnet": (badnet.build, _badnet_all_to_one),
    "badnet_a2o": (badnet.build, _badnet_all_to_one),
    "badnet_a2a": (badnet.build, _badnet_all_to_all),
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


def build_attack(
    attack_name: str, config, image_size: int, target_label: int
) -> Attack:
    """Dispatch to the named attack's builder and return its Attack record."""
    builder = _ATTACKS[attack_name][0]

    attack = builder(config, image_size, target_label)
    return attack
