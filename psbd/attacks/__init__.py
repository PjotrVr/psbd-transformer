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

from typing import Callable

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


# The all_to_m family, the interpolation between the 2 poles PSBD's premise sits
# between. m is the number of distinct classes the trigger maps onto, so the
# backdoor map must encode log2(m) bits about the image. m = 1 is exactly
# all_to_one on target 0 and m = num_classes is exactly all_to_all, so those 2
# poles stay served by badnet_a2o and badnet_a2a rather than being duplicated
# here. Powers of 2 because the axis that matters is log2(m), not m. An m above a
# dataset's class count is rejected at build time rather than silently degenerating.
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


def build_attack(
    attack_name: str, config, image_size: int, target_label: int
) -> Attack:
    """Dispatch to the named attack's builder and return its Attack record."""
    builder = _ATTACKS[attack_name][0]

    attack = builder(config, image_size, target_label)
    return attack
