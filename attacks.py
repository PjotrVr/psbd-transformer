"""Registry mapping an attack name to its builder and default config.

Deterministic pixel-space attacks are implemented from their papers. Learned or
optimized attacks (SSBA, TrojanNN, ISSBA) are served through attack_generated from
pregenerated triggers. Adaptive-Blend and TaCT are standard-training attacks whose
specialization is cover samples, which train_backdoor reads from their config.
Generator-coupled attacks (Input-aware, LIRA) need a co-trained generator and a
bespoke loop, so they are not registered here.
"""

import attack_adaptive_blend
import attack_badnet
import attack_blend
import attack_bpp
import attack_generated
import attack_lc
import attack_lf
import attack_sig
import attack_tact
import attack_wanet
from poison import Attack


def _badnet_all_to_one() -> attack_badnet.BadNetConfig:
    return attack_badnet.BadNetConfig(label_mode="all_to_one")


def _badnet_all_to_all() -> attack_badnet.BadNetConfig:
    return attack_badnet.BadNetConfig(label_mode="all_to_all")


# name maps to (builder, config factory). A factory of None means the attack needs
# arguments with no sensible default, so its config must be built directly.
_ATTACKS = {
    "badnet": (attack_badnet.build, _badnet_all_to_one),
    "badnet_a2o": (attack_badnet.build, _badnet_all_to_one),
    "badnet_a2a": (attack_badnet.build, _badnet_all_to_all),
    "blend": (attack_blend.build, attack_blend.BlendConfig),
    "sig": (attack_sig.build, attack_sig.SigConfig),
    "wanet": (attack_wanet.build, attack_wanet.WaNetConfig),
    "lf": (attack_lf.build, attack_lf.LowFrequencyConfig),
    "lc": (attack_lc.build, attack_lc.LabelConsistentConfig),
    "bpp": (attack_bpp.build, attack_bpp.BppConfig),
    "adaptive_blend": (attack_adaptive_blend.build, attack_adaptive_blend.AdaptiveBlendConfig),
    "tact": (attack_tact.build, attack_tact.TactConfig),
    "generated": (attack_generated.build, None),
}

ATTACK_NAMES = tuple(_ATTACKS)


def default_config(attack_name: str):
    factory = _ATTACKS[attack_name][1]
    if factory is None:
        raise ValueError(f"{attack_name} has no default config, build its config directly")
    return factory()


def build_attack(attack_name: str, config, image_size: int, target_label: int) -> Attack:
    return _ATTACKS[attack_name][0](config, image_size, target_label)
