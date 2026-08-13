from . import adaptive_blend, badnet, blend, lc, wanet

ATTACK_REGISTRY = {
    "adaptive_blend": adaptive_blend,
    "badnet": badnet,
    "blend": blend,
    "lc": lc,
    "wanet": wanet,
}

__all__ = [
    "ATTACK_REGISTRY",
    "adaptive_blend",
    "badnet",
    "blend",
    "lc",
    "wanet",
]
