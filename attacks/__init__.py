from . import badnets, wanet

ATTACK_REGISTRY = {
    "badnets": badnets,
    "wanet": wanet,
}

__all__ = ["ATTACK_REGISTRY", "badnets", "wanet"]
