"""Sharpness-Aware Minimization (SAM), a wrapper over any base optimizer.

SAM (Foret et al., 2020) does not minimize the loss at the current weights but
the worst-case loss in a small neighborhood, which steers training toward flat
minima that generalize better.

    original objective
        min over w of  max over eps with l2_norm(eps) <= rho  of  L(w + eps)
    simplified objective
        minimize the highest loss reachable within a radius rho ball around w

Each update takes two forward-backward passes. The first finds the local
worst-case point w + eps_hat, and the second computes the gradient there and
hands it to the base optimizer. The base optimizer can be SGD or Adam, so this
is the "Adam merged with SAM" the request asked about: pass torch.optim.Adam as
the base and you get Adam updates on the sharpness-aware gradient.

This implementation follows the standard formulation. It is intentionally small
so it can be read and modified rather than pulled in as a dependency.
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer_cls,
        rho: float = 0.1,
        adaptive: bool = False,
        **base_kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascend to the local worst-case weights w + eps_hat.

        eps_hat points along the gradient and is scaled to length rho. The
        adaptive variant weights each parameter by its own magnitude, which makes
        the perturbation invariant to parameter scaling (ASAM).
        """
        gradient_norm = self._gradient_norm()
        for group in self.param_groups:
            scale = group["rho"] / (gradient_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["original"] = parameter.data.clone()
                per_parameter = torch.pow(parameter, 2) if group["adaptive"] else 1.0
                parameter.add_(per_parameter * parameter.grad * scale.to(parameter))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights and let the base optimizer update them.

        The gradient used here was computed at the worst-case point, so the base
        update is the sharpness-aware update.
        """
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data = self.state[parameter]["original"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _gradient_norm(self) -> torch.Tensor:
        reference_device = self.param_groups[0]["params"][0].device
        per_parameter_norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                weighting = torch.abs(parameter) if group["adaptive"] else 1.0
                per_parameter_norms.append(
                    (weighting * parameter.grad).norm(p=2).to(reference_device)
                )
        return torch.norm(torch.stack(per_parameter_norms), p=2)

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
