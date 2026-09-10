"""Sharpness-Aware Minimization (SAM), a wrapper over any base optimizer.

SAM (Foret et al., 2020) does not minimize the loss at the current weights but
the worst-case loss in a small neighborhood, which steers training toward flat
minima that generalize better.

    original objective
        min over w of  max over eps with l2_norm(eps) <= rho  of  L(w + eps)
    simplified objective
        minimize the highest loss reachable within a radius rho ball around w

Each update takes 2 forward-backward passes. The first finds the local
worst-case point w + eps_hat, and the second computes the gradient there and
hands it to the base optimizer. The base optimizer can be SGD or Adam, so this
is "Adam merged with SAM": pass torch.optim.Adam as the base and you get Adam
updates on the sharpness-aware gradient.

Caller discipline, because this optimizer does not implement step(). A training
loop must run the 2 passes itself, in this order:

    loss_fn().backward()
    optimizer.first_step(zero_grad=True)   # move to w + eps_hat
    loss_fn().backward()                   # gradient AT the worst-case point
    optimizer.second_step(zero_grad=True)  # restore w, then base optimizer step

Skipping the second backward would hand second_step the gradient from the
original weights, which is a plain base-optimizer update wearing SAM's name and
costing twice as much. Calling second_step without first_step finds no saved
weights and crashes, which is the intended loud failure.
"""

import torch

# Guards the division when the gradient norm underflows to 0, for example on the
# first step of a frozen backbone.
GRADIENT_NORM_EPSILON = 1e-12


class SAM(torch.optim.Optimizer):
    """2-pass sharpness-aware wrapper around a base optimizer class.

    base_optimizer_cls is a class rather than an instance. It is constructed here
    over this optimizer's own param_groups so both share a single group list.
    """

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
            scale = group["rho"] / (gradient_norm + GRADIENT_NORM_EPSILON)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # second_step restores from here, so the clone has to happen
                # before the parameter is moved.
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
        """The global l2 norm of the gradient, as a 0-dim tensor.

        Norms are gathered onto a single reference device so a model sharded across
        devices still yields a single scalar.
        """
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

        global_norm = torch.norm(torch.stack(per_parameter_norms), p=2)
        return global_norm

    def load_state_dict(self, state_dict) -> None:
        """Restore state, then re-point param_groups at the base optimizer's list.

        Optimizer.load_state_dict rebuilds self.param_groups, which breaks the
        aliasing set up in __init__ and would leave the base optimizer stepping
        a stale group list.
        """
        super().load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
