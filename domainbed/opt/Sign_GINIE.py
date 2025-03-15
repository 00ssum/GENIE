import torch
from torch.optim import Optimizer
from typing import Optional


class Sign_GENIE(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            moving_avg: float = 0.95,
            convergence_rate: float = 0.015,
            p: float = 0.4):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            moving_avg=moving_avg,
            convergence_rate=convergence_rate,
            p=p)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("moving_avg", 0.95)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("convergence_rate", 0.015)
            group.setdefault("p", 0.4)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            moving_avg = group["moving_avg"]
            convergence_rate = group["convergence_rate"]
            p = group["p"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach().clone()
                if weight_decay != 0.0:
                    grad = grad.add(param.data, alpha=weight_decay)

                state = self.state[param]

                if "initialized" not in state:
                    state["gmean"] = torch.zeros_like(param.data)
                    state["ge2"] = torch.zeros_like(param.data)
                    state["prev"] = param.data.clone()  # prev_state[k]
                    state["scale"] = 0.0
                    state["initialized"] = True

                gmean = state["gmean"]
                ge2 = state["ge2"]
                prev = state["prev"]
                scale = state["scale"]

                new_scale = moving_avg * scale + 1.0
                scale1 = (1.0 - moving_avg) * new_scale
                scale2 = 2.0 - scale1
                rho = (1.0 - moving_avg) * scale2 / (1.0 + moving_avg) / scale1

                new_gmean = gmean * moving_avg + grad * (1.0 - moving_avg)
                new_ge2 = ge2 * moving_avg + grad.square() * (1.0 - moving_avg)

                gm = new_gmean / scale1
                ge2_ = new_ge2 / scale1
                var = ge2_ - gm.square()
                var = var / (1.0 - rho)
                var = torch.where(var > 0.0, var, torch.full_like(var, 1e-8))

                invvar = torch.clamp(1.0 / var, min=0.0, max=10.0)
                mvar = rho * var
                mvar = torch.where(mvar > 0.0, mvar, torch.full_like(mvar, 1e-8))

                tanh_invvar = torch.tanh(invvar)
                numerator = 1.0
                denominator = 1.0 + mvar / (gm.square() + 1e-8)
                pGsnr = (numerator / denominator) * tanh_invvar

                #noise_scale = (torch.sum(tanh_invvar * torch.abs(gm) * (numerator / denominator)) / (
                #    torch.sum(tanh_invvar)))
                #noise = torch.randn_like(grad) * noise_scale
                #grad_sgd = (1.0 - tanh_invvar) * noise

                pgrad = new_gmean / scale1 * pGsnr

                #mask = (torch.rand_like(param.data) > p).float()
                #mask.div_(1.0 - p)

                sign_g = torch.sign(gmean)
                zero_mask = (sign_g == 0)
                random_signs = torch.randint(0, 2, sign_x.shape) * 2 - 1  # 0 -> -1, 1 -> 1
                sign_g[zero_mask] = random_signs[zero_mask]

                update_val = pgrad * convergence_rate
                new_prev = prev - update_val

                param.data.copy_(new_prev)

                state["gmean"] = new_gmean
                state["ge2"] = new_ge2
                state["prev"] = new_prev
                state["scale"] = new_scale

        return loss