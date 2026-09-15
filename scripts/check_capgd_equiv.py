#!/usr/bin/env python
"""Check scripts/attack_context.py::capgd against the real TabularBench CAPGD.

The reference class is evasion-shaped (it maximises the CE of the *attacked row's
own* label) while ours is poisoning-shaped (aggregate CE on a separate test set),
so the class cannot be reused directly. The optimiser core is what has to match,
so we run both on an identical toy objective and compare the iterate sequences
element by element.

We load the reference's capgd.py straight from the repo, stubbing the heavy
imports it does not use on the Linf path, and construct the object with
__new__ so no scaler/constraints/objective-calculator machinery is required.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "tabularbench" / "tabularbench" / "attacks" / "capgd" / "capgd.py"
sys.path.insert(0, str(ROOT / "scripts"))


def load_reference():
    """Import the reference capgd.py with its unused dependencies stubbed out."""
    class _Attack:                                  # stand-in for torchattacks.Attack
        def __init__(self, name, model): pass

    stubs = {
        "torchattacks": {}, "torchattacks.attack": {"Attack": _Attack},
        "tabularbench.attacks.objective_calculator": {"ObjectiveCalculator": object},
        "tabularbench.attacks.utils": {"fix_equality_constraints": None,
                                       "fix_immutable": None, "fix_types": None},
        "tabularbench.constraints.constraints": {"Constraints": object},
        "tabularbench.constraints.constraints_backend_executor": {"ConstraintsExecutor": object},
        "tabularbench.constraints.pytorch_backend": {"PytorchBackend": object},
        "tabularbench.constraints.relation_constraint": {"AndConstraint": object},
        "tabularbench.models.model": {"Model": object},
        "tabularbench.models.tab_scaler": {"TabScaler": object},
        "tabularbench.utils.datatypes": {"to_numpy_number": None},
    }
    for name, attrs in stubs.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)

    spec = importlib.util.spec_from_file_location("_ref_capgd", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CAPGD


def run_reference(CAPGD, model, x, y, *, eps, steps, rho, device, norm="Linf"):
    """Drive attack_single_run directly, recording every iterate it evaluates."""
    a = CAPGD.__new__(CAPGD)
    a.steps, a.eps, a.norm = steps, eps, norm
    a.n_restarts, a.init_start, a.random_start = 1, True, True
    a.device, a.loss, a.eot_iter = device, "ce", 1
    a.thr_decr, a.adaptive_eps, a.best_restart, a.verbose = rho, True, True, False
    a.mutable_mask = torch.ones(x.shape[1], device=device)
    a.fix_equality_constraints_iter = False
    a.fix_equality_constraints_end = False
    a.constraints = SimpleNamespace(relation_constraints=None)

    traj = []
    def get_logits(xx):
        traj.append(xx.detach().clone())
        return model(xx)
    a.get_logits = get_logits

    x_best, _, loss_best, _ = a.attack_single_run(x, y, 0)
    return traj, x_best.detach(), float(loss_best.sum())


def run_ours(model, x, y, *, eps, steps, rho, device, norm="Linf"):
    from tabfm_experiments.capgd import capgd

    crit = torch.nn.CrossEntropyLoss(reduction="none")
    traj = []

    def evaluate(xx, need_grad):
        xs = xx.detach().requires_grad_(True) if need_grad else xx
        traj.append(xs.detach().clone())
        loss = crit(model(xs), y).sum()
        if not need_grad:
            return loss.detach(), None
        (g,) = torch.autograd.grad(loss, xs)
        return loss.detach(), g

    def repair(x1, x_ref, pad):
        if norm == "Linf":
            return torch.min(torch.max(x1, x_ref - eps), x_ref + eps).clamp(0.0, 1.0)
        d = x1 - x_ref
        dims = tuple(range(1, d.dim()))
        n = d.pow(2).sum(dim=dims, keepdim=True).sqrt()
        return (x_ref + d / (n + 1e-12) * torch.min(
            torch.full_like(d, eps), n + (1e-12 if pad else 0.0))).clamp(0.0, 1.0)

    x_best, loss_best, _ = capgd(
        evaluate, x, eps=eps, n_iter=steps, rho=rho, momentum=0.75,
        repair=repair, mut=torch.ones_like(x), better=lambda u, v: u > v,
        box=True, norm=norm,
    )
    return traj, x_best.detach(), loss_best


def main():
    device = "cpu"
    torch.manual_seed(0)
    D, C = 12, 4
    # A linear model would be a degenerate test: with 2 classes sign(grad) is
    # constant in x, so every sign-based step points the same way regardless of
    # the schedule. Use a genuinely non-linear net so sign(grad) actually flips.
    net = torch.nn.Sequential(
        torch.nn.Linear(D, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Sin() if hasattr(torch.nn, "Sin") else torch.nn.Tanh(),
        torch.nn.Linear(32, C),
    ).to(device)
    for prm in net.parameters():
        prm.requires_grad_(False)
    model = lambda xx: net(xx) * 3.0

    x = torch.rand(1, D, device=device)
    y = torch.tensor([2], device=device)

    CAPGD = load_reference()
    print(f"reference: {REF.relative_to(ROOT)}\n")
    ok = True
    for norm in ("Linf", "L2"):
      print(f"  --- norm = {norm}")
      for steps in (10, 25, 100):
        for eps in (0.475, 0.1):
            tr, xb_r, lb_r = run_reference(CAPGD, model, x.clone(), y, eps=eps,
                                           steps=steps, rho=0.75, device=device, norm=norm)
            to, xb_o, lb_o = run_ours(model, x.clone(), y, eps=eps,
                                      steps=steps, rho=0.75, device=device, norm=norm)
            sgn = [torch.sign(b - a_) for a_, b in zip(tr, tr[1:])]
            flips = sum(int((sgn[i] != sgn[i - 1]).any()) for i in range(1, len(sgn)))
            n = min(len(tr), len(to))
            diffs = [float((tr[i] - to[i]).abs().max()) for i in range(n)]
            first = next((i for i, d in enumerate(diffs) if d > 1e-6), None)
            same = (len(tr) == len(to)) and first is None
            ok &= same
            print(f"    steps={steps:>3} eps={eps:<6} iterates {len(tr)}v{len(to)}  "
                  f"max|dx| {max(diffs):.2e}  loss {lb_r:.6f} vs {lb_o:.6f}  "
                  f"dir-changes {flips:>3}  "
                  f"{'MATCH' if same else f'DIVERGES at iterate {first}'}")
    print("\n" + ("ALL MATCH" if ok else "MISMATCH -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
