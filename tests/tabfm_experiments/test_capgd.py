import sys
from pathlib import Path

import pytest
import torch

from tabfm_experiments.capgd import capgd, project_ball, random_start_point

ROOT = Path(__file__).resolve().parents[2]


def _sum_evaluate(x, need_grad):
    return x.sum().detach(), torch.ones_like(x)


def _repair(eps, norm):
    return lambda x1, x_ref, pad: project_ball(x1, x_ref, eps, norm, pad).clamp(0.0, 1.0)


def test_linf_sum_hand_checked():
    x0 = torch.full((1, 4), 0.3)
    eps = 0.1
    x_best, loss_best, hist = capgd(
        _sum_evaluate, x0, eps=eps, n_iter=3, rho=0.75, momentum=0.75,
        repair=_repair(eps, "Linf"), mut=torch.ones_like(x0), better=lambda u, v: u > v,
        box=True, norm="Linf",
    )
    # step 1: x + 2*eps*sign(1) then clamp to the eps-ball -> 0.4 everywhere
    assert torch.allclose(x_best, torch.full((1, 4), 0.4))
    assert loss_best == pytest.approx(1.6)
    assert hist[0]["loss"] == pytest.approx(1.2)
    assert len(hist) == 4


def test_mutable_mask_and_box():
    x0 = torch.tensor([[0.95, 0.5]])
    mut = torch.tensor([[1.0, 0.0]])
    x_best, _, _ = capgd(
        _sum_evaluate, x0, eps=0.2, n_iter=2, rho=0.75, momentum=0.75,
        repair=_repair(0.2, "Linf"), mut=mut, better=lambda u, v: u > v, box=True, norm="Linf",
    )
    assert x_best[0, 0] == pytest.approx(1.0)  # box
    assert x_best[0, 1] == pytest.approx(0.5)  # immutable


def test_random_restart_same_seed_same_result():
    x0 = torch.full((2, 5), 0.5)
    mut = torch.ones(1, 5)
    outs = []
    for seed in (7, 7, 8):
        g = torch.Generator().manual_seed(seed)
        xi = random_start_point(x0, eps=0.3, norm="L2", mut=mut, generator=g)
        assert torch.all((xi - x0).pow(2).sum(1).sqrt() <= 0.3 + 1e-6)
        w = torch.linspace(-1, 1, 5)
        ev = lambda x, ng: ((x * w).sin().sum().detach(), (x * w).cos() * w)
        xb, _, _ = capgd(ev, x0, eps=0.3, n_iter=5, rho=0.75, momentum=0.75,
                         repair=_repair(0.3, "L2"), mut=mut, better=lambda u, v: u > v,
                         box=True, norm="L2", x_init=xi)
        outs.append(xb)
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], outs[2])


@pytest.mark.parametrize("norm", ["Linf", "L2"])
def test_matches_tabularbench_reference(norm):
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_capgd_equiv as eq

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
    for p in net.parameters():
        p.requires_grad_(False)
    model = lambda xx: net(xx) * 3.0
    x = torch.rand(1, 12)
    y = torch.tensor([2])
    CAPGD = eq.load_reference()
    tr, xb_r, lb_r = eq.run_reference(CAPGD, model, x.clone(), y, eps=0.475, steps=25, rho=0.75, device="cpu", norm=norm)
    to, xb_o, lb_o = eq.run_ours(model, x.clone(), y, eps=0.475, steps=25, rho=0.75, device="cpu", norm=norm)
    assert len(tr) == len(to)
    assert max(float((a - b).abs().max()) for a, b in zip(tr, to)) < 1e-6
    assert lb_r == pytest.approx(lb_o, abs=1e-5)
