"""Sifting with the clean rows V fixed on the train side, for the tiny-V benchmark (bench_tinyv.py).

Both methods learn a weight w per poisoned row from the same idea (a soft split of the poisoned set: a row is
trained on with strength w and held out with strength 1 - w, and the split is the one under which the held-out
rows are predicted worst), with V always trained on and never held out. The benchmark deploys the weights
(bench_tinyv.run_sift) three ways, V added: the top (1 - R)% of the poisoned rows (the k lowest removed), the top
(1 - R)% class by class (each given class keeps its share of the poisoned set: Meta-Sift's selection rule,
get_sifter_result), and a small clean subset, the top 2% by that rule; and scores them with TabPFN on the test set.
While a method trains, `record` is called with its current weights every few iterations (the benchmark records
the deployments' test metrics and precision there).

  metasift   Meta-Sift (Zeng et al., USENIX Security '23; Meta-Sift/, imported unedited, with the MLP backbone,
             dilution and settings of run_meta_sift_baseline.py). Their training loop (train_sifter) is
             rewritten here with V in every batch: weight 1 in the virtual and the actual update, left out of the
             held-out loss that the weighting network is trained on. Their scoring (test_sifter) is rewritten
             too: it misplaces the last, smaller batch's weights. Batches of 128 poisoned rows plus all of V; the
             MLP by AdamW at lr 1e-3, the weighting network by SGD at lr 0.01; the virtual step also runs at 1e-3
             (their code loads the AdamW state into it, which replaces the nominal meta lr 0.1).
  tabsift    the same split with TabPFN: every poisoned row is in the context, row j weighted by w_j through the
             row-weight mechanism (tabfm_experiments/row_weights.py), and is a query, weighted by 1 - w_j; V is
             in the context at weight 1. w in [0, 1] (start: all 1, or the soft V-filter weights) is moved by
             PGD-L-inf ascent steps on  L(w) = mean_j (1 - w_j) CE_j  over all poisoned rows (full gradient; the
             queries go through in chunks only to fit in memory). `holdout`: "self" (default) masks each query from
             its own context copy, so it cannot read its label straight off it, the rows' only exclusion; "none"
             keeps even that; "chunk" holds each chunk of queries out of the context entirely. `mass`: also
             project onto sum(w) = (1 - R) n (default off).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

# the configuration of the earlier Meta-Sift baseline (results/meta_sift/*/*_tuned), V appended to every batch
METASIFT = dict(num_sifter=5, warmup_epochs=1, res_epochs=5, batch_size=128, v_lr=0.01, meta_lr=0.1, go_lr=0.1,
                top_k=6, num_act=4, momentum=0.9, nesterov=True, noise_sigma=0.1, model_lr=1e-3, record_every=10)
# chunk: query rows per refit of the context (None: all at once, halved on running out of GPU memory; the objective is
# the same for any chunk, except with holdout "chunk")
TABSIFT = dict(init="ones", init_temp=1.0, lr=0.05, steps=100, mass=False, holdout="self", chunk=None, record_every=2,
               ckpts=(0, 1, 2, 5, 10, 20, 30, 50, 75, 100))
SMALL = 0.02  # the small clean subset: this share of the poisoned rows
# the configurations each method is run at (bench_tinyv.run_sift): the default, and the grid V-selection picks from
GRIDS = {"metasift": {"tuned": {}, "model_lr3e-4": dict(model_lr=3e-4), "res_epochs10": dict(res_epochs=10),
                      "v_lr0.05": dict(v_lr=0.05)},
         "tabsift": {"ones": dict(init="ones"), "vfilter": dict(init="vfilter")}}
# the fixed configurations reported as rows (row name suffix: (configuration, checkpoint)); "" is the default
ROWS = {"metasift": {"": ("tuned", None)}, "tabsift": {"": ("ones", 100), "_vfinit": ("vfilter", 100)}}


# ---------------------------------------------------------------- selection

def class_proportional(weights: np.ndarray, y: np.ndarray, total: int) -> np.ndarray:
    """Meta-Sift's selection (get_sifter_result): per given class, its share of `total` (rounded up), highest
    weights first. Their code, called on a stand-in for the dataset."""
    msf, _ = _import_meta_sift()

    class _Args:
        num_classes = int(y.max()) + 1

    class _Set:
        targets = np.asarray(y)

        def __len__(self):
            return len(self.targets)

    if len(np.unique(y)) != _Args.num_classes:  # their loop indexes classes by position; keep it exact
        raise ValueError("a class is missing from the poisoned rows")
    return np.sort(msf.get_sifter_result(_Args, _Set(), np.asarray(weights)[None, :], total_pick=total).astype(np.int64))


# ---------------------------------------------------------------- Meta-Sift with V fixed

def _import_meta_sift():
    import run_meta_sift_baseline  # noqa: F401 - puts Meta-Sift/ on the path and stubs torchvision
    import meta_sift.meta_sift_func as msf
    from meta_sift import util as mu
    return msf, mu


def metasift(d: dict, cfg: dict | None = None, log=print, record=None) -> dict:
    """Mean weight per poisoned row over the sifters (Meta-Sift's v_res.mean(0)). With `record`, every
    record_every meta-iterations of each sifter (and after its warmup): record(sifter, it, weights, objective,
    train_loss) with that sifter's current weights (scored as in the identification stage, with its own noise
    stream so that recording does not change the run), the mean held-out loss it is trained to raise and the mean
    weighted training loss over those iterations."""
    from run_meta_sift_baseline import TabularMLP
    msf, mu = _import_meta_sift()
    c = dict(METASIFT) | (cfg or {})
    device = d["Xc"].device
    torch.cuda.set_device(device)  # their code places everything with .cuda()
    mu.set_seeds(int(d["seed"]))
    A = torch.cat([d["Xc"], d["Xv"]]).float()
    mu_, sd = A.mean(0), A.std(0, unbiased=False)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    Xp, Xv = (d["Xc"].float() - mu_) / sd, (d["Xv"].float() - mu_) / sd  # features only, no labels
    yp = torch.as_tensor(d["y_given"], dtype=torch.long, device=device)
    yv = torch.as_tensor(d["yv"], dtype=torch.long, device=device)
    n, nv, K = len(yp), len(yv), int(max(yp.max(), yv.max()) + 1)

    class Args:  # what their helpers read
        pass

    args = Args()
    for key, val in c.items():
        setattr(args, key, val)
    args.num_classes = K
    gen = torch.Generator(device="cpu").manual_seed(int(d["seed"]))
    batches = lambda: torch.randperm(n, generator=gen).split(c["batch_size"])  # noqa: E731
    vw = torch.ones(nv, device=device)

    def score(model, vnet, noise) -> np.ndarray:
        """Every pool row's weight under sample dilution (Gaussian noise on the standardised features), in order."""
        v = np.zeros(n, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for st in range(0, n, c["batch_size"]):
                x = Xp[st:st + c["batch_size"]]
                eps = torch.randn_like(x) if noise is None else torch.randn(x.shape, generator=noise, device=device)
                cost = F.cross_entropy(model(x + eps * c["noise_sigma"]), yp[st:st + c["batch_size"]], reduction="none")
                v[st:st + len(x)] = vnet(cost.view(-1, 1)).view(-1).cpu().numpy()
        model.train()
        return v

    rec_noise = torch.Generator(device=device).manual_seed(int(d["seed"]) + 7919) if record else None
    t0 = time.time()
    vnets, models = [], []
    for s in range(c["num_sifter"]):
        model = TabularMLP(K, n_features=Xp.shape[1]).cuda()
        opt_a = torch.optim.AdamW(model.parameters(), lr=c["model_lr"], betas=(0.9, 0.95), weight_decay=0.05)
        vnet = mu.nnVent(1, 100, 150, 1).cuda()
        opt_c = torch.optim.SGD(vnet.parameters(), c["v_lr"])
        grad_models, grad_opts = mu.build_grad_models(args, model)
        for _ in range(c["warmup_epochs"]):  # their warmup, V in every batch
            for b in batches():
                model.train()
                opt_a.zero_grad()
                F.cross_entropy(model(torch.cat([Xp[b.to(device)], Xv])), torch.cat([yp[b.to(device)], yv])).backward()
                opt_a.step()
        if record:
            record(sifter=s, it=0, weights=score(model, vnet, rec_noise), objective=None, train_loss=None)
        raw_meta = TabularMLP(K, n_features=Xp.shape[1]).cuda()
        it, held, train = 0, [], []
        for _ in range(c["res_epochs"]):
            for b in batches():
                b = b.to(device)
                x, y, nb = torch.cat([Xp[b], Xv]), torch.cat([yp[b], yv]), len(b)
                # virtual training: the pool rows weighted by the net, V at 1
                meta_model = copy.deepcopy(raw_meta)
                meta_model.load_state_dict(model.state_dict())
                cost = F.cross_entropy(meta_model(x), y, reduction="none")
                v_lambda = mu.norm_weight(torch.cat([vnet(cost[:nb].data.view(-1, 1)).view(-1), vw]))
                grads = torch.autograd.grad(torch.sum(v_lambda * cost), meta_model.parameters(), create_graph=True,
                                            allow_unused=True)
                new_grads, _ = mu.compute_gated_grad(grads, grad_models, c["top_k"], c["num_act"])
                pseudo = mu.MetaSGD(meta_model, meta_model.parameters(), lr=c["meta_lr"])
                pseudo.load_state_dict(opt_a.state_dict())
                pseudo.meta_step(new_grads)
                # the held-out side: the pool rows only, weighted 1 - net
                res_cost = F.cross_entropy(meta_model(Xp[b]), yp[b], reduction="none")
                res_v = mu.norm_weight(1 - vnet(res_cost.data.view(-1, 1)).view(-1))
                valid_loss = -torch.sum(res_v * res_cost)
                opt_c.zero_grad()
                for go in grad_opts:
                    go.zero_grad()
                valid_loss.backward()
                opt_c.step()
                for go in grad_opts:
                    go.step()
                del grads, new_grads
                # actual update: the pool rows weighted by the (fixed) net, V at 1
                cost_w = F.cross_entropy(model(x), y, reduction="none")
                with torch.no_grad():
                    w_new = mu.norm_weight(torch.cat([vnet(cost_w[:nb].view(-1, 1)).view(-1), vw]))
                opt_a.zero_grad()
                l_f = torch.sum(w_new * cost_w)
                l_f.backward()
                opt_a.step()
                it += 1
                held.append(-float(valid_loss))
                train.append(float(l_f))
                if record and it % c["record_every"] == 0:
                    record(sifter=s, it=it, weights=score(model, vnet, rec_noise), objective=float(np.mean(held)),
                           train_loss=float(np.mean(train)))
                    held, train = [], []
        vnets.append(copy.deepcopy(vnet))
        models.append(copy.deepcopy(model))
        log(f"      metasift sifter {s}: {time.time() - t0:.0f} s")
    # their identification stage: every pool row scored by each sifter under sample dilution
    v_res = np.stack([score(model, vnet, None) for vnet, model in zip(vnets, models)])
    return {"weights": v_res.mean(0), "v_res": v_res, "config": c, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- the TabPFN counterpart

def project_mass(w: torch.Tensor, mass: float | None) -> torch.Tensor:
    """Onto [0, 1]^n, and with `mass` onto sum(w) = mass as well (Euclidean: clip(w - tau, 0, 1), tau by
    bisection)."""
    if mass is None:
        return w.clamp(0.0, 1.0)
    lo, hi = float(w.min()) - 1.0, float(w.max())
    for _ in range(60):
        tau = 0.5 * (lo + hi)
        if float((w - tau).clamp(0.0, 1.0).sum()) > mass:
            lo = tau
        else:
            hi = tau
    return (w - 0.5 * (lo + hi)).clamp(0.0, 1.0)


def sift_objective(clf, d: dict, w: torch.Tensor, *, chunk: int, holdout: str = "self", need_grad: bool = True):
    """L = mean over the poisoned rows j of (1 - w_j) CE_j, every poisoned row in the context at weight w_j and V at
    weight 1 (holdout: see the module docstring). Returns (L, dL/dw or None, the rows' CE). The context channel's
    gradient comes through log w (dL/dw = dL/dlog w / w; a row at w = 0 gets none, as for the soft form); the query
    channel's is -CE_j / n."""
    from tabfm_experiments.row_weights import evaluate_weighted, log_weights
    n, nv = d["n"], len(d["yv"])
    rows = np.arange(n)
    X = torch.cat([d["Xc"], d["Xv"]])
    y = np.concatenate([d["y_given"], d["yv"]])
    base = torch.cat([log_weights(w.detach()), torch.zeros(nv, device=w.device)])
    total, g_ctx, ce = 0.0, torch.zeros(n + nv, device=w.device), np.zeros(n)
    # "self" / "none": one context for all queries (the chunks are exact); "chunk": each chunk's own context
    spans = [(0, n)] if holdout != "chunk" else [(st, min(st + chunk, n)) for st in range(0, n, chunk)]
    for st, en in spans:
        q = rows[st:en]
        q_t = torch.as_tensor(q, dtype=torch.long, device=w.device)
        lw = base.clone()
        if holdout == "chunk":
            lw[q_t] = float("-inf")  # the chunk is out of the context
        lw.requires_grad_(need_grad)
        loss, g_lw, p = evaluate_weighted(clf, X, y, d["Xc"][q_t], d["y_given"][q], lw, need_grad=need_grad,
                                          batch_size=chunk, query_ctx_rows=q if holdout == "self" else None,
                                          query_weights=1.0 - w.detach()[q_t])
        total += loss * len(q) / n
        if need_grad:
            g_ctx += g_lw * len(q) / n
        ce[st:en] = -np.log(np.clip(p[np.arange(len(q)), d["y_given"][q]], 1e-12, None))
    if not need_grad:
        return total, None, ce
    g = torch.zeros(n, device=w.device)
    pos = w.detach() > 0
    g[pos] = g_ctx[:n][pos] / w.detach()[pos]
    g -= torch.as_tensor(ce, dtype=torch.float32, device=w.device) / n
    return total, torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), ce


def tabsift(clf, d: dict, cfg: dict | None = None, log=print, record=None) -> dict:
    """Ascent on sift_objective: the weights at every checkpoint in cfg['ckpts'] (V-selection's candidates), and
    with `record`, record(it, weights, objective) every record_every steps and at the last."""
    import attn_learn as A
    c = dict(TABSIFT) | (cfg or {})
    device = d["Xc"].device
    n, k = d["n"], d["k"]
    mass = float(n - k) if c["mass"] else None
    if c["init"] == "ones":
        w = torch.ones(n, device=device)
    else:  # the soft V-filter start: p(given label | V) / its max, sharpened by init_temp
        dd = {"n": n, "Xc": d["Xc"], "yp": d["y_given"], "Xf": d["Xv"], "yf": d["yv"]}
        w = A.init_raw({"param": "w", "init": "vfilter", "init_temp": c["init_temp"], "floor": -1000.0}, dd, clf, device)
    w = project_mass(w.float(), mass)
    chunk = c["chunk"] or (512 if c["holdout"] == "chunk" else n)
    ckpts, t0 = {}, time.time()
    for it in range(c["steps"] + 1):
        last = it == c["steps"]
        while True:
            try:
                loss, g, _ = sift_objective(clf, d, w, chunk=chunk, holdout=c["holdout"], need_grad=not last)
                break
            except torch.OutOfMemoryError:
                if c["chunk"] or chunk <= 256:
                    raise
                chunk = (chunk + 1) // 2
                torch.cuda.empty_cache()
                log(f"      tabsift: out of GPU memory, {chunk} query rows per refit from now on")
        c["chunk_used"] = chunk
        if it in c["ckpts"] or last:
            ckpts[it] = {"w": w.detach().cpu().numpy().copy(), "objective": float(loss)}
        if record and (it % c["record_every"] == 0 or last):
            record(it=it, weights=w.detach().cpu().numpy().copy(), objective=float(loss))
        if last:
            break
        w = project_mass(w + c["lr"] * g / g.abs().max().clamp_min(1e-30), mass)
        if it % 10 == 0:
            log(f"      tabsift {c['init']} it {it}: objective {loss:.4f}  w mean {float(w.mean()):.3f} at 0 "
                f"{int((w == 0).sum())} at 1 {int((w == 1).sum())}  ({time.time() - t0:.0f} s)")
    return {"ckpts": ckpts, "config": c, "seconds": round(time.time() - t0, 1)}
