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
             in the context at weight 1. Ascent on  L(w) = mean_j (1 - w_j) CE_j  over all poisoned rows (full
             gradient; the queries go through in chunks only to fit in memory). Default (the sigmoid study's
             choice): w = sigmoid(t), t unbounded, started from the hard V-filter split (the k rows the V-filter
             finds least plausible at t = -30, the rest at t = 4), Adam at lr 5. param "w" is the earlier form:
             w in [0, 1] itself (start: all 1, or the soft V-filter weights) by PGD-L-inf steps, clipped (`mass`:
             projected onto sum(w) = (1 - R) n). `holdout`: "self" (default) masks each query from its own context
             copy, so it cannot read its label straight off it, the rows' only exclusion; "none" keeps even that;
             "chunk" holds each chunk of queries out of the context entirely.
  tabsift_m1 tabsift in M1's form (param "topk": tinyv_learned.ATTN_CONFIGS["topk"], attn_learn.py --param topk): a
             free score per poisoned row, started at the V-filter's log p (so the first split is the V-filter's); the
             split is hard, the k lowest-scored rows held out (w = 0, query weight 1) and the rest kept (w = 1, query
             weight 0). The gradient is taken at the relaxed point (held out at w = 0.05, kept at 0.95, query weights
             1 - w) and moves the scores like log-weights, dL/dlog w = (context part) - w_j CE_j / n, by PGD-L-inf
             ascent steps of lr 0.2 (s += lr g / max|g|), 100 steps. The scores are the deployed weights; the records
             also carry the objective at the hard split itself and the number of rows that have left the start's
             held-out set.
  tabsift_m1_rand_s<i>, tabsift_m1_rand_ens
             tabsift_m1 from random N(0, 1) scores (seed i = 0..4) instead of the V-filter's, 300 steps, each recorded
             every 5 steps; and, once the 5 have finished, their vote: each row's hard weight (1 kept, 0 held out)
             averaged over the 5 seeds' chosen checkpoints (ties: its mean score rank), deployed once.
  tabsift_adam_lr<x>, tabsift_sgd_lr<x>
             tabsift (the sigmoid form, hard V-filter start, 100 steps) by Adam at lr x = 0.01, 0.1, 0.5, 1 or 5 (the
             default tabsift uses Adam at 5), or by plain SGD (t += lr dL/dt, no momentum) at lr x = 1e3, 1e4, 5e4, 1e5
             or 5e5 (an optimizer x learning-rate sweep), ranked by t (rank "t", as tabsift_zero), t in float64 (in
             float32, SGD's steps on most held-out rows, about 1e-8, round away and leave a hundred or more rows tied
             at -30, the top-70% cut among them, broken by file order). SGD's rates are
             scaled up because the gradients are small (L is a mean over the n poisoned rows): at the start |dL/dt| is
             about 1e-5 for a typical kept row, so 1e3 to 5e5 move it by about 0.01 to 10 per step, the range of Adam's
             steps (about lr) at 0.01 to 5
  tabsift_zero
             tabsift (sigmoid, Adam lr 5, 100 steps) started with every poisoned row near w = 0: t = -30 (w about 1e-13)
             plus a N(0, 0.01) jitter (so that the step-0 split is random, not in file order), V at weight 1 as always.
             Every poisoned row starts as a query (query weight about 1): the whole poisoned set is the held-out side and
             the training side is V. The gradients flow at this start (median |dL/dt| about 6e-6, far above Adam's eps);
             -30 is the start closest to a V-only context (from -4 to -20 the predictions are as far from V-only as at
             w = 1). Its weights are recorded and deployed as t (rank "t"), the same order as w without ties from
             float32 underflow (the records' w_mean is then the mean t).
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
TABSIFT = dict(param="sigmoid", init="vf_hard", t_kept=4.0, t_removed=-30.0, opt="adam", lr=5.0, steps=100, mass=False,
               holdout="self", chunk=None, record_every=2, ckpts=(0, 1, 2, 5, 10, 20, 30, 50, 75, 100))
# the earlier form (w in [0, 1], PGD-L-inf at lr 0.05, start all 1), for reference: TABSIFT | TABSIFT_W
TABSIFT_W = dict(param="w", init="ones", init_temp=1.0, opt="pgd", lr=0.05)
# M1's form (tabsift_m1): TABSIFT | TABSIFT_M1; probe and lr are M1's (tinyv_learned.ATTN_CONFIGS["topk"])
TABSIFT_M1 = dict(param="topk", init="vfilter", probe=0.05, opt="pgd", lr=0.2)
SMALL = 0.02  # the small clean subset: this share of the poisoned rows
# the configurations each method is run at (bench_tinyv.run_sift): the default, and the grid V-selection picks from
GRIDS = {"metasift": {"tuned": {}, "model_lr3e-4": dict(model_lr=3e-4), "res_epochs10": dict(res_epochs=10),
                      "v_lr0.05": dict(v_lr=0.05)},
         "tabsift": {"sig_hard": {}},
         "tabsift_m1": {"m1": TABSIFT_M1}}
# the fixed configurations reported as rows (row name suffix: (configuration, checkpoint)); "" is the default
ROWS = {"metasift": {"": ("tuned", None)}, "tabsift": {"": ("sig_hard", 100)}, "tabsift_m1": {"": ("m1", 100)}}
# the sigmoid form started near w = 0 for every poisoned row (tabsift_zero): TABSIFT | TABSIFT_ZERO
TABSIFT_ZERO = dict(init="zeros", t_start=-30.0, t_jitter=0.01, rank="t")
GRIDS["tabsift_zero"] = {"sig_zero": TABSIFT_ZERO}
ROWS["tabsift_zero"] = {"": ("sig_zero", 100)}
# the sigmoid form's optimizer x learning-rate sweep (tabsift_<opt>_lr<x>): TABSIFT by that optimizer at lr x, ranked by t
SWEEP = {"adam": ("0.01", "0.1", "0.5", "1", "5"), "sgd": ("1e3", "1e4", "5e4", "1e5", "5e5")}
for _opt, _lrs in SWEEP.items():
    for _lr in _lrs:
        GRIDS[f"tabsift_{_opt}_lr{_lr}"] = {f"sig_hard_{_opt}_lr{_lr}": dict(opt=_opt, lr=float(_lr), rank="t", t_float64=True)}
        ROWS[f"tabsift_{_opt}_lr{_lr}"] = {"": (f"sig_hard_{_opt}_lr{_lr}", 100)}
# M1's form from a random start (tabsift_m1_rand_s<i>): N(0, 1) scores drawn with seed i (attn_learn's random topk start),
# 300 steps, recorded every 5; V-selection's candidates at the steps in ckpts
TABSIFT_M1_RAND = dict(TABSIFT_M1, init="random", init_std=1.0, steps=300, record_every=5,
                       ckpts=(0, 10, 20, 50, 100, 150, 200, 250, 300))
RAND_SEEDS = 5
for _i in range(RAND_SEEDS):
    GRIDS[f"tabsift_m1_rand_s{_i}"] = {f"rand_s{_i}": TABSIFT_M1_RAND | {"init_seed": _i}}
    ROWS[f"tabsift_m1_rand_s{_i}"] = {"": (f"rand_s{_i}", 300)}
# the vote of the 5 seeds after they finished (bench_tinyv.run_sift_ensemble), over each seed's chosen checkpoint: its
# last step (the default row) or its V-selected one (the _vsel rows, per deployment)
ENSEMBLES = {"tabsift_m1_rand_ens": [(f"tabsift_m1_rand_s{i}", f"rand_s{i}") for i in range(RAND_SEEDS)]}


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
    weight 1 (holdout: see the module docstring). Returns (L, dL/dlog w through the context or None, the rows' CE);
    the query channel's gradient is -CE_j / n in w (ascent_direction combines the two for a parameterisation)."""
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
    return total, torch.nan_to_num(g_ctx[:n], nan=0.0, posinf=0.0, neginf=0.0), ce


def ascent_direction(param: str, w: torch.Tensor, g_logw: torch.Tensor, ce: np.ndarray) -> torch.Tensor:
    """dL/dw (param "w": the context part dL/dlog w / w, none for a row at w = 0, as for the soft form), dL/dt for
    w = sigmoid(t) (dlog w/dt = 1 - w, dw/dt = w (1 - w)) or, for "topk" (M1's scores, which move like log-weights),
    dL/dlog w at the relaxed point; the query channel is -CE_j / n in w."""
    w = w.detach()
    q = torch.as_tensor(ce, dtype=torch.float32, device=w.device) / len(ce)
    if param == "w":
        return torch.where(w > 0, g_logw / w.clamp_min(1e-30), torch.zeros_like(w)) - q
    if param == "topk":
        return g_logw - q * w
    return g_logw * (1 - w) - q * w * (1 - w)


def held_out(scores: torch.Tensor, k: int) -> torch.Tensor:
    """M1's hard split: the k lowest-scored rows (ties as in the benchmark's deployment, which keeps the first n - k of
    np.argsort(-scores, kind="stable"))."""
    order = np.argsort(-scores.detach().cpu().numpy(), kind="stable")
    m = torch.zeros(len(order), dtype=torch.bool, device=scores.device)
    m[torch.as_tensor(order[len(order) - k:], device=scores.device)] = True
    return m


def hard_objective(clf, d: dict, held: torch.Tensor, chunk: int) -> float:
    """sift_objective at the hard split: the held-out rows out of the context (w = 0) and the only queries (weight 1),
    the kept rows and V in the context at weight 1; mean over all n poisoned rows."""
    from tabfm_experiments.row_weights import evaluate_weighted
    q = torch.nonzero(held).squeeze(1)
    X = torch.cat([d["Xc"], d["Xv"]])
    y = np.concatenate([d["y_given"], d["yv"]])
    lw = torch.cat([torch.where(held, float("-inf"), 0.0), torch.zeros(len(d["yv"]), device=held.device)])
    ce = evaluate_weighted(clf, X, y, d["Xc"][q], d["y_given"][q.cpu().numpy()], lw, need_grad=False, batch_size=chunk)[0]
    return ce * len(q) / d["n"]


def vfilter_scores(clf, d: dict) -> np.ndarray:
    """log p(given label | V as the context), per poisoned row."""
    from tabfm_experiments.row_weights import evaluate_weighted
    p = evaluate_weighted(clf, d["Xv"], d["yv"], d["Xc"], d["y_given"], torch.zeros(len(d["yv"]), device=d["Xc"].device),
                          need_grad=False)[2]
    return np.log(np.clip(p[np.arange(d["n"]), d["y_given"]], 1e-12, None))


def tabsift(clf, d: dict, cfg: dict | None = None, log=print, record=None) -> dict:
    """Ascent on sift_objective: the weights at every checkpoint in cfg['ckpts'] (V-selection's candidates), and
    with `record`, record(it, weights, objective) every record_every steps and at the last (param "topk": the scores
    are the weights, and the records add objective_hard and swapped)."""
    import attn_learn as A
    c = dict(TABSIFT) | (cfg or {})
    device = d["Xc"].device
    n, k = d["n"], d["k"]
    sig, topk = c["param"] == "sigmoid", c["param"] == "topk"
    mass = float(n - k) if c["mass"] and c["param"] == "w" else None
    if topk:  # M1: scores from the V-filter's log p, or random; the split and the relaxed point are set in the loop
        for key in ("t_kept", "t_removed", "mass"):  # the sigmoid and w forms' settings
            c.pop(key)
        if c["init"] == "vfilter":
            s = torch.as_tensor(vfilter_scores(clf, d), dtype=torch.float32, device=device)
        elif c["init"] == "random":
            s = torch.as_tensor(np.random.default_rng(c["init_seed"]).normal(0.0, c["init_std"], n), dtype=torch.float32,
                                device=device)
        else:
            raise ValueError(f"unknown start {c['init']!r} for the topk form")
        held0 = held_out(s, k)
    elif sig:  # t unbounded, w = sigmoid(t)
        # t_float64: t (the optimizer's state, and the ranking with rank "t") in float64, so that steps below float32's
        # resolution at |t| = 30 (about 2e-6: SGD's lr dL/dt for most held-out rows) still move t instead of leaving rows
        # tied at the start's value (the context's log w and the query weights are float32 either way)
        dtype = torch.float64 if c.get("t_float64") else torch.float32
        if c["init"] == "vf_hard":  # the V-filter's k least plausible rows at t_removed, the rest at t_kept
            t = torch.full((n,), c["t_kept"], dtype=dtype, device=device)
            t[torch.as_tensor(np.argsort(vfilter_scores(clf, d), kind="stable")[:k], device=device)] = c["t_removed"]
        elif c["init"] == "ones":
            t = torch.full((n,), c["t_kept"], dtype=dtype, device=device)
        elif c["init"] == "zeros":  # every poisoned row held out: t_start, jittered so ties break at random (float64:
            # in float32 a 0.01 jitter at -30 has too few distinct values, and colliding rows would stay tied)
            t = torch.as_tensor(c["t_start"] + np.random.default_rng(int(d["seed"])).normal(0.0, c["t_jitter"], n),
                                dtype=torch.float64, device=device)
        else:
            raise ValueError(f"unknown start {c['init']!r} for the sigmoid form")
        t.requires_grad_(True)
        # opt "adam" (the default), or "sgd": plain SGD, t += lr dL/dt (no momentum)
        optim = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[c["opt"]]([t], lr=c["lr"], maximize=True)
        w = torch.sigmoid(t.detach())
    elif c["init"] == "ones":
        w = torch.ones(n, device=device)
    else:  # the soft V-filter start: p(given label | V) / its max, sharpened by init_temp
        dd = {"n": n, "Xc": d["Xc"], "yp": d["y_given"], "Xf": d["Xv"], "yf": d["yv"]}
        w = A.init_raw({"param": "w", "init": "vfilter", "init_temp": c["init_temp"], "floor": -1000.0}, dd, clf, device)
    if c["param"] == "w":
        w = project_mass(w.float(), mass)
    chunk = c["chunk"] or (512 if c["holdout"] == "chunk" else n)
    ckpts, t0 = {}, time.time()
    for it in range(c["steps"] + 1):
        last = it == c["steps"]
        if topk:
            held = held_out(s, k)
            w = torch.where(held, c["probe"], 1.0 - c["probe"])  # the relaxed point
        while True:
            try:
                loss, g_logw, ce = sift_objective(clf, d, w, chunk=chunk, holdout=c["holdout"], need_grad=not last)
                break
            except torch.OutOfMemoryError:
                if c["chunk"] or chunk <= 256:
                    raise
                chunk = (chunk + 1) // 2
                torch.cuda.empty_cache()
                log(f"      tabsift: out of GPU memory, {chunk} query rows per refit from now on")
        c["chunk_used"] = chunk
        # the deployed / recorded weights; rank "t": t itself, the same order as w = sigmoid(t) without float32 underflow
        # (w is exactly 0 below t of about -104, and log w exactly 0 above about +104, which would tie rows)
        weights = (s if topk else t.detach() if sig and c.get("rank") == "t" else w).detach().cpu().numpy().copy()
        if it in c["ckpts"] or last:
            ckpts[it] = {"w": weights, "objective": float(loss)}
        if record and (it % c["record_every"] == 0 or last):
            extra = {"objective_hard": hard_objective(clf, d, held, chunk), "swapped": int((held0 & ~held).sum())} if topk else {}
            record(it=it, weights=weights, objective=float(loss), **extra)
        if last:
            break
        g = ascent_direction(c["param"], w, g_logw, ce)
        if topk:
            s = s + c["lr"] * g / g.abs().max().clamp_min(1e-30)
        elif sig:
            t.grad = g
            optim.step()
            w = torch.sigmoid(t.detach())
        else:
            w = project_mass(w + c["lr"] * g / g.abs().max().clamp_min(1e-30), mass)
        if it % 10 == 0 and topk:
            log(f"      tabsift topk it {it}: objective (relaxed) {loss:.4f}  {int((held0 & ~held).sum())} of the {k} "
                f"held-out rows swapped since the start  ({time.time() - t0:.0f} s)")
        elif it % 10 == 0:
            log(f"      tabsift {c['param']} {c['init']} it {it}: objective {loss:.4f}  w mean {float(w.mean()):.3f} "
                f"below 0.5 {int((w < 0.5).sum())} of {n}  ({time.time() - t0:.0f} s)")
    return {"ckpts": ckpts, "config": c, "seconds": round(time.time() - t0, 1)}


def m1_vote(scores: list[np.ndarray], k: int) -> tuple[np.ndarray, float]:
    """The members' vote on their hard splits: each row's mean hard weight (1 kept, 0 held out: the k lowest of that
    member's scores), plus 0.1 x its mean score rank (0 = lowest, 1 = highest) to break ties, so higher = kept; and the
    share of rows on which every member agrees."""
    n = len(scores[0])
    hard, rank = [], []
    for s in scores:
        h = np.ones(n)
        h[np.argsort(-s, kind="stable")[n - k:]] = 0.0
        hard.append(h)
        rank.append(np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(n - 1, 1))
    vote = np.mean(hard, axis=0)
    return vote + 0.1 * np.mean(rank, axis=0), float(np.mean((vote == 0) | (vote == 1)))

