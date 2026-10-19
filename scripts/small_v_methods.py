#!/usr/bin/env python
"""Baselines for cleaning a label-poisoned set with a tiny clean set V (one or a few rows per class).

Every method gets the same inputs (a Draw): the poisoned rows Xp with their given labels yp, the
clean rows Xv with their true labels yv (features transformed over poisoned + clean rows: z-scores
or normal quantiles, the `feat` axis of the grids), the number of classes K and the number of rows
to remove k (= R% of the poisoned rows; known to every method, as to ours). Every method returns a
Result:
  score     suspicion of every poisoned row (higher = more likely poisoned); removing the k most
            suspicious rows is the method's removal filter
  relabel   (relabelling methods) the label the method assigns to every poisoned row
  guess     (relabelling methods) its best guess of every row's true label, for relabelling only
            the k most suspicious rows
  predict   (training methods) the method's own classifier: transformed X -> class probabilities
  flagged   (cleanlab) the method's own issue set

  knn_sanitize   Paudice et al. 2018, Alg. 2: a row is relabelled to the majority label of its k
                 nearest neighbours (among the poisoned rows + V, itself excluded) when that label's
                 share is >= eta, optionally repeated; score = share of the neighbours disagreeing
                 with the given label
  label_spread   label spreading (Zhou et al. 2004), closed form F = (1 - alpha)(I - alpha S)^-1 Y0, on
                 a symmetric k-NN graph (Gaussian weights) over V + the poisoned rows, seeded with V's
                 labels (weight 1) and the given labels (weight beta; beta = 0: from V alone); score =
                 1 - the spread distribution's mass on the given label
  knn_shapley    exact KNN-Shapley (Jia et al. 2019) of every poisoned row for a K-NN classifier whose
                 utility is its likelihood of the right label on the rows of V; score = -value
  data_oob       Data-OOB (Kwon & Zou 2023): bagged decision trees on the poisoned rows + V (anchor_v: V
                 in every bag); value = out-of-bag accuracy on the row's given label; score = -value
  cleanlab       confident learning (Northcutt et al. 2021; cleanlab 2.x): out-of-fold probabilities of a
                 classifier (TabPFN, histogram gradient boosting, logistic regression) trained on the
                 other folds' poisoned rows + V; score = -label quality; flagged = cleanlab's own issues
  simifeat       SimiFeat (Zhu et al. 2022), training-free k-NN label agreement (cosine or euclidean),
                 optionally over 21 random 80% feature subsets as views; rank: score = 1 - agreement,
                 vote: score = share of views whose neighbour majority disagrees with the given label.
                 The removal count is the known k instead of SimiFeat's HOC noise-rate estimate
  glc            Gold Loss Correction (Hendrycks et al. 2018): an MLP on the poisoned rows gives the
                 corruption matrix C from V; a second MLP is trained with the forward-corrected loss on
                 the poisoned rows and CE on V (weighted); score = 1 - posterior of the given label
  mwnet          Meta-Weight-Net (Shu et al. 2019): an MLP whose per-row loss weights come from a small
                 net of the loss, meta-trained on V through one virtual SGD step; score = -final weight
  emlc           Enhanced Meta Label Correction (Taraday & Baskin 2023), tabular version: a teacher MLP
                 with a gate on (features, label embedding) gives corrected labels
                 q = w * onehot(given) + (1 - w) * p_teacher; a student MLP learns from q; the teacher is
                 trained by the student's one-step-lookahead CE on V (exact meta-gradient), CE on V and
                 a BCE gate loss on V with half its labels moved to the teacher's best wrong class;
                 score = 1 - q(given label)
  ieg            Distilling Effective Supervision (Zhang et al. 2020), tabular version: L2RW meta
                 weights and meta re-labelling (rectified sign of the meta gradient) from V, pseudo labels
                 averaged over a Gaussian-noise view and sharpened, KL consistency, mixup with V; score =
                 -mean meta weight
GRIDS holds every method's hyperparameter grid (`feat` is applied by the caller).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Draw:
    Xp: torch.Tensor          # poisoned rows, transformed [n, d]
    yp: np.ndarray            # given labels [n]
    Xv: torch.Tensor          # clean rows, transformed [m, d]
    yv: np.ndarray            # true labels [m]
    K: int
    k: int                    # rows to remove
    seed: int
    tabpfn_probs: Callable | None = None   # (train positions, query positions) -> TabPFN probabilities
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.yp)

    @property
    def device(self):
        return self.Xp.device


@dataclass
class Result:
    score: np.ndarray
    relabel: np.ndarray | None = None
    guess: np.ndarray | None = None
    predict: Callable | None = None
    flagged: np.ndarray | None = None


def _onehot(y: np.ndarray, K: int, device) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(np.asarray(y), device=device).long(), K).float()


def _knn(Zq: torch.Tensor, Zr: torch.Tensor, k: int, exclude_self: bool, metric: str = "euclidean") -> tuple[torch.Tensor, torch.Tensor]:
    """(indices, distances) [nq, k] of the k nearest rows of Zr for every row of Zq (row i of Zq is
    row i of Zr when exclude_self)."""
    idx, dist = [], []
    for st in range(0, len(Zq), 2048):
        q = Zq[st:st + 2048]
        if metric == "cosine":
            d = 1.0 - F.normalize(q, dim=1) @ F.normalize(Zr, dim=1).T
        else:
            d = torch.cdist(q, Zr)
        if exclude_self:
            d[torch.arange(len(q), device=q.device), torch.arange(st, st + len(q), device=q.device)] = float("inf")
        v, i = torch.topk(d, k, dim=1, largest=False)
        idx.append(i)
        dist.append(v)
    return torch.cat(idx), torch.cat(dist)


# ---------------------------------------------------------------- training-free methods

def knn_sanitize(d: Draw, k_nn: int = 10, eta: float = 0.5, rounds: int = 1) -> Result:
    Z = torch.cat([d.Xp, d.Xv])
    labels = np.concatenate([d.yp, d.yv]).astype(int)
    nn_idx = _knn(Z[:d.n], Z, k_nn, exclude_self=True)[0].cpu().numpy()
    counts = lambda lab: np.stack([(lab[nn_idx] == c).sum(1) for c in range(d.K)], 1)  # noqa: E731
    for _ in range(rounds):
        cnt = counts(labels)
        new = labels.copy()
        flip = cnt.max(1) / k_nn >= eta
        new[:d.n][flip] = cnt.argmax(1)[flip]
        if np.array_equal(new, labels):
            break
        labels = new
    cnt = counts(labels)
    return Result(score=1.0 - cnt[np.arange(d.n), d.yp] / k_nn, relabel=labels[:d.n].copy(), guess=cnt.argmax(1))


def label_spread(d: Draw, n_neighbors: int = 10, alpha: float = 0.9, beta: float = 0.0) -> Result:
    Z = torch.cat([d.Xp, d.Xv]).double()
    N = len(Z)
    idx, dist = _knn(Z, Z, n_neighbors, exclude_self=True)
    sigma = dist.median().clamp_min(1e-12)
    W = torch.zeros(N, N, device=Z.device, dtype=Z.dtype)
    W.scatter_(1, idx, torch.exp(-(dist / sigma) ** 2))
    W = torch.maximum(W, W.T)
    dis = W.sum(1).clamp_min(1e-300).rsqrt()
    S = dis[:, None] * W * dis[None, :]
    Y0 = torch.cat([beta * _onehot(d.yp, d.K, Z.device), _onehot(d.yv, d.K, Z.device)]).double()
    Fm = (1 - alpha) * torch.linalg.solve(torch.eye(N, device=Z.device, dtype=Z.dtype) - alpha * S, Y0)
    P = Fm[:d.n].clamp_min(0)
    s = P.sum(1, keepdim=True)
    P = torch.where(s > 1e-300, P / s.clamp_min(1e-300), torch.full_like(P, 1.0 / d.K)).cpu().numpy()
    return Result(score=1.0 - P[np.arange(d.n), d.yp], relabel=P.argmax(1), guess=P.argmax(1))


def knn_shapley(d: Draw, K_nn: int = 5) -> Result:
    """Exact KNN-Shapley (Jia et al. 2019, Thm. 1), averaged over the rows of V."""
    D = torch.cdist(d.Xv, d.Xp)                                    # [m, n]
    order = torch.argsort(D, dim=1, stable=True)                   # nearest first
    yp = torch.as_tensor(d.yp, device=D.device)
    match = (yp[order] == torch.as_tensor(d.yv, device=D.device)[:, None]).double()   # [m, n] by rank
    n = d.n
    rank = torch.arange(1, n + 1, device=D.device, dtype=torch.float64)                 # i = 1..n
    c = torch.zeros_like(match)
    c[:, :-1] = (match[:, :-1] - match[:, 1:]) / K_nn * torch.clamp(rank[:-1], max=K_nn) / rank[:-1]
    s_by_rank = match[:, -1:] / n + torch.flip(torch.cumsum(torch.flip(c, [1]), 1), [1])   # s_i = s_n + sum_{t>=i} c_t
    values = torch.zeros_like(match).scatter_(1, order, s_by_rank).mean(0).cpu().numpy()
    return Result(score=-values)


def data_oob(d: Draw, n_trees: int = 400, max_depth: int | None = None, min_samples_leaf: int = 1,
             max_features: str | float = "sqrt", anchor_v: bool = False, n_jobs: int = 16) -> Result:
    from joblib import Parallel, delayed
    from sklearn.tree import DecisionTreeClassifier
    X = torch.cat([d.Xp, d.Xv]).cpu().numpy()
    y = np.concatenate([d.yp, d.yv]).astype(int)
    n, N = d.n, len(y)
    seeds = np.random.default_rng(d.seed).integers(0, 2**31 - 1, size=n_trees)

    def one(seed):
        r = np.random.default_rng(seed)
        bag = np.concatenate([r.integers(0, n, n), np.arange(n, N)]) if anchor_v else r.integers(0, N, N)
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                      random_state=int(seed)).fit(X[bag], y[bag])
        oob = np.setdiff1d(np.arange(n), bag)
        return oob, tree.predict(X[oob]) == y[oob]

    hits, counts = np.zeros(n), np.zeros(n)
    for oob, ok in Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in seeds):
        hits[oob] += ok
        counts[oob] += 1
    return Result(score=-np.where(counts > 0, hits / np.maximum(counts, 1), 0.5))


def _oof_probs(d: Draw, model: str, folds: int = 5) -> np.ndarray:
    """Out-of-fold class probabilities of the poisoned rows, the model trained on the other folds + V."""
    n = d.n
    fold = np.random.default_rng(d.seed).permutation(n) % folds
    P = np.zeros((n, d.K))
    for f in range(folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        if model == "tabpfn":
            P[te] = d.tabpfn_probs(tr, te)
            continue
        Xtr = torch.cat([d.Xp[torch.as_tensor(tr, device=d.device)], d.Xv]).cpu().numpy()
        ytr = np.concatenate([d.yp[tr], d.yv]).astype(int)
        if model == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(Xtr, ytr)
        P[np.ix_(te, clf.classes_.astype(int))] = clf.predict_proba(d.Xp[torch.as_tensor(te, device=d.device)].cpu().numpy())
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def cleanlab_method(d: Draw, model: str = "tabpfn", method: str = "self_confidence") -> Result:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    cache = d.extra.setdefault("oof", {})
    if model not in cache:
        cache[model] = _oof_probs(d, model)
    P = cache[model]
    try:
        flagged = find_label_issues(d.yp, P, filter_by="prune_by_noise_rate")
    except Exception:  # noqa: BLE001 - cleanlab refuses some degenerate inputs; the ranking still stands
        flagged = None
    return Result(score=-get_label_quality_scores(d.yp, P, method=method), guess=P.argmax(1), flagged=flagged)


def simifeat(d: Draw, k_nn: int = 10, metric: str = "cosine", views: int = 1, variant: str = "rank") -> Result:
    Z = torch.cat([d.Xp, d.Xv])
    labels = torch.as_tensor(np.concatenate([d.yp, d.yv]).astype(int), device=Z.device)
    yp = labels[:d.n]
    rng = np.random.default_rng(d.seed)
    agree = torch.zeros(d.n, device=Z.device, dtype=torch.float64)
    disagree = torch.zeros(d.n, device=Z.device, dtype=torch.float64)
    for _ in range(views):
        cols = torch.arange(Z.shape[1], device=Z.device)
        if views > 1:
            cols = torch.as_tensor(np.sort(rng.choice(Z.shape[1], max(1, int(0.8 * Z.shape[1])), replace=False)), device=Z.device)
        idx = _knn(Z[:d.n][:, cols], Z[:, cols], k_nn, exclude_self=True, metric=metric)[0]
        nl = labels[idx]
        counts = torch.stack([(nl == c).sum(1) for c in range(d.K)], 1)
        agree += (nl == yp[:, None]).double().mean(1)
        disagree += (counts.argmax(1) != yp).double()
    agree, disagree = (agree / views).cpu().numpy(), (disagree / views).cpu().numpy()
    return Result(score=(1.0 - agree) if variant == "rank" else disagree + 1e-3 * (1.0 - agree))


# ---------------------------------------------------------------- neural (training) methods

def make_mlp(d_in: int, d_out: int, width: int = 256, depth: int = 2) -> nn.Module:
    layers, h = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(h, width), nn.ReLU()]
        h = width
    layers.append(nn.Linear(h, d_out))
    return nn.Sequential(*layers)


def _batches(n: int, batch: int, epochs: int, rng: np.random.Generator):
    for _ in range(epochs):
        perm = rng.permutation(n)
        for st in range(0, n, batch):
            yield perm[st:st + batch]


def _soft_ce(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, 1)).sum(1)


def _predictor(model: nn.Module) -> Callable:
    def predict(X: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(X), 1).cpu().numpy()
    return predict


def _setup(d: Draw):
    torch.manual_seed(d.seed)
    dev = d.device
    return (np.random.default_rng(d.seed), torch.as_tensor(d.yp, device=dev).long(),
            torch.as_tensor(d.yv, device=dev).long())


def glc(d: Draw, lr: float = 1e-3, epochs: int = 40, trusted_weight: float = 1.0, batch: int = 128) -> Result:
    rng, yp_t, yv_t = _setup(d)
    dev, Kc = d.device, d.K
    f = make_mlp(d.Xp.shape[1], Kc).to(dev)          # step 1: p(given label | x) from the poisoned rows
    opt = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=1e-4)
    for b in _batches(d.n, batch, epochs, rng):
        b = torch.as_tensor(b, device=dev)
        opt.zero_grad()
        F.cross_entropy(f(d.Xp[b]), yp_t[b]).backward()
        opt.step()
    f.eval()
    with torch.no_grad():
        pv = torch.softmax(f(d.Xv), 1)
    C = torch.eye(Kc, device=dev)                     # C[true, given]
    for c in range(Kc):
        if (yv_t == c).any():
            C[c] = pv[yv_t == c].mean(0)
    g = make_mlp(d.Xp.shape[1], Kc).to(dev)          # step 2: forward-corrected loss on the poisoned rows + CE on V
    opt = torch.optim.Adam(g.parameters(), lr=lr, weight_decay=1e-4)
    m = len(d.yv)
    for b in _batches(d.n, batch, epochs, rng):
        b = torch.as_tensor(b, device=dev)
        loss_u = F.nll_loss(torch.log((torch.softmax(g(d.Xp[b]), 1) @ C).clamp_min(1e-12)), yp_t[b])
        loss_t = F.cross_entropy(g(d.Xv), yv_t)
        loss = (len(b) * loss_u + trusted_weight * m * loss_t) / (len(b) + trusted_weight * m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    g.eval()
    with torch.no_grad():
        p = torch.softmax(g(d.Xp), 1)
        joint = p * C[:, yp_t].T                      # p(true = y | x) * C[y, given]
        post = (joint / joint.sum(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    return Result(score=1.0 - post[np.arange(d.n), d.yp], relabel=post.argmax(1), guess=p.argmax(1).cpu().numpy(),
                  predict=_predictor(g))


def mwnet(d: Draw, lr: float = 0.05, meta_lr: float = 1e-3, epochs: int = 40, batch: int = 128) -> Result:
    from torch.func import functional_call
    rng, yp_t, yv_t = _setup(d)
    dev = d.device
    f = make_mlp(d.Xp.shape[1], d.K).to(dev)
    vnet = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()).to(dev)
    opt_f = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=meta_lr, weight_decay=1e-4)
    for b in _batches(d.n, batch, epochs, rng):
        b = torch.as_tensor(b, device=dev)
        xb, yb = d.Xp[b], yp_t[b]
        params = dict(f.named_parameters())       # meta step: one virtual SGD step, scored on V
        losses = F.cross_entropy(functional_call(f, params, (xb,)), yb, reduction="none")
        w = vnet(losses.detach()[:, None]).squeeze(1)
        grads = torch.autograd.grad((w * losses).sum() / len(b), list(params.values()), create_graph=True)
        fast = {name: p - lr * g_ for (name, p), g_ in zip(params.items(), grads)}
        opt_v.zero_grad()
        F.cross_entropy(functional_call(f, fast, (d.Xv,)), yv_t).backward()
        opt_v.step()
        losses = F.cross_entropy(f(xb), yb, reduction="none")
        with torch.no_grad():
            w = vnet(losses.detach()[:, None]).squeeze(1)
            w = w / w.sum() if w.sum() > 0 else torch.full_like(w, 1.0 / len(w))
        opt_f.zero_grad()
        (w * losses).sum().backward()
        opt_f.step()
    f.eval()
    with torch.no_grad():
        w_all = vnet(F.cross_entropy(f(d.Xp), yp_t, reduction="none")[:, None]).squeeze(1).cpu().numpy()
    return Result(score=-w_all, predict=_predictor(f))


class _Teacher(nn.Module):
    def __init__(self, d_in: int, K: int, width: int = 256, emb: int = 32):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.cls = nn.Linear(width, K)
        self.emb = nn.Embedding(K, emb)
        self.gate = nn.Sequential(nn.Linear(width + emb, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.K = K

    def forward(self, x, y):
        h = self.feat(x)
        logits = self.cls(h)
        w = self.gate(torch.cat([h, self.emb(y)], 1))
        return w * F.one_hot(y, self.K).float() + (1 - w) * torch.softmax(logits, 1), w.squeeze(1), logits


def emlc(d: Draw, lr: float = 0.02, teacher_lr: float = 1e-3, epochs: int = 40, batch: int = 128) -> Result:
    from torch.func import functional_call
    rng, yp_t, yv_t = _setup(d)
    dev = d.device
    student = make_mlp(d.Xp.shape[1], d.K).to(dev)
    teacher = _Teacher(d.Xp.shape[1], d.K).to(dev)
    opt_s = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=teacher_lr)
    m = len(d.yv)
    for b in _batches(d.n, batch, epochs, rng):
        b = torch.as_tensor(b, device=dev)
        xb, yb = d.Xp[b], yp_t[b]
        q, _, _ = teacher(xb, yb)                  # teacher: meta loss through one virtual student step,
        params = dict(student.named_parameters())  # CE on V, BCE gate on V with half its labels corrupted
        grads = torch.autograd.grad(_soft_ce(q, functional_call(student, params, (xb,))).mean(), list(params.values()),
                                    create_graph=True)
        fast = {name: p - lr * g_ for (name, p), g_ in zip(params.items(), grads)}
        l_meta = F.cross_entropy(functional_call(student, fast, (d.Xv,)), yv_t)
        _, w_clean, logits_v = teacher(d.Xv, yv_t)
        half = torch.as_tensor(rng.permutation(m)[:max(1, m // 2)], device=dev)
        with torch.no_grad():                      # corrupted to the teacher's strongest wrong class
            lv = logits_v[half].clone()
            lv[torch.arange(len(half), device=dev), yv_t[half]] = -float("inf")
            y_bad = lv.argmax(1)
        _, w_bad, _ = teacher(d.Xv[half], y_bad)
        l_bce = F.binary_cross_entropy(w_clean, torch.ones_like(w_clean)) + F.binary_cross_entropy(w_bad, torch.zeros_like(w_bad))
        opt_t.zero_grad()
        (l_meta + F.cross_entropy(logits_v, yv_t) + l_bce).backward()
        opt_t.step()
        with torch.no_grad():                      # student: one step on the corrected labels
            q, _, _ = teacher(xb, yb)
        opt_s.zero_grad()
        _soft_ce(q, student(xb)).mean().backward()
        opt_s.step()
    teacher.eval()
    with torch.no_grad():
        q, _, logits = teacher(d.Xp, yp_t)
    q = q.cpu().numpy()
    return Result(score=1.0 - q[np.arange(d.n), d.yp], relabel=q.argmax(1), guess=logits.argmax(1).cpu().numpy(),
                  predict=_predictor(student))


def ieg(d: Draw, lr: float = 1e-3, kl_weight: float = 1.0, p_weight: float = 5.0, sigma: float = 0.1,
        epochs: int = 40, batch: int = 128, tau: float = 0.5, lam0: float = 0.9) -> Result:
    from torch.func import functional_call
    rng, yp_t, yv_t = _setup(d)
    dev, Kc = d.device, d.K
    f = make_mlp(d.Xp.shape[1], Kc).to(dev)
    opt = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=1e-4)
    aug = lambda x: x + sigma * torch.randn_like(x)  # noqa: E731
    w_sum, w_cnt = torch.zeros(d.n, device=dev), torch.zeros(d.n, device=dev)
    y_star = yp_t.clone()
    beta = torch.distributions.Beta(torch.tensor(0.5, device=dev), torch.tensor(0.5, device=dev))
    oh_v = F.one_hot(yv_t, Kc).float()
    mv = len(d.yv)
    for b in _batches(d.n, batch, epochs, rng):
        b = torch.as_tensor(b, device=dev)
        xb, yb = d.Xp[b], yp_t[b]
        B = len(b)
        oh = F.one_hot(yb, Kc).float()
        with torch.no_grad():                      # pseudo labels: the row and a noisy view averaged, sharpened
            pr = 0.5 * (torch.softmax(f(xb), 1) + torch.softmax(f(aug(xb)), 1))
            g = pr ** (1 / tau)
            g = g / g.sum(1, keepdim=True)
        omega = torch.full((B,), 1.0 / B, device=dev, requires_grad=True)   # meta step on V
        lam = torch.full((B,), lam0, device=dev, requires_grad=True)
        params = dict(f.named_parameters())
        target = lam[:, None] * oh + (1 - lam[:, None]) * g
        grads = torch.autograd.grad((omega * _soft_ce(target, functional_call(f, params, (xb,)))).sum(),
                                    list(params.values()), create_graph=True)
        fast = {name: p - lr * g_ for (name, p), g_ in zip(params.items(), grads)}
        g_omega, g_lam = torch.autograd.grad(F.cross_entropy(functional_call(f, fast, (d.Xv,)), yv_t), [omega, lam])
        w_star = torch.clamp(-g_omega, min=0)
        w_star = w_star / w_star.sum() if w_star.sum() > 0 else torch.full_like(w_star, 1.0 / B)
        keep = (-g_lam > 0).float()[:, None]       # rectified sign: keep the given label, else the pseudo label
        tgt_star = keep * oh + (1 - keep) * g
        w_sum[b] += B * w_star
        w_cnt[b] += 1
        y_star[b] = tgt_star.argmax(1)
        logits = f(xb)
        l_w = (w_star * _soft_ce(lam0 * oh + (1 - lam0) * g, logits)).sum()
        l_l = _soft_ce(tgt_star, logits).mean()
        l_kl = F.kl_div(F.log_softmax(f(aug(xb)), 1), torch.softmax(logits, 1).detach(), reduction="batchmean")
        bp = beta.sample((mv,))[:, None]           # mixup of V with random batch rows
        j = b[torch.as_tensor(rng.integers(0, B, mv), device=dev)]
        l_mp = _soft_ce(bp * oh_v + (1 - bp) * F.one_hot(y_star[j], Kc).float(), f(bp * d.Xv + (1 - bp) * d.Xp[j])).mean()
        bu = beta.sample((B,))[:, None]            # mixup within the batch
        perm = torch.as_tensor(rng.permutation(B), device=dev)
        l_mu = _soft_ce(bu * tgt_star + (1 - bu) * tgt_star[perm], f(bu * xb + (1 - bu) * xb[perm])).mean()
        opt.zero_grad()
        (l_w + l_l + l_mp + p_weight * l_mu + kl_weight * l_kl).backward()
        opt.step()
    f.eval()
    with torch.no_grad():
        guess = f(d.Xp).argmax(1).cpu().numpy()
    return Result(score=-(w_sum / w_cnt.clamp_min(1)).cpu().numpy(), relabel=y_star.cpu().numpy(), guess=guess,
                  predict=_predictor(f))


METHODS = {"knn_sanitize": knn_sanitize, "label_spread": label_spread, "knn_shapley": knn_shapley, "data_oob": data_oob,
           "cleanlab": cleanlab_method, "simifeat": simifeat, "glc": glc, "mwnet": mwnet, "emlc": emlc, "ieg": ieg}
NEURAL = ("glc", "mwnet", "emlc", "ieg")


def _grid(**axes) -> list[dict]:
    out = [{}]
    for key, values in axes.items():
        out = [dict(o, **{key: v}) for o in out for v in values]
    return out


FEATS = ["z", "q"]
GRIDS = {
    "knn_sanitize": _grid(feat=FEATS, k_nn=[3, 5, 10, 20, 50], eta=[0.5, 0.6, 0.7, 0.8], rounds=[1, 3]),
    "label_spread": _grid(feat=FEATS, n_neighbors=[5, 10, 20, 50], alpha=[0.2, 0.5, 0.8, 0.95, 0.99], beta=[0.0, 0.1, 0.5]),
    "knn_shapley": _grid(feat=FEATS, K_nn=[1, 3, 5, 10, 20, 50]),
    "data_oob": _grid(max_depth=[None, 10], min_samples_leaf=[1, 5], max_features=["sqrt", 0.5], anchor_v=[False, True]),
    "cleanlab": _grid(model=["tabpfn", "hgb", "logreg"], method=["self_confidence", "normalized_margin", "confidence_weighted_entropy"]),
    "simifeat": _grid(feat=FEATS, k_nn=[5, 10, 20, 50], metric=["cosine", "euclidean"], views=[1, 21], variant=["rank", "vote"]),
    "glc": _grid(feat=FEATS, lr=[1e-3, 3e-3], epochs=[10, 40], trusted_weight=[1.0, 10.0]),
    "mwnet": _grid(feat=FEATS, lr=[0.01, 0.05], meta_lr=[1e-4, 1e-3], epochs=[10, 40]),
    "emlc": _grid(feat=FEATS, lr=[0.01, 0.05], teacher_lr=[1e-4, 1e-3], epochs=[10, 40]),
    "ieg": _grid(feat=FEATS, lr=[1e-3, 3e-3], kl_weight=[1.0, 10.0], epochs=[10, 40]),
}
