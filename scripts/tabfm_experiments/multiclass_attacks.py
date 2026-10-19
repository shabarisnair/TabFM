"""Label-perturbation attacks for K-class contexts.

    label-flip-bernoulli  every row independently, with probability R, is relabelled to a
                          uniformly random OTHER class. Stochastic: the rows and the
                          targets are redrawn every run.
    hub-flip-local        the R% densest rows (hubs: smallest mean distance to their 10
                          nearest rows) are each relabelled to their LOCALLY CLOSEST other
                          class -- the class whose 10 nearest members lie closest on
                          average. Deterministic.
    hub-flip-random       the SAME hub rows, relabelled to a uniformly random other class.
                          Stochastic in the targets only.
    boundary-flip-knn     the R% rows whose labels are most AMBIGUOUS among their neighbours
                          -- the smallest margin between the own class and the best other
                          class in a distance-weighted vote of the 10 nearest rows -- each
                          relabelled to that other class. The new labels look plausible to a
                          model of the clean data, so a filter that keeps the labels a clean
                          model finds plausible misses many of them. Deterministic.

hub-flip-local and hub-flip-random poison identical rows, so the gap between them
isolates what the COHERENCE of the poison contributes: neighbouring hubs share a locally
closest class (0.83-0.96 agreement on the two multiclass datasets) and so reinforce each
other, while random targets scatter them (agreement at chance, 1/(K-1)).

"Closest" is deliberately LOCAL. The mean distance to ALL members of a class is a
global, centroid-like quantity: on gas-drift it sends 91% of hubs to the two classes that
happen to sit centrally in feature space, which quietly turns a hub attack into a
targeted one. Distance to a class's nearest members measures local confusability.

All three are test-agnostic: rows and targets depend on the context alone, and the
CLEAN labels are used to pick targets (the attacker knows them). At K=2 the only other
class is 1 - y, so each reduces exactly to the binary attack of the same name.
"""

from __future__ import annotations

import numpy as np
import torch

from .sampling import bernoulli_row_indices, k_from_percent, random_flip_rng

ATTACKS = ("label-flip-bernoulli", "hub-flip-local", "hub-flip-random", "boundary-flip-knn")
STOCHASTIC = ("label-flip-bernoulli", "hub-flip-random")


def random_other_class(y_rows: np.ndarray, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """For each label c, a class drawn uniformly from {0..K-1} minus {c}.

    Draw r uniformly from 0..K-2 and shift it past c: target = r + (r >= c). Every class
    other than c is hit by exactly one value of r, so the draw is exactly uniform.
    """
    y_rows = np.asarray(y_rows).astype(int)
    r = rng.integers(0, n_classes - 1, size=len(y_rows))
    return r + (r >= y_rows)


def standardise(X: torch.Tensor) -> torch.Tensor:
    """The same z-scoring hub_scores applies, so 'close' means the same thing in both."""
    Z = X.double()
    return (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-12)


def local_nearest_other_class(X: torch.Tensor, y: np.ndarray, rows: np.ndarray, n_classes: int,
                              k: int = 10, chunk: int = 512) -> np.ndarray:
    """For each row in ``rows``: the other class whose ``k`` nearest members are closest.

    Score of class c for row i = mean Euclidean distance (standardised features) from i
    to the k members of c nearest to i. The row's own class is excluded, as is the row
    itself. Ties break toward the lower class index, so the result is deterministic.
    """
    y = np.asarray(y).astype(int)
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    Z = standardise(X)
    yt = torch.as_tensor(y, device=Z.device)
    out = np.empty(len(rows), dtype=np.int64)
    for s in range(0, len(rows), chunk):
        r = torch.as_tensor(rows[s:s + chunk], device=Z.device)
        D = torch.cdist(Z[r], Z)                                   # [c, n]
        D[torch.arange(len(r), device=Z.device), r] = float("inf")  # never itself
        score = torch.full((len(r), n_classes), float("inf"), dtype=D.dtype, device=D.device)
        for c in range(n_classes):
            members = (yt == c).nonzero(as_tuple=True)[0]
            kk = min(k, len(members))
            if kk == 0:
                continue
            score[:, c] = D[:, members].topk(kk, largest=False).values.mean(1)
        score[torch.arange(len(r), device=Z.device), yt[r]] = float("inf")  # not its own class
        out[s:s + len(r)] = score.argmin(1).cpu().numpy()
    return out


def boundary_flip_knn(X: torch.Tensor, y: np.ndarray, n_classes: int, m: int, k: int = 10,
                      chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(rows, targets) of boundary-flip-knn: the m rows with the smallest distance-weighted kNN
    margin, each with the best other class as its target.

    Vote of row i for class c = sum of exp(-d_ij / d_med) over i's k nearest rows j of class c
    (standardised features, i itself excluded; d_med = the median of all n * k such distances,
    the lower middle value); margin = own-class vote minus the best other-class vote, the best
    other class being the first of the largest. Rows sorted by margin, stable.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    Z = standardise(X)
    yt = torch.as_tensor(y, device=Z.device)
    d_all = torch.empty(n, k, device=Z.device, dtype=Z.dtype)
    j_all = torch.empty(n, k, device=Z.device, dtype=torch.long)
    for st in range(0, n, chunk):
        D = torch.cdist(Z[st:st + chunk], Z)
        idx = torch.arange(D.shape[0], device=Z.device)
        D[idx, idx + st] = float("inf")  # never itself
        d_all[st:st + D.shape[0]], j_all[st:st + D.shape[0]] = D.topk(k, largest=False)
    w = torch.exp(-d_all / d_all.median().clamp_min(1e-12)).double()
    vote = torch.zeros(n, n_classes, device=Z.device, dtype=torch.float64).scatter_add_(1, yt[j_all], w)
    rows_all = torch.arange(n, device=Z.device)
    own = vote[rows_all, yt].clone()
    other = vote.clone()
    other[rows_all, yt] = -1.0
    best, target = other.max(1)
    rows = np.sort(np.argsort((own - best).cpu().numpy(), kind="stable")[:m])
    return rows, target.cpu().numpy()[rows]


def poison(attack: str, X: torch.Tensor, y_clean: np.ndarray, n_classes: int, row_percent: float, *,
           run_id: int, row_seed: int, hub_metric: str = "knn", hub_k: int = 10,
           target_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flipped_row_indices, y_poisoned)`` for one run of ``attack``.

    Every flipped row is guaranteed to change class; every other row keeps its label.
    """
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; expected one of {ATTACKS}")
    y = np.asarray(y_clean).astype(int)
    n = len(y)
    if row_percent <= 0:
        return np.zeros(0, dtype=np.int64), y.copy()

    if attack == "boundary-flip-knn":
        rows, targets = boundary_flip_knn(X, y, n_classes, k_from_percent(n, row_percent), k=target_k)
    elif attack == "label-flip-bernoulli":
        # Same stream and same row draw as the binary label-flip-bernoulli, so at K=2 the
        # rows are identical. The targets are drawn from the stream AFTER the rows.
        rng = random_flip_rng(row_seed, run_id, 0)
        rows = bernoulli_row_indices(y, row_percent / 100.0, attack_class=None, rng=rng)
        targets = random_other_class(y[rows], n_classes, rng)
    else:
        # Lazy: hub_flip_attention is a script in scripts/, which callers put on the path.
        from hub_flip_attention import hub_scores, pick_hubs
        rows = pick_hubs(hub_scores(X, hub_metric, hub_k), k_from_percent(n, row_percent))
        if attack == "hub-flip-local":
            targets = local_nearest_other_class(X, y, rows, n_classes, k=target_k)
        else:
            # Separate stream from the Bernoulli flip (subsample id 1).
            targets = random_other_class(y[rows], n_classes, random_flip_rng(row_seed, run_id, 1))

    rows = np.asarray(rows, dtype=np.int64)
    y_poisoned = y.copy()
    y_poisoned[rows] = targets
    if len(rows) and np.any(y_poisoned[rows] == y[rows]):
        raise AssertionError("a flipped row kept its own class")
    return rows, y_poisoned


def transition_counts(y_clean: np.ndarray, y_poisoned: np.ndarray, rows: np.ndarray,
                      n_classes: int) -> list[list[int]]:
    """K x K counts of clean -> poisoned class over the flipped rows."""
    t = np.zeros((n_classes, n_classes), dtype=np.int64)
    if len(rows):
        np.add.at(t, (np.asarray(y_clean)[rows], np.asarray(y_poisoned)[rows]), 1)
    return t.tolist()
