#!/usr/bin/env python
"""Meta-Sift (Zeng et al., USENIX'23) as a baseline on our tabular poisoning setting.

Meta-Sift solves the SAME bilevel objective this project's sift_context.py does -- split
the poisoned set so that a model trained on one half scores badly on the other -- but
differs in three ways that matter:

  * the per-sample weight is produced by a small WEIGHT-ASSIGNING NETWORK from the
    sample's loss, v_i = S(L_i), rather than being a free parameter per row. Far fewer
    parameters, and it generalises across rows with similar loss.
  * the inner problem RETRAINS a model on the weighted data (we hold TabPFN frozen and
    reweight its attention instead).
  * m independently seeded "sifters" are ensembled, each seeing a randomly perturbed
    copy of the data ("sample dilution"), and their weights are averaged.

NOTHING IN Meta-Sift/ IS EDITED. Their algorithm -- the meta-update, the gradient
samplers, the warmup, the ensembling, the selection rule and the NCR metric -- is
imported and used as written. The only substitution is the backbone: their ResNet18 is
an image model, so ``meta_sift_func.ResNet18`` is rebound to an MLP of matching
parameter-block structure. Their transforms are image ops (RandomCrop, RandomRotation),
so sample dilution becomes Gaussian noise on standardised features, which is the
tabular equivalent of the same idea.

Scored on exactly the setting used for our own method: url_unique, hub-flip, R=30%,
poison draw 0, so the flipped row set is bit-identical to the one sift_context.py saw.

    python scripts/run_meta_sift_baseline.py --gpu 3 --dataset url_unique \
        --attack hub-flip --row-percent 30
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "Meta-Sift"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# meta_sift_func.py does `import torchvision.transforms as transforms` at module level,
# but only uses it in get_dataset()/meta_sift(), which are the IMAGE entry points and are
# never called here. Installing torchvision for real is not an option: the wheel built
# against this torch (2.14) is not available, and the one that installs raises
# "operator torchvision::nms does not exist" on import. A stub satisfies the import
# without pulling in a mismatched binary, and nothing on the tabular path touches it.
if "torchvision" not in sys.modules:
    import types
    _tv = types.ModuleType("torchvision")
    _tv.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.transforms"] = _tv.transforms

from defend_context import default_splits, poison_rows  # noqa: E402
from tabfm_experiments.config import Seeds  # noqa: E402
from tabfm_experiments.io import setup_logger, write_json  # noqa: E402
from tabfm_experiments.sampling import k_from_percent  # noqa: E402
from tabfm_experiments.sift import select_clean_subset, subset_report  # noqa: E402


class TabularMLP(nn.Module):
    """Backbone for the sifter's inner model.

    Depth is chosen so that ``list(parameters())`` has enough blocks for Meta-Sift's
    gradient samplers, which attach to ``[-top_k:-2]``: four Linear layers give eight
    parameter tensors, so top_k=6 leaves exactly four gated blocks (their num_act).
    """

    def __init__(self, num_classes: int, n_features: int = 63, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, width // 2), nn.ReLU(inplace=True),
            nn.Linear(width // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TabularPoisonSet(torch.utils.data.Dataset):
    """Stands in for their h5 image dataset: ``.targets``, ``.transform``, (x, y) items.

    ``transform`` is how Meta-Sift injects its "randomized sample-dilution" at the
    identification stage. Theirs is RandomCrop + RandomRotation; the tabular counterpart
    is additive Gaussian noise on standardised features.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.data = torch.as_tensor(X, dtype=torch.float32)
        self.targets = np.asarray(y).astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = self.data[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[i])


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--dataset", default="url_unique")
    ap.add_argument("--attack", default="hub-flip", choices=["hub-flip", "label-flip-random"])
    ap.add_argument("--row-percent", type=float, default=30.0)
    ap.add_argument("--poison-draw", type=int, default=0)
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed)
    ap.add_argument("--hub-metric", default="knn")
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000],
                    help="fixed subset sizes; the (1-R) full budget is always added too")
    ap.add_argument("--no-downstream", action="store_true",
                    help="skip scoring the selected subsets with TabPFNv2")
    ap.add_argument("--n-pure-draws", type=int, default=5,
                    help="draws for the pure-m control (random CLEAN rows of the same size)")
    # Meta-Sift's own hyperparameters, at their defaults except where the backbone forces
    # a change (top_k: their ResNet18 has many more parameter blocks than an MLP).
    ap.add_argument("--num-sifter", type=int, default=5)
    ap.add_argument("--res-epochs", type=int, default=1)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--v-lr", type=float, default=0.0005)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--go-lr", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--num-act", type=int, default=4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--nesterov", type=bool, default=True)
    ap.add_argument("--noise-sigma", type=float, default=0.1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results/meta_sift")
    ap.add_argument("--tag", default=None, help="subdirectory suffix, for sweeping")
    ap.add_argument("--model-lr", type=float, default=None,
                    help="AdamW lr for the backbone; default is their 2.5e-5*batch/32")
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.gpu)
    tag = f"{a.attack}_r{int(round(a.row_percent)):03d}" + (f"_{a.tag}" if a.tag else "")
    out = a.out / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, "meta_sift").info

    # ---- their code, imported and used as written -----------------------------------
    import meta_sift.meta_sift_func as msf
    from meta_sift.util import get_NCR, set_seeds

    from tabfm_experiments.data import load_split

    set_seeds(a.random_seed)
    train, _ = default_splits(a.dataset)
    Xdf, y_clean, cols = load_split(train)
    Xraw = Xdf.to_numpy(np.float32)
    y_clean = np.asarray(y_clean).astype(int)
    n, n_feat = Xraw.shape

    # The exact same poison the rest of this project was evaluated on.
    k = k_from_percent(n, a.row_percent)
    Xt = torch.tensor(Xraw, device=f"cuda:{a.gpu}")
    flipped = poison_rows(a.attack, Xt, y_clean, k, run_id=a.poison_draw,
                          row_seed=a.row_seed, hub_metric=a.hub_metric, hub_k=a.hub_k)
    y_given = y_clean.copy()
    y_given[flipped] = 1 - y_given[flipped]

    # An MLP needs standardised inputs; TabPFN does this internally. Uses feature
    # statistics only, so no label information leaks.
    mu, sd = Xraw.mean(0), Xraw.std(0)
    Xz = (Xraw - mu) / np.where(sd < 1e-12, 1.0, sd)

    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%  {n}x{n_feat}, {k} flipped")
    log(f"  given class counts {np.bincount(y_given, minlength=2).tolist()}")
    log(f"  meta-sift: {a.num_sifter} sifters, warmup {a.warmup_epochs}, "
        f"res_epochs {a.res_epochs}, top_k {a.top_k}, v_lr {a.v_lr}, meta_lr {a.meta_lr}, "
        f"model_lr {a.model_lr}")
    # The (1-R) budget is what the k-fold defence removed to, so it is always included.
    # On credit-approval (457 rows) that is 320, not 3500 -- a fixed 3500 is impossible.
    full_budget = n - k
    sizes = sorted({m for m in a.sizes if m < full_budget} | {full_budget})
    log(f"  subset sizes: {sizes}  (full budget = n - k = {full_budget})")

    # Swap the backbone. build_training() and train_sifter() both construct
    # `ResNet18(num_classes=...)`, so rebinding that name is the whole substitution --
    # their algorithm code is untouched.
    msf.ResNet18 = lambda num_classes: TabularMLP(num_classes, n_features=n_feat)
    a.num_classes = 2
    if a.model_lr is not None:
        # Their build_training hard-codes lr = 2.5e-5 * batch/32, tuned for ResNet18 on
        # CIFAR-scale data. On 5000 tabular rows that is too slow to fit inside the
        # epoch budget, so allow it to be set.
        _orig_build = msf.build_training

        def _build(args):
            model, _, vnet, opt_c = _orig_build(args)
            opt_a = torch.optim.AdamW(model.parameters(), lr=a.model_lr,
                                      betas=(0.9, 0.95), weight_decay=0.05)
            return model, opt_a, vnet, opt_c
        msf.build_training = _build

    train_set = TabularPoisonSet(Xz, y_given, transform=None)
    vnet_list, mnet_list = msf.train_sifter(a, train_set)

    # Identification stage under sample dilution, as in their main.py.
    test_set = copy.deepcopy(train_set)
    test_set.transform = AddGaussianNoise(a.noise_sigma)
    v_res = msf.test_sifter(a, test_set, vnet_list, mnet_list)
    weights = v_res.mean(0)

    # Is the weight-assigning net pointing the right way at all? It should give LOW
    # weight to poisoned rows. A negative value here means the selection is inverted,
    # which shows up as NCR above 100 (worse than random).
    from sklearn.metrics import roc_auc_score
    auc_w = float(roc_auc_score(np.isin(np.arange(n), flipped).astype(int), -weights))
    log(f"\n  weight direction check: AUC(low weight -> poisoned) = {auc_w:.4f} "
        f"(0.5 = no signal, <0.5 = INVERTED)")
    log(f"  weight spread: mean {weights.mean():.4f} sd {weights.std():.4f} "
        f"min {weights.min():.4f} max {weights.max():.4f}")

    results = {"weight_auc_low_means_poisoned": auc_w, "full_budget": int(full_budget),
               "sizes": [int(m) for m in sizes],
               "weight_mean": float(weights.mean()), "weight_std": float(weights.std())}
    log("\n  Meta-Sift selection (their get_sifter_result, class-proportional):")
    for m in sizes:
        idx = msf.get_sifter_result(a, test_set, v_res, total_pick=m)
        rep = subset_report(idx, flipped, n)
        ncr = get_NCR(train_set, list(flipped), list(idx))
        results[f"native_{m}"] = {**rep, "n_selected": int(len(idx)), "NCR": float(ncr)}
        log(f"    m={m:<5d} selected {len(idx):<5d} purity {rep['purity']:.4f}  "
            f"poison in it {int(round((1 - rep['purity']) * len(idx)))}  NCR {ncr:.1f}%")

    log("\n  same weights, extracted with OUR rules (like-for-like with sift_context.py):")
    for strat in ("balanced", "proportional", "global"):
        for m in sizes:
            kept = select_clean_subset(weights, y_given, m, strat)
            rep = subset_report(kept, flipped, n)
            ncr = (1 - rep["purity"]) / (len(flipped) / n) * 100
            results[f"{strat}_{m}"] = {**rep, "NCR": float(ncr)}
            log(f"    {strat:<13s} m={m:<5d} purity {rep['purity']:.4f}  "
                f"poison in it {int(round((1 - rep['purity']) * m)):<4d} NCR {ncr:.1f}%")

    if not a.no_downstream:
        from tabfm_experiments.defense import METRICS, score_on_test
        from tabfm_experiments.runtime import (
            build_tabpfn_v2, device_from_gpu, evaluate_context,
        )
        dev = device_from_gpu(a.gpu)
        Xte_df, y_te, _ = load_split(default_splits(a.dataset)[1])
        X_all = torch.tensor(Xraw, device=dev)
        X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=dev)
        y_te = np.asarray(y_te).astype(int)
        clf = build_tabpfn_v2(dev, 0)
        clean_rows = np.setdiff1d(np.arange(n), flipped)

        def tab_score(rows, labels):
            rows = np.asarray(rows)
            i = torch.as_tensor(rows, dtype=torch.long, device=dev)
            yc = torch.as_tensor(labels[rows].astype(np.float32), device=dev)
            return score_on_test(clf, X_all[i], yc, X_test, y_te, evaluate_context)

        down = {"clean": tab_score(np.arange(n), y_clean),
                "poisoned": tab_score(np.arange(n), y_given),
                "oracle": tab_score(clean_rows, y_given)}
        log("\n  TabPFNv2 on the held-out test set "
            f"({len(y_te)} rows):")
        for kk in ("clean", "poisoned", "oracle"):
            log(f"    {kk:<22s} " + "  ".join(f"{m} {down[kk][m]:.4f}" for m in METRICS[:5]))

        rng = np.random.default_rng(0)
        for m in sizes:
            for strat in ("balanced", "global"):
                kept = select_clean_subset(weights, y_given, m, strat)
                down[f"{strat}_{m}"] = tab_score(kept, y_given)
            # Control: the same size drawn from the CLEAN rows. Separates "too few rows"
            # from "not pure enough".
            acc = [tab_score(rng.choice(clean_rows, min(m, len(clean_rows)), replace=False),
                             y_given) for _ in range(a.n_pure_draws)]
            down[f"pure_{m}"] = {kk: float(np.mean([r[kk] for r in acc])) for kk in acc[0]}
            log(f"    m={m:<5d} balanced " + "  ".join(
                f"{mm} {down[f'balanced_{m}'][mm]:.4f}" for mm in METRICS[:5]))
            log(f"    {'':<10s} pure-m   " + "  ".join(
                f"{mm} {down[f'pure_{m}'][mm]:.4f}" for mm in METRICS[:5]))
        results["downstream"] = down

    np.savez_compressed(out / "weights.npz", weights=weights.astype(np.float32),
                        v_res=v_res.astype(np.float32), flipped_indices=np.asarray(flipped),
                        y_clean=y_clean.astype(np.int8), y_given=y_given.astype(np.int8))
    write_json(out / "summary.json", {
        "method": "meta-sift", "dataset": a.dataset, "attack": a.attack,
        "row_percent": a.row_percent, "k_flipped": int(k), "n_context": n,
        "n_features": n_feat, "poison_draw": a.poison_draw,
        "num_sifter": a.num_sifter, "warmup_epochs": a.warmup_epochs,
        "res_epochs": a.res_epochs, "top_k": a.top_k, "v_lr": a.v_lr,
        "meta_lr": a.meta_lr, "model_lr": a.model_lr, "noise_sigma": a.noise_sigma,
        "backbone": "TabularMLP(63->256->256->128->2)",
        "results": results,
    })
    log(f"\n  wrote {out / 'summary.json'}")
    return results


if __name__ == "__main__":
    main()
