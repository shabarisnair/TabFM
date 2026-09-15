import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabfm_experiments.data import fingerprint_frame, find_target, load_split
from tabfm_experiments.io import jsonable, sha256_file


def test_duplicates_share_fingerprint_and_int_float_agree():
    a = pd.DataFrame({"x": [1, 1, 2], "y": [0.5, 0.5, np.nan], "t": [0, 0, 1]})
    fp = fingerprint_frame(a)
    assert fp[0] == fp[1] and fp[0] != fp[2]
    # the same row read with a float dtype (e.g. a column holding NaN elsewhere)
    b = pd.DataFrame({"x": [1.0], "y": [0.5], "t": [0.0]})
    assert fingerprint_frame(b)[0] == fp[0]
    # NaN is normalised
    c = pd.DataFrame({"x": [2], "y": [float("nan")], "t": [1]})
    assert fingerprint_frame(c)[0] == fp[2]


def test_fingerprint_ignores_index_but_not_column_order():
    a = pd.DataFrame({"x": [1], "y": [2]}, index=[7])
    b = pd.DataFrame({"x": [1], "y": [2]}, index=[0])
    assert fingerprint_frame(a)[0] == fingerprint_frame(b)[0]
    assert fingerprint_frame(a[["y", "x"]])[0] != fingerprint_frame(a)[0]


def test_jsonable_and_hash(tmp_path: Path):
    obj = {"a": np.float32(1.5), "b": np.arange(3), Path("p"): (np.bool_(True), None)}
    s = json.dumps(jsonable(obj))
    assert json.loads(s) == {"a": 1.5, "b": [0, 1, 2], "p": [True, None]}
    f = tmp_path / "f.txt"
    f.write_text("abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_find_target_walks_up(tmp_path: Path):
    splits = tmp_path / "ds" / "splits"
    sel = splits / "selected" / "natural"
    sel.mkdir(parents=True)
    (splits / "split_info.json").write_text(json.dumps({"target": "lab"}))
    csv = sel / "context_10.csv"
    pd.DataFrame({"f": [1, 2], "lab": [0, 1]}).to_csv(csv, index=False)
    assert find_target(csv, None) == "lab"
    X, y, cols = load_split(csv)
    assert cols == ["f"] and y.tolist() == [0, 1]
