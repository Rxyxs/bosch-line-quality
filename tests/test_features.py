import numpy as np
import polars as pl

from bosch.analysis import best_policy, capture_curve
from bosch.features import sequence_features


def test_sequence_features_use_order_not_labels():
    proc = pl.DataFrame({"Id": [1, 2, 3, 10, 11], "start": [5.0, 5.0, 5.0, 7.0, 7.0]})
    s = sequence_features(proc).sort("Id")
    assert s["batch_size"].to_list() == [3, 3, 3, 2, 2]
    assert s["same_lot_next"].to_list() == [1, 1, 0, 1, None]  # last part has no successor
    assert s["id_gap_next"].to_list()[2] == 7
    assert "Response" not in s.columns


def test_capture_curve_perfect_and_random_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = capture_curve(y, y.astype(float), np.array([0.2]))
    assert perfect[0] == 1.0


def test_policy_inspects_nothing_when_failures_are_cheap():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.7])  # model ranks the failure last
    assert best_policy(y, p, ratio=1.0) == (0.0, 1.0)
    frac, cost = best_policy(y, y.astype(float), ratio=10.0)
    assert frac == 0.25 and cost == 0.1
