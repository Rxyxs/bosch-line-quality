import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from bosch.model import best_threshold, mcc_at, mcc_curve


@pytest.mark.parametrize("seed", range(5))
def test_mcc_curve_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    mcc, thr = mcc_curve(y, p)
    for i in rng.choice(len(thr), 20, replace=False):
        assert mcc[i] == pytest.approx(matthews_corrcoef(y, p >= thr[i]), abs=1e-9)
    t, best = best_threshold(y, p)
    assert best == pytest.approx(matthews_corrcoef(y, p >= t), abs=1e-9)
    assert mcc_at(y, p, t) == pytest.approx(best, abs=1e-9)


def test_mcc_degenerate_is_zero():
    y = np.array([0, 0, 1])
    assert mcc_at(y, np.zeros(3), 0.5) == 0.0
