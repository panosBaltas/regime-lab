import numpy as np
import pytest

from regime_lab.ensemble import selected_allocator
from regime_lab.statistics import block_max_sharpe, deflated_sharpe


def test_multiple_trial_penalty_is_directionally_correct_and_catalog_adjustment_is_bounded():
    rng = np.random.default_rng(31)
    values = rng.normal(.003, .02, (120, 3))
    few = deflated_sharpe(values, 3)
    many = deflated_sharpe(values, 150)
    assert np.all(np.array(many["iid_probabilities"]) < few["iid_probabilities"])
    one = block_max_sharpe(values[:, :1], 0, replications=500)
    catalog = block_max_sharpe(values, 0, replications=500)
    assert one["max_statistic_p_value"] <= catalog["max_statistic_p_value"] <= 1
    assert block_max_sharpe(values, 0, replications=500) == catalog


def test_primary_report_choice_is_the_recorded_selection_and_invalid_rules_fail():
    assert selected_allocator({"selected": "rolling_risk"}) == "rolling_risk"
    assert selected_allocator({"selected": "static_risk"}) == "static_risk"
    with pytest.raises(ValueError, match="Unknown"):
        selected_allocator({"selected": "evaluation_best"})
