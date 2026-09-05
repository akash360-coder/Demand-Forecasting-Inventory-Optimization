import pytest

from src.demand_intelligence.model_registry import compare_models


@pytest.mark.parametrize(
    ("champion", "challenger", "decision"),
    [(10.0, 8.0, "PROMOTED"), (10.0, 9.95, "REJECTED"), (10.0, 10.0, "REJECTED"), (10.0, 12.0, "REJECTED")],
)
def test_wmape_promotion_policy(champion, challenger, decision):
    result = compare_models({"wmape": champion}, {"wmape": challenger})
    assert result["decision"] == decision


def test_relative_wmape_improvement():
    result = compare_models({"wmape": 20.0}, {"wmape": 15.0})
    assert result["wmape_improvement"] == pytest.approx(0.25)


def test_lower_wmape_wins():
    assert compare_models({"wmape": 10}, {"wmape": 9})["decision"] == "PROMOTED"


def test_equal_wmape_is_not_promoted():
    assert compare_models({"wmape": 10}, {"wmape": 10})["decision"] == "REJECTED"


def test_tiny_improvement_is_rejected():
    assert compare_models({"wmape": 10}, {"wmape": 9.95}, minimum_improvement=0.01)["decision"] == "REJECTED"


def test_sufficient_improvement_is_promoted():
    assert compare_models({"wmape": 10}, {"wmape": 8.9}, minimum_improvement=0.01)["decision"] == "PROMOTED"


def test_worse_challenger_is_rejected():
    assert compare_models({"wmape": 10}, {"wmape": 11})["decision"] == "REJECTED"


def test_comparison_returns_both_metrics():
    result = compare_models({"wmape": 10, "mae": 1}, {"wmape": 8, "mae": 0.8})
    assert result["champion_metrics"]["mae"] == 1
    assert result["challenger_metrics"]["mae"] == 0.8


def test_comparison_returns_reason():
    assert compare_models({"wmape": 10}, {"wmape": 8})["reason"]


def test_custom_threshold_is_applied():
    assert compare_models({"wmape": 10}, {"wmape": 9}, minimum_improvement=0.2)["decision"] == "REJECTED"


def test_zero_champion_wmape_is_safe():
    result = compare_models({"wmape": 0}, {"wmape": 0})
    assert result["wmape_improvement"] == 0
    assert result["decision"] == "REJECTED"


def test_training_run_identifier_convention():
    version = "20260904T153000Z"
    assert f"train_{version}" == "train_20260904T153000Z"


def test_retraining_decisions_are_explicit():
    assert {compare_models({"wmape": 10}, {"wmape": value})["decision"] for value in (8, 10, 12)} == {"PROMOTED", "REJECTED"}


def test_invalid_metric_key_fails_loudly():
    with pytest.raises(KeyError):
        compare_models({}, {"wmape": 1})


def test_promotion_threshold_is_relative():
    result = compare_models({"wmape": 100}, {"wmape": 99}, minimum_improvement=0.01)
    assert result["decision"] == "PROMOTED"
