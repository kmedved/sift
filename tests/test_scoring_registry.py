import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, log_loss

from sift.scoring import VALID_SCORERS, get_scoring


class _PredictThenProbaModel:
    def __init__(self, predictions, probabilities=None, classes=None):
        self._predictions = np.asarray(predictions)
        self._probabilities = None if probabilities is None else np.asarray(probabilities)
        if self._probabilities is not None and classes is not None:
            self.classes_ = np.asarray(classes)

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        if self._probabilities is None:
            raise AttributeError("No probabilities configured")
        return self._probabilities


class _PredictOnlyModel:
    def __init__(self, predictions):
        self._predictions = np.asarray(predictions)

    def predict(self, X):
        return self._predictions


def test_scoring_registry_exposes_supported_names():
    assert set(VALID_SCORERS) == {
        "neg_mse",
        "neg_rmse",
        "neg_mae",
        "r2",
        "accuracy",
        "balanced_accuracy",
        "neg_error",
        "neg_logloss",
    }


def test_scoring_registry_scores_match_expected_values():
    X = np.zeros((4, 1))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 2.8, 4.2])
    w = np.array([1.0, 2.0, 1.0, 1.0])
    model = _PredictThenProbaModel(y_pred)

    regression_expected = {
        "neg_mse": -0.022,
        "neg_rmse": -np.sqrt(0.022),
        "neg_mae": -0.14,
        "r2": 1.0 - 0.022 / (1.04 + 1e-10),
    }
    for name, target in regression_expected.items():
        scorer = get_scoring(name)
        assert scorer.higher_is_better is True
        result = scorer(model, X, y, w)
        assert result == pytest.approx(target, rel=1e-12, abs=1e-12)

    y_class = np.array([1, 0, 1, 0])
    y_class_pred = np.array([1, 1, 0, 0])
    w_class = np.array([1.0, 2.0, 1.0, 1.0])
    class_model = _PredictThenProbaModel(y_class_pred)

    class_expected = {
        "accuracy": 0.4,
        "balanced_accuracy": float(
            balanced_accuracy_score(y_class, y_class_pred, sample_weight=w_class)
        ),
        "neg_error": -0.6,
    }
    for name, target in class_expected.items():
        scorer = get_scoring(name)
        assert scorer.higher_is_better is True
        result = scorer(class_model, X, y_class, w_class)
        assert result == pytest.approx(target, rel=1e-12, abs=1e-12)


def test_scoring_registry_flattens_single_column_predictions():
    X = np.zeros((4, 1))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([[1.0], [2.0], [3.0], [5.0]])
    w = np.ones(4)
    model = _PredictThenProbaModel(y_pred)

    assert get_scoring("neg_mse")(model, X, y, w) == pytest.approx(-0.25)
    assert get_scoring("neg_mae")(model, X, y, w) == pytest.approx(-0.25)

    y_class = np.array(["a", "b", "a", "b"])
    y_class_pred = np.array([["a"], ["b"], ["b"], ["b"]])
    class_model = _PredictThenProbaModel(y_class_pred)

    assert get_scoring("accuracy")(class_model, X, y_class, w) == pytest.approx(0.75)
    assert get_scoring("neg_error")(class_model, X, y_class, w) == pytest.approx(-0.25)


def test_scoring_registry_rejects_plain_aliases():
    with pytest.raises(ValueError, match="Unknown scoring"):
        get_scoring("error")

    with pytest.raises(ValueError, match="Unknown scoring"):
        get_scoring("logloss")


def test_scoring_registry_neg_logloss_uses_probabilities():
    y = np.array([0, 0, 1, 1])
    X = np.zeros((4, 2))
    proba = np.array(
        [
            [0.90, 0.10],
            [0.65, 0.35],
            [0.20, 0.80],
            [0.30, 0.70],
        ]
    )
    w = np.array([1.0, 1.0, 2.0, 1.0])

    model = _PredictThenProbaModel(predictions=np.array([0, 0, 1, 1]), probabilities=proba, classes=(0, 1))
    scorer = get_scoring("neg_logloss")
    expected = -float(log_loss(y, proba, labels=np.array([0, 1]), sample_weight=w))
    result = scorer(model, X, y, w)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)

    no_proba_model = _PredictOnlyModel(predictions=np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="requires model.predict_proba"):
        get_scoring("neg_logloss")(no_proba_model, X, y, w)


def test_scoring_registry_higher_is_better_metadata():
    for name in VALID_SCORERS:
        assert get_scoring(name).higher_is_better is True
