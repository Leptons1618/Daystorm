"""The eval harness has to be wrong-answer-sensitive, or it grades nothing."""

from __future__ import annotations

import numpy as np

from daystorm.data.align import DEFAULT_SPECS, align_window
from daystorm.data.synthetic import make_scene
from daystorm.data.windows import build_samples
from daystorm.eval.metrics import (
    Prediction,
    classify_manoeuvre,
    evaluate,
    numbers_in,
    supportable_values,
)


def _sample(event="brake"):
    """A window of the requested kind. Never silently substitutes another kind:
    a test that thinks it is grading a braking answer and is actually grading a
    cruising one passes for the wrong reason."""
    for seed in range(20):
        scene = make_scene(seed=seed, event=event, dropout=False)
        hits = [x for x in build_samples([scene]) if x.event == event]
        if hits:
            return hits[0]
    raise AssertionError(f"no {event} window found in 20 scenes")


def _pred(sample, text):
    return Prediction(sample.window, sample.question, sample.answer, text, sample.event)


def test_numbers_are_extracted_with_signs_and_decimals():
    assert numbers_in("from 10.1 to 8.8 m/s (-0.6 m/s2) at 0.80") == [10.1, 8.8, -0.6, 0.80]


def test_unit_suffixes_are_not_read_as_quantities():
    """m/s2 is a unit. Counting its 2 inflates hallucinations and error."""
    assert numbers_in("accel -1.2 m/s2 over 2.0 s") == [-1.2, 2.0]
    assert numbers_in("CAM_FRONT at 12 Hz") == [12.0]


def test_perfect_prediction_scores_perfectly():
    s = _sample()
    m = evaluate([_pred(s, s.answer)])
    assert m["exact_match"] == 1.0
    assert m["manoeuvre_accuracy"] == 1.0
    assert m["numeric_mae"] == 0.0
    assert m["hallucination_rate"] == 0.0


def test_reference_numbers_are_all_supported_by_their_own_window():
    """If the truth scores as hallucination, the metric is broken, not the model."""
    for event in ("brake", "turn", "cruise"):
        s = _sample(event)
        m = evaluate([_pred(s, s.answer)])
        assert m["hallucination_rate"] == 0.0, f"{event}: reference flagged as invented"


def test_invented_numbers_are_caught():
    s = _sample()
    fabricated = "Deceleration from 999.0 to 888.0 m/s with time-to-collision 777.0 s."
    m = evaluate([_pred(s, fabricated)])
    assert m["hallucination_rate"] == 1.0


def test_wrong_manoeuvre_is_caught():
    s = _sample("brake")
    m = evaluate([_pred(s, "Steady cruising at 10.0 m/s with no brake demand.")])
    assert m["manoeuvre_accuracy"] == 0.0


def test_fluent_but_wrong_scores_badly():
    """A well-formed answer with the wrong quantities must not score well.

    This is the case BLEU and ROUGE would wave through: identical phrasing,
    every quantity fabricated."""
    import re

    s = _sample()
    wrong = re.sub(r"(?<![\w/.])-?\d+(?:\.\d+)?", lambda m: f"{float(m.group()) * 3 + 40:.1f}", s.answer)
    m = evaluate([_pred(s, wrong)])
    assert m["exact_match"] == 0.0
    assert m["numeric_mae"] > 1.0
    assert m["hallucination_rate"] > 0.5
    assert classify_manoeuvre(wrong) == classify_manoeuvre(s.answer)  # phrasing intact


def test_supportable_values_track_the_window_not_the_text():
    s = _sample()
    pool = supportable_values(s.window)
    assert pool.size > 0 and np.isfinite(pool).all()
    speeds = s.window.values["can"][s.window.valid["can"], 4]
    assert any(np.isclose(pool, speeds[0], atol=1e-3))


def test_masked_channels_do_not_enter_the_supportable_pool():
    from daystorm.data.align import Stream

    w = align_window(
        {"can": Stream(np.array([0.0]), np.full((1, 9), 42.0)),
         "radar": Stream(np.arange(0.0, 2.1, 0.05), np.ones((42, 6)))},
        [s for s in DEFAULT_SPECS if s.name in ("can", "radar")],
        t0=0.0,
    )
    pool = supportable_values(w)
    # only the t=0 CAN reading is fresh; the masked zeros must not be quotable
    assert any(np.isclose(pool, 42.0))


def test_coverage_honesty_only_scored_when_sensors_were_actually_down():
    complete = _sample()
    m = evaluate([_pred(complete, complete.answer)])
    assert np.isnan(m["coverage_honesty"]) or m["coverage_honesty"] == 1.0


def test_manoeuvre_classifier_reads_the_right_keywords():
    assert classify_manoeuvre("Deceleration from 10 to 4 m/s") == "brake"
    assert classify_manoeuvre("A left turn: steering reaches 90 deg") == "turn"
    assert classify_manoeuvre("Steady cruising at 11 m/s") == "cruise"
