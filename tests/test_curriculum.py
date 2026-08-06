"""Tests for the Stage-C batch curriculum (packet P7 §8)."""

from __future__ import annotations

import random

import pytest

from whetstone.curriculum import BANDS, Curriculum, Problem, band_of


def _p(uid: str, p_hat: float, level: int = 1, seen: bool = False) -> Problem:
    return Problem(uid=uid, prompt="q", ground_truth="1", level=level,
                   seen=seen, p_hat=p_hat)


def _curriculum(n_high=40, n_mid=40, n_low=40) -> Curriculum:
    probs = ([_p(f"h{i}", 0.875) for i in range(n_high)]
             + [_p(f"m{i}", 0.5) for i in range(n_mid)]
             + [_p(f"l{i}", 0.25) for i in range(n_low)])
    return Curriculum(problems=probs)


def test_band_edges() -> None:
    assert band_of(1.0) == "high"
    assert band_of(0.875) == "high"
    assert band_of(5 / 8) == "high"
    assert band_of(0.5) == "mid"
    assert band_of(3 / 8) == "mid"
    assert band_of(0.25) == "low"
    assert band_of(0.125) == "low"
    assert band_of(0.0) == "zero", "0/K must never be batched"


def test_zero_bucket_never_enters_the_curriculum() -> None:
    rows = [
        {"_uid": "a", "bucket": "mixed", "p_hat": 0.5, "level": 1, "seen": True},
        {"_uid": "b", "bucket": "0/K", "p_hat": 0.0, "level": 9, "seen": False},
        {"_uid": "c", "bucket": "K/K", "p_hat": 1.0, "level": 1, "seen": True},
    ]
    pool = {u: {"prompt": "q", "ground_truth": "1"} for u in "abc"}
    c = Curriculum.from_bucket_rows(rows, pool)
    assert [p.uid for p in c.problems] == ["a"]


def test_early_tilt_favours_easy_but_never_excludes_hard() -> None:
    """~75/25, and the hard share is never zero — the loop tail lives there."""
    c = _curriculum()
    rng = random.Random(0)
    counts = {"high": 0, "mid": 0, "low": 0}
    for _ in range(200):
        for p in c.sample(8, rng):
            counts[p.band] += 1
    total = sum(counts.values())
    assert 0.68 < counts["high"] / total < 0.82, counts
    assert counts["mid"] + counts["low"] > 0, "a pure-easy diet postpones the cure"
    assert counts["low"] / total > 0.03


def test_sample_never_repeats_a_problem_within_one_batch() -> None:
    c = _curriculum(n_high=5, n_mid=0, n_low=0)
    rng = random.Random(1)
    batch = c.sample(5, rng)
    assert len({p.uid for p in batch}) == len(batch)


def test_saturated_problems_retire_and_stop_being_sampled() -> None:
    c = _curriculum(n_high=3, n_mid=0, n_low=0)
    c.observe("h0", 8, 8)
    assert "h0" in c.retired
    rng = random.Random(0)
    seen = {p.uid for _ in range(50) for p in c.sample(2, rng)}
    assert "h0" not in seen


def test_a_problem_can_fall_back_out_of_saturation() -> None:
    c = _curriculum(n_high=2, n_mid=0, n_low=0)
    c.observe("h0", 8, 8)
    assert "h0" in c.retired
    c.observe("h0", 6, 8)
    assert "h0" not in c.retired


def test_observe_rebands_on_the_live_rate() -> None:
    c = _curriculum(n_high=1, n_mid=0, n_low=0)
    assert c.problems[0].band == "high"
    c.observe("h0", 2, 8)               # 0.25 -> low
    assert c.problems[0].band == "low"
    assert c.problems[0].effective_p_hat == pytest.approx(0.25)


def test_retilt_shifts_down_when_the_easy_tier_is_exhausted() -> None:
    c = _curriculum(n_high=10, n_mid=10, n_low=10)
    assert c.tilt["high"] == pytest.approx(0.75)
    for i in range(9):                  # 9/10 of the high band saturates
        c.observe(f"h{i}", 8, 8)
    tilt = c.retilt()
    assert c.shifts == 1
    assert tilt["mid"] == pytest.approx(0.75), tilt
    assert tilt["high"] == pytest.approx(0.0)
    assert tilt["low"] > 0


def test_retilt_is_a_no_op_while_the_easy_tier_still_has_work() -> None:
    c = _curriculum()
    before = dict(c.tilt)
    assert c.retilt() == before
    assert c.shifts == 0


def test_retilt_stops_at_the_last_band() -> None:
    c = _curriculum(n_high=1, n_mid=1, n_low=1)
    for uid in ("h0", "m0", "l0"):
        c.observe(uid, 8, 8)
    for _ in range(5):
        c.retilt()
    assert c.shifts <= len(BANDS) - 1


def test_empty_band_gives_its_share_back_rather_than_shrinking_the_batch() -> None:
    """A shrinking batch would quietly change the effective learning rate."""
    c = _curriculum(n_high=0, n_mid=20, n_low=20)
    batch = c.sample(8, random.Random(0))
    assert len(batch) == 8


def test_sample_returns_empty_when_everything_is_retired() -> None:
    c = _curriculum(n_high=2, n_mid=0, n_low=0)
    c.observe("h0", 8, 8)
    c.observe("h1", 8, 8)
    assert c.sample(4, random.Random(0)) == []


def test_stats_reports_what_the_dashboard_needs() -> None:
    c = _curriculum()
    c.observe("h0", 8, 8)
    s = c.stats()
    assert s["n_total"] == 120 and s["n_retired"] == 1 and s["n_active"] == 119
    assert set(s["band_sizes"]) >= {"high", "mid", "low"}
    assert 0.0 <= s["live_mixed_fraction"]["high"] <= 1.0
