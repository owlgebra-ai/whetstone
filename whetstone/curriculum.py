"""Success-rate-ordered, saturation-paced batch curriculum (packet P7 §8).

The user's direction was "easy first"; the packet's refinement delivers it **by
measurement rather than by label** — batches are tilted by the *measured* pass
rate p̂ from the K=8 bucketing run, not by the `level` field and not on a step
schedule. On this pool the two happen to coincide (all 2,000 level-1 problems
are GSM8K, every level 2–10 problem is DeepMath — activity 010 run 4), which is
exactly why tilting on the measurement is safe: it reproduces the intent without
inheriting the label's assumptions.

Three rules, each with its reason:

1. **~75% from the high-p̂ bucket early, but never 100%.** A pure-easy diet
   postpones the cure: the loop tail lives on hard problems, and the
   ``g=0``-vs-``r_fmt`` contrast *within* a hard group is the gradient that
   extinguishes it.
2. **Progression is saturation-driven, not scheduled.** Problems that reach K/K
   drop out by themselves (dynamic sampling), the live mixed-fraction per bucket
   is recomputed at every eval, and the tilt shifts down a bucket when the easy
   tier's mixed share falls below ``shift_threshold``. No fixed switch step —
   the buckets know whether "step 15" is too early or too late.
3. **0/K problems are never batched.** They are pedagogy rescue's clientele
   (packet §10); batching them spends rollouts on groups that cannot produce a
   within-group advantage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

#: Bucket edges on p̂ (fraction of K correct). Ordered easiest → hardest.
BANDS: Sequence[tuple] = (
    ("high", 5 / 8, 1.0),      # p̂ >= 5/8 — mostly GSM8K on this pool
    ("mid", 3 / 8, 5 / 8),
    ("low", 1e-9, 3 / 8),      # > 0 but < 3/8; 0/K is excluded entirely
)

#: Early tilt: 75% high, then split the rest toward the harder bands.
DEFAULT_TILT: Dict[str, float] = {"high": 0.75, "mid": 0.15, "low": 0.10}

#: When a band's live mixed share drops below this, shift the tilt one band down.
SHIFT_THRESHOLD = 1 / 3


@dataclass
class Problem:
    uid: str
    prompt: str
    ground_truth: str
    level: int
    seen: bool
    p_hat: float
    band: str = ""
    n_seen_in_training: int = 0
    live_p_hat: Optional[float] = None    # updated from observed rollouts

    @property
    def effective_p_hat(self) -> float:
        """Live measurement when we have one, else the bucketing run's."""
        return self.p_hat if self.live_p_hat is None else self.live_p_hat


def band_of(p_hat: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= p_hat < hi or (hi == 1.0 and p_hat >= lo):
            return name
    return "zero"


@dataclass
class Curriculum:
    """Holds the mixed-group pool and hands out batches."""

    problems: List[Problem]
    tilt: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TILT))
    shift_threshold: float = SHIFT_THRESHOLD
    retired: set = field(default_factory=set)     # reached K/K under training
    shifts: int = 0

    def __post_init__(self) -> None:
        for p in self.problems:
            p.band = band_of(p.p_hat)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_bucket_rows(cls, rows: Iterable[dict], pool: Dict[str, dict]) -> "Curriculum":
        """Build from ``stagec_bucket.py``'s ``buckets.jsonl``.

        Only mixed groups enter: ``0/K`` goes to rescue and ``K/K`` is already
        saturated, and neither can produce a within-group advantage.
        """
        probs = []
        for r in rows:
            if r["bucket"] != "mixed":
                continue
            rec = pool.get(r["_uid"])
            if rec is None:
                continue
            probs.append(Problem(
                uid=r["_uid"], prompt=rec["prompt"],
                ground_truth=rec.get("ground_truth", ""),
                level=int(r.get("level", 0)), seen=bool(r.get("seen", False)),
                p_hat=float(r["p_hat"]),
            ))
        return cls(problems=probs)

    # --- sampling ----------------------------------------------------------

    def _active(self) -> List[Problem]:
        return [p for p in self.problems if p.uid not in self.retired]

    def band_pools(self) -> Dict[str, List[Problem]]:
        out: Dict[str, List[Problem]] = {name: [] for name, _, _ in BANDS}
        for p in self._active():
            out.setdefault(p.band, []).append(p)
        return out

    def sample(self, n: int, rng: random.Random) -> List[Problem]:
        """Draw ``n`` problems under the current tilt.

        Bands that have run dry give their share back to the others rather than
        shrinking the batch — a smaller batch would quietly change the effective
        learning rate mid-run.
        """
        pools = self.band_pools()
        available = {b: pool for b, pool in pools.items() if pool}
        if not available:
            return []

        weights = {b: self.tilt.get(b, 0.0) for b in available}
        total = sum(weights.values())
        if total <= 0:
            weights = {b: 1.0 for b in available}
            total = float(len(available))

        picked: List[Problem] = []
        chosen: set = set()
        bands = list(available)
        probs = [weights[b] / total for b in bands]
        # Sample band-by-band with replacement across draws but not within a
        # batch: repeating a problem inside one optimizer step would double its
        # gradient without doubling its evidence.
        guard = 0
        while len(picked) < n and guard < 50 * n:
            guard += 1
            b = rng.choices(bands, weights=probs, k=1)[0]
            pool = available[b]
            cand = pool[rng.randrange(len(pool))]
            if cand.uid in chosen:
                continue
            chosen.add(cand.uid)
            cand.n_seen_in_training += 1
            picked.append(cand)
        return picked

    # --- feedback ----------------------------------------------------------

    def observe(self, uid: str, n_correct: int, k: int) -> None:
        """Record a group's realized pass rate; retire it if saturated."""
        for p in self.problems:
            if p.uid != uid:
                continue
            p.live_p_hat = n_correct / max(1, k)
            p.band = band_of(p.live_p_hat)
            if n_correct == k:
                self.retired.add(uid)
            elif uid in self.retired:
                self.retired.discard(uid)   # fell back out of saturation
            return

    def live_mixed_fraction(self) -> Dict[str, float]:
        """Share of each band's problems that are still un-retired."""
        by_band: Dict[str, List[Problem]] = {name: [] for name, _, _ in BANDS}
        for p in self.problems:
            by_band.setdefault(p.band, []).append(p)
        out = {}
        for b, ps in by_band.items():
            if not ps:
                out[b] = 0.0
            else:
                out[b] = sum(1 for p in ps if p.uid not in self.retired) / len(ps)
        return out

    def retilt(self) -> Dict[str, float]:
        """Shift the tilt one band down when the easy tier is exhausted.

        Called at every eval, not on a step schedule (packet §8).
        """
        live = self.live_mixed_fraction()
        order = [name for name, _, _ in BANDS]
        top = order[0]
        pools = self.band_pools()
        top_share = live.get(top, 0.0)
        top_empty = not pools.get(top)

        if (top_share < self.shift_threshold or top_empty) and self.shifts < len(order) - 1:
            self.shifts += 1
            remaining = order[self.shifts:]
            if remaining:
                head, *rest = remaining
                new = {head: 0.75}
                if rest:
                    share = 0.25 / len(rest)
                    for b in rest:
                        new[b] = share
                else:
                    new = {head: 1.0}
                for b in order[: self.shifts]:
                    new.setdefault(b, 0.0)
                self.tilt = new
        return dict(self.tilt)

    # --- reporting ---------------------------------------------------------

    def stats(self) -> dict:
        pools = self.band_pools()
        return {
            "n_total": len(self.problems),
            "n_active": len(self._active()),
            "n_retired": len(self.retired),
            "tilt": dict(self.tilt),
            "shifts": self.shifts,
            "band_sizes": {b: len(p) for b, p in pools.items()},
            "live_mixed_fraction": self.live_mixed_fraction(),
        }


__all__ = ["Problem", "Curriculum", "band_of", "BANDS", "DEFAULT_TILT"]
