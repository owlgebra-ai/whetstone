"""File-based rollout bus between the trainer (spark) and the generator (turing).

Packet P7 §4 puts the trainer on spark and rollout generation on turing, with
`/data` (ZFS on turing, NFS-mounted on spark) as the shared artifact store. This
module is the contract between the two halves, in one file so neither side can
drift from it.

Why files rather than HTTP
--------------------------
The trainer needs **token ids**, not text: every segment mask in this project
comes from :func:`whetstone.segments.parse_segments` on ids, because splitting
the decoded string and re-tokenizing does not round-trip (segments.py's module
docstring) and a one-token-off mask silently corrupts the routing. vLLM's
offline ``LLM.generate`` hands back ``output.token_ids`` directly; the OpenAI
HTTP surface hands back text and string-keyed logprobs. Going through `/data`
keeps the ids exact and costs one NFS round-trip per step, which is noise next
to a 6-second optimizer step.

Every write is **temp-then-rename**. A reader must never observe a partial
request, response or checkpoint: the packet's own gotcha is that "a torn
checkpoint generates garbage that *parses* (g=1) and poisons a whole batch".
``os.replace`` is atomic within a directory on both ext4/ZFS and NFS.

Layout::

    <run_dir>/
      req/step_%06d.json      trainer → worker: prompts + sampling params
      resp/step_%06d.jsonl    worker → trainer: rollouts with token ids
      resp/step_%06d.done     written last; its presence means the jsonl is whole
      weights/v%06d/          bf16 export of the current policy
      weights/CURRENT         text file holding the newest complete version
      worker.status           worker heartbeat (json), for the trainer's timeout message
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_jsonl(path: str, rows: List[dict]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass
class RolloutRequest:
    step: int
    items: List[dict]          # [{uid, prompt, ground_truth, level, p_hat, seen}]
    params: Dict[str, Any]     # {K, temperature, top_p, max_tokens, seed}
    weights_version: int


class RolloutBus:
    """Both sides construct one of these against the same ``run_dir``."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.req_dir = os.path.join(run_dir, "req")
        self.resp_dir = os.path.join(run_dir, "resp")
        self.weights_dir = os.path.join(run_dir, "weights")
        for d in (self.req_dir, self.resp_dir, self.weights_dir):
            os.makedirs(d, exist_ok=True)

    # --- trainer side ------------------------------------------------------

    def post_request(self, req: RolloutRequest) -> str:
        path = os.path.join(self.req_dir, f"step_{req.step:06d}.json")
        _atomic_write_text(path, json.dumps({
            "step": req.step, "items": req.items, "params": req.params,
            "weights_version": req.weights_version, "posted_at": time.time(),
        }))
        return path

    def wait_response(self, step: int, timeout: float = 1800.0,
                      poll: float = 0.5) -> List[dict]:
        """Block until the worker's response for ``step`` is complete.

        Waits on the ``.done`` marker, never on the jsonl itself — the marker is
        written after the data, so its existence is the only safe signal that
        the file is whole.
        """
        done = os.path.join(self.resp_dir, f"step_{step:06d}.done")
        data = os.path.join(self.resp_dir, f"step_{step:06d}.jsonl")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(done):
                with open(data) as f:
                    return [json.loads(l) for l in f if l.strip()]
            time.sleep(poll)
        raise TimeoutError(
            f"no rollout response for step {step} after {timeout:.0f}s. "
            f"worker status: {self.worker_status()}"
        )

    def publish_weights(self, src_dir: str, version: int) -> str:
        """Publish a bf16 export as ``version``, atomically.

        The export is staged under a temp name and renamed into place before
        ``CURRENT`` is updated, so a worker that reads ``CURRENT`` always finds a
        complete directory behind it.
        """
        dst = os.path.join(self.weights_dir, f"v{version:06d}")
        staging = f"{dst}.staging"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        shutil.move(src_dir, staging)
        os.replace(staging, dst)
        _atomic_write_text(os.path.join(self.weights_dir, "CURRENT"), str(version))
        return dst

    def prune_weights(self, keep_versions: int = 2) -> int:
        """Delete old exports; each is ~3.4 GB. Returns how many were removed."""
        vs = sorted(glob.glob(os.path.join(self.weights_dir, "v[0-9]*")))
        removed = 0
        for p in vs[:-keep_versions] if keep_versions > 0 else vs:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        return removed

    # --- worker side -------------------------------------------------------

    def poll_request(self, handled: set) -> Optional[RolloutRequest]:
        """Oldest request not yet in ``handled``, or ``None``."""
        for path in sorted(glob.glob(os.path.join(self.req_dir, "step_*.json"))):
            step = int(os.path.basename(path)[5:11])
            if step in handled:
                continue
            try:
                with open(path) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                continue          # mid-rename on NFS; it will be there next poll
            return RolloutRequest(step=d["step"], items=d["items"],
                                  params=d["params"],
                                  weights_version=d.get("weights_version", 0))
        return None

    def post_response(self, step: int, rows: List[dict], meta: Optional[dict] = None) -> None:
        _atomic_write_jsonl(os.path.join(self.resp_dir, f"step_{step:06d}.jsonl"), rows)
        # The marker goes last and is what `wait_response` keys on.
        _atomic_write_text(os.path.join(self.resp_dir, f"step_{step:06d}.done"),
                           json.dumps(meta or {}))

    def write_status(self, status: dict) -> None:
        status = {**status, "ts": time.time()}
        _atomic_write_text(os.path.join(self.run_dir, "worker.status"),
                           json.dumps(status))

    # --- shared ------------------------------------------------------------

    def current_weights(self) -> Optional[Tuple[int, str]]:
        cur = os.path.join(self.weights_dir, "CURRENT")
        if not os.path.exists(cur):
            return None
        try:
            with open(cur) as f:
                v = int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return None
        path = os.path.join(self.weights_dir, f"v{v:06d}")
        return (v, path) if os.path.isdir(path) else None

    def worker_status(self) -> dict:
        p = os.path.join(self.run_dir, "worker.status")
        if not os.path.exists(p):
            return {"state": "never started"}
        try:
            with open(p) as f:
                s = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"state": "unreadable"}
        s["age_seconds"] = round(time.time() - s.get("ts", 0), 1)
        return s


__all__ = ["RolloutBus", "RolloutRequest"]
