"""Generate register-card variants for a paired faithfulness A/B (activity 005 finding 9).

The pilot audit found the compressor drops derivations and branches on hard
problems (49% faithful overall, 0% at level 7). Two competing explanations, and
they imply different fixes:

  * **rules** — the card does not say clearly enough what must be preserved;
  * **scale** — the card's five exemplars all compress *short* problems to
    127–533 characters, so a model imitating them compresses an 8,800-token
    trace ~60x whatever its content. Card §1.4 already says branch elimination
    stays, and it is violated 56% of the time, which is evidence *against* the
    rules explanation and *for* this one.

Rather than argue, this emits one variant per lever and lets the judge decide.
Each variant is a **surgical insert after the §1.4 table** — notation, symbols,
exemplars and every other section are byte-identical across arms, so a
difference in faithfulness is attributable to the inserted block alone.

**The ratified card is never modified.** It is read as the control (arm A) and
variants are written to ``--out_dir``. Nothing here edits ``configs/``.

Usage::

    python scripts/make_card_variants.py --out_dir /data/whetstone/runs/card_ab/cards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

# Anchor: the last row of the §1.4 elision table. Variants insert directly
# after it, before §1.5.
ANCHOR = re.compile(r"(?m)^(\| Markdown headers.*\n)")

SCALE = """
**Length scales with the source.** The exemplars below are short because their
*problems* are short — they are not a target length. A long, many-step original
must produce a long compact rewrite. Compressing a 40-step derivation into six
lines is a failure, not a success: it means steps were dropped. Never shorten by
omitting reasoning; shorten only by removing the prose in the left column above.
"""

DERIVATION = """
**Never assert a result whose derivation is absent.** Every value that appears
in the rewrite must be traceable to an operation shown in the rewrite. If the
original derives a result over several steps, those steps appear; writing the
conclusion alone — even when it is correct — is the worst failure this register
has, because the reader cannot follow how it was reached.
"""

BRANCH = """
**Every branch survives, including the ones that failed.** If the original
considers a case, tries an approach that does not work, or corrects itself, that
appears in the rewrite with its verdict (`⇒` result or `✗` rejection). Rejected
alternatives and self-corrections *are* the reasoning; a rewrite that keeps only
the path that worked has deleted the thinking and kept the answer.
"""

VARIANTS = {
    "A_control": "",
    "B_scale": SCALE,
    "C_derivation": DERIVATION,
    "D_branch": BRANCH,
    "E_all": SCALE + DERIVATION + BRANCH,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    base = open(args.card).read()
    if not ANCHOR.search(base):
        raise SystemExit("[variants] §1.4 table anchor not found — the card "
                         "changed shape; update ANCHOR before trusting an A/B.")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {}
    for name, block in VARIANTS.items():
        text = base if not block else ANCHOR.sub(r"\1" + block, base, count=1)
        path = os.path.join(args.out_dir, f"{name}.md")
        with open(path, "w") as f:
            f.write(text)
        meta[name] = {
            "path": path,
            "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "added_chars": len(text) - len(base),
        }
        print(f"[out] {name:<14} {meta[name]['sha1'][:12]}  "
              f"+{meta[name]['added_chars']} chars  -> {path}")

    assert meta["A_control"]["sha1"] == hashlib.sha1(base.encode()).hexdigest(), \
        "control must be byte-identical to the ratified card"
    with open(os.path.join(args.out_dir, "variants.json"), "w") as f:
        json.dump({"base_card": args.card, "variants": meta}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
