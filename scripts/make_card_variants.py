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


EXEMPLAR_TMPL = """
### Exemplar {n} (long trace) — level {level}, {ntok:,}-token original

- **_uid:** `{uid}` — **gold:** `{gold}`
- **verbose trace:** {ntok:,} tokens. Note the rewrite is **{lines} lines** — a
  faithful compression of a long derivation is itself long. Every case tried,
  every rejection (`✗`), and every self-correction (`!`) survives.

**Compact-register rewrite:**

```
{body}
```
"""


def _drop_exemplars_from(text: str, first: int) -> str:
    """Remove `### Exemplar N` sections with N >= ``first`` (in-place, §3 only)."""
    if first <= 0:
        return text
    parts = re.split(r"(?m)^(### Exemplar (\d+).*)$", text)
    out = [parts[0]]
    for head, num, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        if int(num) >= first:
            continue
        out.append(head + body)
    return "".join(out)


def _append_exemplars(text: str, exemplars: list[dict]) -> str:
    """Append long-trace exemplars to the end of §3, before §4."""
    n_existing = len(re.findall(r"(?m)^### Exemplar \d+", text))
    blocks = "".join(
        EXEMPLAR_TMPL.format(n=n_existing + i + 1, level=e.get("level"),
                             uid=e["_uid"], gold=e.get("ground_truth", ""),
                             ntok=e["verbose_think_tokens"],
                             lines=e["exemplar_lines"], body=e["exemplar"])
        for i, e in enumerate(exemplars))
    # The `---` rule belongs to the *previous* exemplar's body, so dropping the
    # trailing exemplars takes it with them. Anchor on §4 itself and treat the
    # rule as optional.
    m = re.search(r"(?m)^(?:---\n\n)?## 4\.", text)
    if not m:
        raise SystemExit("[variants] §4 boundary not found; cannot append exemplars")
    return text[:m.start()].rstrip() + "\n" + blocks + "\n---\n\n" + text[m.start():].lstrip("-\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glm_exemplars", default=None,
                    help="glm_exemplars.jsonl — adds an F_glm_exemplars arm "
                         "testing whether long-trace exemplars beat a rule")
    ap.add_argument("--replace_exemplars_from", type=int, default=0,
                    help="Drop existing `### Exemplar N` sections with N >= this "
                         "before appending. The card's short exemplars are what "
                         "set the length anchor a model imitates, so diluting "
                         "them with long ones is weaker than replacing them; "
                         "1–2 are kept because they are correct for the easy "
                         "regime, where long rewrites would be wrong.")
    ap.add_argument("--glm_max_chars", type=int, default=2500,
                    help="skip candidate exemplars larger than this; the card "
                         "rides in the Stage-A teacher context, so size is a "
                         "real recurring cost")
    args = ap.parse_args()

    base = open(args.card).read()
    if not ANCHOR.search(base):
        raise SystemExit("[variants] §1.4 table anchor not found — the card "
                         "changed shape; update ANCHOR before trusting an A/B.")

    variants = dict(VARIANTS)
    glm: list[dict] = []
    if args.glm_exemplars:
        with open(args.glm_exemplars) as f:
            cands = [json.loads(l) for l in f if l.strip()]
        for e in cands:
            if e.get("has_boxed") or e.get("has_fence"):
                print(f"[skip] {e['_uid']}: card §1.5 violation in candidate")
                continue
            if e["exemplar_chars"] > args.glm_max_chars:
                print(f"[skip] {e['_uid']}: {e['exemplar_chars']} chars "
                      f"> --glm_max_chars {args.glm_max_chars}")
                continue
            glm.append(e)
        if not glm:
            raise SystemExit("[variants] no usable GLM exemplars")
        variants["F_glm_exemplars"] = None      # sentinel: appended, not inserted
        print(f"[glm] using {len(glm)} exemplars "
              f"(+{sum(e['exemplar_chars'] for e in glm)} chars)")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {}
    for name, block in variants.items():
        if block is None:
            text = _append_exemplars(
                _drop_exemplars_from(base, args.replace_exemplars_from), glm)
        else:
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
