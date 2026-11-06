"""Print measured vs allotted height for every text item, and column slack."""
import json, string, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import poster_spec as spec
from richtext import needed_height

BOX_PAD = 0.35  # must match builders


class _Safe(dict):
    def __missing__(self, k):
        return 0


def fill(text, facts):
    try:
        return text.format_map(facts)
    except (KeyError, ValueError, IndexError):
        return string.Formatter().vformat(text, (), _Safe(facts))


def main():
    f = spec.DATA / "facts.json"
    facts = json.loads(f.read_text()) if f.exists() else {}
    els = spec.layout()
    for e in els:
        if e.kind not in ("text", "box", "header") or not e.text:
            continue
        pad = BOX_PAD if e.kind == "box" else 0
        need = needed_height(fill(e.text, facts), e.w - 2 * pad, e.size) + 2 * pad
        flag = "OVER" if need > e.h + 1e-3 else ""
        print(f"x={e.x:5.1f} y={e.y:5.1f} h={e.h:5.2f} need={need:5.2f} {flag:4} {e.text[:50]!r}")
    for c, o in spec.OVERFLOW.items():
        print(f"column {c+1}: {'OVERFLOW' if o > 0 else 'slack'} {abs(o):.2f} in")


if __name__ == "__main__":
    main()
