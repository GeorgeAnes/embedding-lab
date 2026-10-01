"""Build data/corpus.jsonl from the docstrings of the installed NumPy and SciPy.

One record per public function, class or ufunc: id, summary (first paragraph of the docstring),
full (summary plus the rest of the docstring, Examples sections removed). The committed
data/corpus.jsonl is what every reported number used; regenerating it on other NumPy or SciPy
versions gives a different corpus and different numbers.
"""
import importlib, inspect, json, re, sys
from pathlib import Path

import numpy, scipy

MODULES = ["numpy", "numpy.linalg", "numpy.fft", "numpy.random", "numpy.polynomial", "scipy.optimize",
           "scipy.integrate", "scipy.interpolate", "scipy.signal", "scipy.linalg", "scipy.stats", "scipy.special",
           "scipy.fft", "scipy.sparse", "scipy.sparse.linalg", "scipy.spatial", "scipy.spatial.distance",
           "scipy.cluster.vq", "scipy.cluster.hierarchy", "scipy.ndimage", "scipy.odr", "scipy.stats.qmc",
           "scipy.differentiate"]
HEADER = r"\n(?:Parameters|Returns|Yields|Raises|See Also|Notes|Examples|References|Attributes|Methods|Warns|Other Parameters)\n-{3,}\n"
PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"


def split_docstring(doc):
    """Return (summary, full) or None. A leading signature line (ufuncs) is skipped."""
    paras = [p for p in re.split(r"\n\s*\n", inspect.cleandoc(doc)) if p.strip()]
    if paras and re.match(r"^\w+\(.*\)\s*(->.*)?$", paras[0].strip().split("\n")[0]) and len(paras) > 1:
        paras = paras[1:]
    if not paras:
        return None
    body = "\n\n".join(paras[1:])
    # Drop every Examples section (doctest code is not prose), keep the other sections.
    sections = re.split(f"({HEADER})", "\n" + body)
    kept, skip = [sections[0]], False
    for part in sections[1:]:
        if re.fullmatch(HEADER, part):
            skip = part.strip().startswith("Examples")
        elif not skip:
            kept.append(part)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    summary, rest = norm(paras[0]), norm(" ".join(kept))
    if len(summary.split()) < 2 or len(rest.split()) < 10:
        return None
    return summary, f"{summary} {rest}"


def build():
    out, seen = [], set()
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]):
            obj = getattr(mod, name, None)
            if not (inspect.isroutine(obj) or inspect.isclass(obj) or isinstance(obj, numpy.ufunc)) or not obj.__doc__:
                continue
            if getattr(obj, "__name__", name) != name or id(obj) in seen:  # skip aliases; the canonical name is listed
                continue
            seen.add(id(obj))
            parts = split_docstring(obj.__doc__)
            if parts and all(parts[1] != d["full"] for d in out):
                out.append({"id": f"{modname}.{name}", "summary": parts[0], "full": parts[1]})
    return out


def write():
    docs = build()
    PATH.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf8")
    print(f"wrote {len(docs)} docs to {PATH} (numpy {numpy.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]})")


if __name__ == "__main__":
    write()
