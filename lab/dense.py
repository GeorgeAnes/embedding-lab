"""Dense baselines that are not trained here: a static pretrained vector table (optional) and a real embedding endpoint."""
import json
import os
import urllib.request

import numpy as np

from .lexical import unit

MAX_CHARS = 6000  # documents sent to a real endpoint are cut to this length (about the first 1,500 tokens)


def static_vectors():
    """WordLlama's bundled 256-d static table (pip install wordllama; no download). Returns (name, embed) or None."""
    try:
        from wordllama import WordLlama
    except ImportError:
        return None
    wl = WordLlama.load()
    return "static vectors (WordLlama 256-d), full text", lambda texts: unit(wl.embed(list(texts)).astype(np.float64))


def real_endpoint():
    """An OpenAI-compatible /embeddings endpoint from LAB_EMBED_BASE_URL, LAB_EMBED_MODEL, LAB_EMBED_KEY. Returns (name, embed)."""
    base, model, key = (os.environ.get(v) for v in ("LAB_EMBED_BASE_URL", "LAB_EMBED_MODEL", "LAB_EMBED_KEY"))
    if not (base and model):
        raise SystemExit("Set LAB_EMBED_BASE_URL and LAB_EMBED_MODEL (and LAB_EMBED_KEY if the server needs one).")

    def embed(texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            body = json.dumps({"model": model, "input": [t[:MAX_CHARS] for t in texts[i:i + batch]]}).encode()
            req = urllib.request.Request(base.rstrip("/") + "/embeddings", body, {"Content-Type": "application/json"})
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)["data"]
            out += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        return unit(np.array(out, dtype=np.float64))

    return f"real model {model}, full text", embed
