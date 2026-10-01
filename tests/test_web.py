import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "index.html"


def payload():
    import re
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', PAGE.read_text(encoding="utf8"), re.S).group(1).replace("<\\/", "</"))


def test_embedded_data_is_complete():
    d = payload()
    if not d:
        pytest.skip("run python -m lab eval first")
    assert len(d["queries"]) == 99 and {"own64", "lsa64", "bm25_full"} <= set(d["docstrings"]["methods"])
    q = d["queries"][0]
    assert q["gold_doc"]["own64"] in d["vectors"]["own64"]["d"] and len(d["vectors"]["own64"]["q"][0]) == 64


def test_page_in_a_browser(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    d = payload()
    if not d:
        pytest.skip("run python -m lab eval first")
    shots = Path(os.environ.get("LAB_SCREENSHOTS", tmp_path))
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"no Chromium available: {str(e)[:80]}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        problems, requests = [], []
        page.on("pageerror", lambda e: problems.append(str(e)))
        page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
        page.on("request", lambda r: requests.append(r.url))
        page.goto(PAGE.as_uri())
        assert page.locator("#chart tbody tr").count() == len(d["docstrings"]["methods"])
        page.click("#tier button:nth-child(2)")
        assert "n=37" in page.inner_text("#chart thead")
        page.screenshot(path=shots / "results.png")
        assert page.locator("tr.q").count() == 99
        page.select_option("#ftier", "high")
        assert page.locator("tr.q").count() == sum(q["tier"] == "high" for q in d["queries"])
        page.select_option("#ftier", "")
        page.click('tr.q[data-i="3"]')
        text = page.inner_text("#maths")
        q = d["queries"][3]
        uq, ud = (np.array(x) for x in (d["vectors"]["own64"]["q"][3], d["vectors"]["own64"]["d"][q["gold_doc"]["own64"]]))
        cosine = float(uq @ ud / np.linalg.norm(uq) / np.linalg.norm(ud))
        assert f"= {cosine:.4f}" in text and q["gold_doc"]["own64"] in text
        page.locator("#detail").screenshot(path=shots / "detail.png")
        page.set_viewport_size({"width": 390, "height": 800})
        assert page.evaluate("document.documentElement.scrollWidth") <= 390
        page.screenshot(path=shots / "mobile.png")
        browser.close()
    assert not problems and all(u.startswith("file:") for u in requests)
