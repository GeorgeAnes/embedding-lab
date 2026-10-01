import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from lab import dense


class Stub(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, self.headers.get("Authorization"), body))
        data = [{"index": i, "embedding": [float(len(t)), 1.0, float(i)]} for i, t in enumerate(body["input"])]
        out = json.dumps({"data": data[::-1]}).encode()  # returned out of order on purpose
        self.send_response(200), self.send_header("Content-Length", str(len(out))), self.end_headers(), self.wfile.write(out)

    def log_message(self, *args):
        pass


def test_openai_compatible_client_against_a_local_stub(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LAB_EMBED_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1/")
    monkeypatch.setenv("LAB_EMBED_MODEL", "stub-model")
    monkeypatch.setenv("LAB_EMBED_KEY", "secret")
    name, embed = dense.real_endpoint()
    vecs = embed(["a", "bbb"] * 20)  # 40 texts, batches of 32
    server.shutdown()
    assert name.startswith("real model stub-model") and vecs.shape == (40, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1) and vecs[1, 0] > vecs[0, 0]  # order restored: text 1 is longer
    assert {p for p, _, _ in Stub.seen} == {"/v1/embeddings"} and {a for _, a, _ in Stub.seen} == {"Bearer secret"}
    assert all(b["model"] == "stub-model" for _, _, b in Stub.seen) and len(Stub.seen) == 2


def test_missing_settings_stop_with_a_message(monkeypatch):
    monkeypatch.delenv("LAB_EMBED_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        dense.real_endpoint()
