#!/usr/bin/env python3
"""Minimal repro: boot server in-process, fire one streaming /api/ask,
print each NDJSON line with its arrival time so we can see WHERE it stalls."""
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

PORT = 8891
httpd = server.ThreadingHTTPServer(("127.0.0.1", PORT), server.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)
print("server up", flush=True)

conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=180)
t0 = time.time()
conn.request("POST", "/api/ask",
             body=json.dumps({"text": "Give me three short tips for better sleep.",
                              "session": "debug-" + os.urandom(3).hex(),
                              "stream": True}),
             headers={"Content-Type": "application/json"})
print(f"[{time.time()-t0:5.1f}s] request sent, awaiting headers", flush=True)
resp = conn.getresponse()
print(f"[{time.time()-t0:5.1f}s] got status {resp.status} ctype={resp.getheader('Content-Type')}", flush=True)
try:
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if line:
            msg = json.loads(line)
            keep = {k: (v if k != "audio" else f"<{len(v)} b64 chars>") for k, v in msg.items()}
            print(f"[{time.time()-t0:5.1f}s] {json.dumps(keep)[:200]}", flush=True)
except Exception as e:
    print(f"[{time.time()-t0:5.1f}s] EXC: {type(e).__name__}: {e}", flush=True)
conn.close()
httpd.shutdown()
print("done", flush=True)
