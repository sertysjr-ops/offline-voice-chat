#!/usr/bin/env python3
"""Hang hunter: same as debug_stream but dumps ALL thread tracebacks
15s after the last stream line arrives, so we can see exactly where
the server thread is stuck."""
import faulthandler
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

PORT = 8892
httpd = server.ThreadingHTTPServer(("127.0.0.1", PORT), server.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)
print("server up", flush=True)

last_line = time.time()
dumped = False


def watchdog():
    global dumped
    while True:
        time.sleep(1)
        if not dumped and time.time() - last_line > 15:
            dumped = True
            print("\n===== 15s SINCE LAST LINE — THREAD DUMP =====", flush=True)
            faulthandler.dump_traceback()
            print("===== END DUMP =====\n", flush=True)
            return


threading.Thread(target=watchdog, daemon=True).start()

conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
t0 = time.time()
conn.request("POST", "/api/ask",
             body=json.dumps({"text": "Give me three short tips for better sleep.",
                              "session": "debug-" + os.urandom(3).hex(),
                              "stream": True}),
             headers={"Content-Type": "application/json"})
print(f"[{time.time()-t0:5.1f}s] request sent", flush=True)
resp = conn.getresponse()
print(f"[{time.time()-t0:5.1f}s] status {resp.status}", flush=True)
try:
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if line:
            last_line = time.time()
            msg = json.loads(line)
            keep = {k: (v if k != "audio" else f"<{len(v)} b64>") for k, v in msg.items()}
            print(f"[{time.time()-t0:5.1f}s] {json.dumps(keep)[:180]}", flush=True)
except Exception as e:
    print(f"[{time.time()-t0:5.1f}s] EXC: {type(e).__name__}: {e}", flush=True)
conn.close()
httpd.shutdown()
print("done", flush=True)
