#!/usr/bin/env python3
"""
Dynamic reasoning-token logit bias proxy for reasoning models.

Sits between client and llama-server. Injects logit_bias on the </think>
token to force reasoning models to exit thinking and commit to an answer.

Stdlib-only (no FastAPI/uvicorn needed). Uses http.server + http.client.

Usage:
  # No bias (baseline)
  python3 think_bias_proxy.py --port 8082 --upstream-port 8081 --mode none

  # Static bias (constant +3.0 on </think> from token 0)
  python3 think_bias_proxy.py --port 8082 --upstream-port 8081 --mode static --static-bias 3.0

  # Dynamic bias (ramp from 0 after 1500 tokens, +5 per 2000 tokens, max 30)
  python3 think_bias_proxy.py --port 8082 --upstream-port 8081 --mode dynamic --threshold 1500 --ramp-interval 2000 --bias-step 5.0 --max-bias 30.0

Token IDs (VibeThinker-3B / Qwen2.5 tokenizer):
  <think>    = 151665
  </think>   = 151666
  <|endoftext|> = 151643

Findings (see references/vibethinker-3b-research.md):
  - Static +3.0: 30% fewer reasoning tokens, zero accuracy loss on solvable problems
  - Static +1.0: no measurable improvement over baseline
  - Dynamic +5-30: catastrophically breaks generation (model exits thinking too early)
  - Limitation: llama-server applies logit_bias at request time, not per-token.
    True dynamic biasing requires custom inference loop.
"""

import argparse
import json
import time
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler

THINK_END = 151666  # </think>

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8081
MODE = "dynamic"
MAX_BIAS = 30.0
RAMP_INTERVAL = 2000
BIAS_STEP = 5.0
STATIC_BIAS = 10.0
THRESHOLD = 1500
DEBUG = True

def log(msg):
    if DEBUG:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

def compute_bias(max_tokens):
    if MODE == "none":
        return 0.0
    if MODE == "static":
        return STATIC_BIAS
    # dynamic
    if max_tokens <= THRESHOLD:
        return 0.0
    excess = max_tokens - THRESHOLD
    steps = excess // RAMP_INTERVAL
    return min(MAX_BIAS, (steps + 1) * BIAS_STEP)

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True, "mode": MODE,
                "upstream": f"{UPSTREAM_HOST}:{UPSTREAM_PORT}"
            }).encode())
            return
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=30)
        try:
            conn.request("GET", self.path)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(data)
        finally:
            conn.close()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length) if length > 0 else b""

        is_completion = "completions" in self.path
        if not is_completion:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=300)
            try:
                conn.request("POST", self.path, body_raw, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(data)
            finally:
                conn.close()
            return

        try:
            body = json.loads(body_raw)
        except:
            body = {}

        max_tokens = body.get("max_tokens", 4096)
        was_stream = body.get("stream", False)

        bias = compute_bias(max_tokens)
        if bias > 0:
            body["logit_bias"] = {str(THINK_END): bias}
            body["stream"] = False
            log(f"Applied bias={bias:.1f} to </think> (max_tokens={max_tokens}, mode={MODE})")
        else:
            log(f"No bias (max_tokens={max_tokens}, mode={MODE})")
            body["stream"] = False

        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=600)
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                        {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
        finally:
            conn.close()

        result = json.loads(data)
        choice = result.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish = choice.get("finish_reason", "?")
        usage = result.get("usage", {})
        tokens = usage.get("completion_tokens", 0)

        has_think = "<think>" in content
        has_end = "</think>" in content
        reasoning_chars = 0
        if has_think and has_end:
            ts = content.index("<think>") + 7
            te = content.index("</think>")
            reasoning_chars = te - ts

        log(f"Response: finish={finish} tokens={tokens} content_len={len(content)} "
            f"has_think={has_think} has_end={has_end} reasoning_chars={reasoning_chars}")

        if was_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk = {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning model think-bias proxy")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--upstream-port", type=int, default=8081)
    parser.add_argument("--mode", choices=["none", "static", "dynamic"], default="dynamic")
    parser.add_argument("--static-bias", type=float, default=10.0)
    parser.add_argument("--max-bias", type=float, default=30.0)
    parser.add_argument("--threshold", type=int, default=1500)
    parser.add_argument("--ramp-interval", type=int, default=2000)
    parser.add_argument("--bias-step", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    UPSTREAM_PORT = args.upstream_port
    MODE = args.mode
    STATIC_BIAS = args.static_bias
    MAX_BIAS = args.max_bias
    THRESHOLD = args.threshold
    RAMP_INTERVAL = args.ramp_interval
    BIAS_STEP = args.bias_step
    DEBUG = not args.quiet

    log(f"Proxy on :{args.port} -> upstream :{UPSTREAM_PORT}")
    log(f"  Mode: {MODE}, </think> token: {THINK_END}")
    if MODE == "dynamic":
        log(f"  Threshold: {THRESHOLD}, Ramp: +{BIAS_STEP}/{RAMP_INTERVAL}tok, Max: {MAX_BIAS}")
    elif MODE == "static":
        log(f"  Static bias: {STATIC_BIAS}")

    server = HTTPServer(("0.0.0.0", args.port), ProxyHandler)
    server.timeout = None
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()
