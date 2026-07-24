#!/usr/bin/env python3
"""Scenario construct-validity audit server (stdlib only) — a sibling of web/app/server.py.

Serves the SCENARIO audit (items type "S": annotators verify citation-graph construct validity of
canonical LaunderBench scenarios), on its own port with its own annotators and annotation store, so
the frozen answer-label audit in web/app/ is untouched. Sit behind nginx:
proxy /scenaudit/ -> http://127.0.0.1:8093/.
"""
import base64
import hashlib
import hmac
import http.server
import json
import os
import socketserver
import time
import urllib.parse
from http import cookies

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8093"))
BASE_PATH = os.environ.get("BASE_PATH", "/scenaudit/")
ITEMS_PATH = os.environ.get("ITEMS_PATH", os.path.normpath(
    os.path.join(HERE, "..", "..", "outputs", "human_eval", "scenario_audit_items.jsonl")))
ANNOT_JSON = os.environ.get("ANNOT_JSON", os.path.join(HERE, "annotators.json"))
ANNOT_DIR = os.environ.get("ANNOT_DIR", os.path.join(HERE, "annotations"))
SECRET_FILE = os.path.join(HERE, "secret")
SESS_TTL = 30 * 24 * 3600
SECURE_COOKIE = os.environ.get("COOKIE_SECURE", "1") == "1"


def is_done(it_type, rec):
    if not rec:
        return False
    if it_type == "S":
        core = all(rec.get(k) in ("Y", "N") for k in
                   ("q_single_origin", "q_origin_weak", "q_leak", "q_tell"))
        solid_ok = rec.get("q_solid_ok") in ("Y", "N", "NA")
        return core and bool(rec.get("q_origin_pick")) and solid_ok
    return False


os.makedirs(ANNOT_DIR, exist_ok=True)
if not os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, "wb") as f:
        f.write(os.urandom(32))
    os.chmod(SECRET_FILE, 0o600)
SECRET = open(SECRET_FILE, "rb").read()


def load_items():
    d = {}
    if os.path.exists(ITEMS_PATH):
        for line in open(ITEMS_PATH):
            line = line.strip()
            if line:
                it = json.loads(line)
                d[it["id"]] = it
    return d


ITEMS = load_items()
ANNOTATORS = json.load(open(ANNOT_JSON)) if os.path.exists(ANNOT_JSON) else {}


def pw_ok(user, pw):
    a = ANNOTATORS.get(user)
    if not a:
        return False
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(a["salt"]), 120000).hex()
    return hmac.compare_digest(h, a["hash"])


def sign(user):
    msg = f"{user}|{int(time.time()) + SESS_TTL}".encode()
    sig = hmac.new(SECRET, msg, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(msg).decode() + "." + sig


def verify(tok):
    try:
        b64, sig = tok.split(".")
        msg = base64.urlsafe_b64decode(b64)
        if not hmac.compare_digest(sig, hmac.new(SECRET, msg, hashlib.sha256).hexdigest()):
            return None
        user, exp = msg.decode().split("|")
        if int(exp) < time.time() or user not in ANNOTATORS:
            return None
        return user
    except Exception:
        return None


def ann_path(user):
    return os.path.join(ANNOT_DIR, user + ".json")


def load_ann(user):
    p = ann_path(user)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_ann(user, data):
    tmp = ann_path(user) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, ann_path(user))


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else (
            json.dumps(body).encode() if ctype == "application/json" else body.encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _user(self):
        c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return verify(c["sess"].value) if "sess" in c else None

    def _json_body(self):
        n = int(self.headers.get("Content-Length", "0"))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
            return
        if path == "/api/me":
            u = self._user()
            self._send(200 if u else 401, {"user": u} if u else {"error": "not logged in"})
            return
        if path == "/api/items":
            u = self._user()
            if not u:
                self._send(401, {"error": "not logged in"})
                return
            assigned = ANNOTATORS[u].get("items", [])
            saved = load_ann(u)
            out = [{**ITEMS[i], "saved": saved.get(i, {})} for i in assigned if i in ITEMS]
            acked = bool(saved.get("_ack", {}).get("agreed"))
            self._send(200, {"user": u, "items": out, "acked": acked})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/login":
            d = self._json_body()
            u = (d.get("user") or "").strip()
            if pw_ok(u, d.get("password") or ""):
                ck = (f"sess={sign(u)}; Path={BASE_PATH}; HttpOnly; "
                      f"{'Secure; ' if SECURE_COOKIE else ''}SameSite=Lax; Max-Age={SESS_TTL}")
                self._send(200, {"ok": True, "user": u}, extra=[("Set-Cookie", ck)])
            else:
                self._send(401, {"ok": False, "error": "wrong user or password"})
            return
        if path == "/api/logout":
            self._send(200, {"ok": True}, extra=[("Set-Cookie", f"sess=; Path={BASE_PATH}; HttpOnly; Max-Age=0")])
            return
        if path == "/api/ack":
            u = self._user()
            if not u:
                self._send(401, {"error": "not logged in"})
                return
            data = load_ann(u)
            data["_ack"] = {"ts": int(time.time()), "agreed": True}
            save_ann(u, data)
            self._send(200, {"ok": True})
            return
        if path == "/api/save":
            u = self._user()
            if not u:
                self._send(401, {"error": "not logged in"})
                return
            d = self._json_body()
            iid = d.get("item_id")
            if iid not in ANNOTATORS[u].get("items", []):
                self._send(403, {"error": "not your item"})
                return
            data = load_ann(u)
            rec = dict(d.get("answers") or {})
            rec["telemetry"] = d.get("telemetry") or {}
            rec["ts"] = int(time.time())
            data[iid] = rec
            save_ann(u, data)
            done = sum(1 for k, v in data.items()
                       if not k.startswith("_") and is_done(ITEMS.get(k, {}).get("type"), v))
            total = sum(1 for i in ANNOTATORS[u]["items"] if i in ITEMS)
            self._send(200, {"ok": True, "done": done, "total": total})
            return
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H) as srv:
        print(f"serving 127.0.0.1:{PORT}  items={len(ITEMS)}  annotators={list(ANNOTATORS)}", flush=True)
        srv.serve_forever()
