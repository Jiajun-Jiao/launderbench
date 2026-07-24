"""Stateless mock search endpoint for the gpt-researcher ecological run (data-void threat model).

Serves LaunderBench scenario pages as web search results in gpt-researcher's custom-retriever format
([{"url", "raw_content"}]). The DATA-VOID conditions are implemented at the retrieval layer:

  cond=void       any query returns only the confident echo pages; the weak origin page is crowded
                  out entirely (adversarial/SEO-buried root; retrieval incompleteness as a mechanism).
  cond=reachable  echo pages for any query; the origin page is returned IFF the query names the
                  origin site (deliberate provenance tracing is rewarded — tests whether an agentic
                  loop *chooses* to trace, not whether it could).

Page content is the natural reference-card rendering used in the rendering ablation: page text plus
"References listed by this page: <domain>". No caveat text is added or removed relative to the
canonical scenario pages.

Stateless: scenario id + condition arrive as query params (driver passes them via RETRIEVER_ARG_*,
which gpt-researcher's CustomRetriever forwards on every request), so parallel driver processes can
share one server.

Run: python3 experiments/gptr_mock_search.py   (env: PORT=8377, SCENARIOS=<path>)
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

PORT = int(os.environ.get("PORT", "8377"))
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))


def load_scenarios():
    rows = [json.loads(l) for l in open(SCENARIOS) if l.strip()]
    return {sc["id"]: sc for sc in rows}


BY_ID = load_scenarios()


def page_card(page, by_id):
    target = page.get("cites")
    ref = by_id[target]["domain"] if target and target in by_id else None
    refs_line = f"References listed by this page: {ref}" if ref else "References listed by this page: none"
    return f"{page['text']}\n\n{refs_line}"


def results_for(scenario, cond, query):
    by_id = {p["id"]: p for p in scenario["pages"]}
    origin_id = scenario["origin_id"]
    q = (query or "").lower()
    out = []
    for p in scenario["pages"]:
        is_origin = p["id"] == origin_id
        if is_origin and cond == "void":
            continue
        if is_origin and cond == "reachable" and p["domain"].split(".")[0].lower() not in q:
            continue
        out.append({"url": f"https://{p['domain']}/article", "raw_content": page_card(p, by_id)})
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        sid = (qs.get("scenario") or [""])[0]
        cond = (qs.get("cond") or ["void"])[0]
        query = (qs.get("query") or [""])[0]
        sc = BY_ID.get(sid)
        if parsed.path != "/search" or sc is None or cond not in ("void", "reachable"):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"[]")
            return
        body = json.dumps(results_for(sc, cond, query)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep run logs quiet; driver logs what matters
        pass


if __name__ == "__main__":
    print(f"mock search on 127.0.0.1:{PORT} scenarios={len(BY_ID)} ({os.path.basename(SCENARIOS)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
