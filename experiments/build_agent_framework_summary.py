#!/usr/bin/env python3
"""Summarize framework and agentic ecological-validity runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def row_from_file(path: Path) -> dict | None:
    data = load(path)
    if not data:
        return None
    cfg = data["config"]
    res = data["results"]
    fw = cfg.get("framework")
    mode = cfg.get("mode") or "complete"
    if fw == "gpt-researcher":
        mode = "agentic"
    n = cfg.get("n_scenarios")
    label = f"{fw} {mode} n={n} {cfg.get('model')}"
    row = {"label": label, "file": str(path.relative_to(ROOT)), "n": n}
    for cond in ("trace", "flagged", "solid", "void", "reachable"):
        if cond in res:
            row[cond] = res[cond]["launder_n"]
            row[f"{cond}_rate"] = res[cond]["launder"]
            if "origin_page_retrieved" in res[cond]:
                row[f"{cond}_origin"] = res[cond]["origin_page_retrieved"]
    rows = data.get("rows") or []
    for cond in ("void", "reachable"):
        sub = [r for r in rows if r.get("cond") == cond]
        if sub:
            row[f"{cond}_n"] = len(sub)
            row[f"{cond}_tool_called"] = sum(bool(r.get("tool_calls")) for r in sub)
            row[f"{cond}_origin_retrieved_rows"] = sum(bool(r.get("origin_page_retrieved")) for r in sub)
    return row


def discover() -> list[dict]:
    rows = []
    for path in sorted(OUTDIR.glob("framework_*.json")):
        row = row_from_file(path)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (str(r["n"]), r["label"]))
    return rows


def write_md(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Agent/framework extension summary",
        "",
        "Complete mode means the full weak-origin page is already in context. Agentic mode means the origin must be actively retrieved.",
        "",
        "| framework / mode | n | trace | flagged | solid | void | reachable | search tool called | origin actually retrieved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        search = "-"
        origin = "-"
        if "void_tool_called" in r:
            search = f"void {r['void_tool_called']}/{r['void_n']}; reach {r.get('reachable_tool_called', 0)}/{r.get('reachable_n', 0)}"
            origin = f"void {r['void_origin_retrieved_rows']}/{r['void_n']}; reach {r.get('reachable_origin_retrieved_rows', 0)}/{r.get('reachable_n', 0)}"
        elif "void_origin" in r:
            origin = f"void {r.get('void_origin', 0)}/{r.get('n')}; reach {r.get('reachable_origin', 0)}/{r.get('n')}"
        lines.append(
            f"| {r['label']} | {r.get('n', '-')} | {r.get('trace', '-')} | {r.get('flagged', '-')} | "
            f"{r.get('solid', '-')} | {r.get('void', '-')} | {r.get('reachable', '-')} | {search} | {origin} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = discover()
    json_path = OUTDIR / "agent_framework_extension_summary.json"
    md_path = OUTDIR / "agent_framework_extension_summary.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    write_md(md_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
