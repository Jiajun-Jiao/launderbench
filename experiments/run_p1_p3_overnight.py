#!/usr/bin/env python3
"""Sequential overnight runner for the P1-P3 n=100 expansion plan."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "society"
LOG_DIR = OUT / "overnight_p1_p3_logs"
STATE_PATH = OUT / "overnight_p1_p3_state.jsonl"
SC100 = "outputs/society/citation_scenarios_100_v2.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def load_dotenv() -> dict[str, str]:
    env = os.environ.copy()
    path = ROOT / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            env.setdefault(key.strip(), value)
    if env.get("OPENAI_API_KEY_2"):
        env["OPENAI_API_KEY"] = env["OPENAI_API_KEY_2"]
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


BASE_ENV = load_dotenv()


def rel(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def json_n(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(json.loads(path.read_text()).get("config", {}).get("n_scenarios") or 0)
    except Exception:  # noqa: BLE001
        return None


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def complete_json(path: Path, min_n: int = 100) -> bool:
    n = json_n(path)
    return n is not None and n >= min_n


def complete_json_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 20


def complete_json_scenarios(path: Path, min_rows: int = 100) -> bool:
    return line_count(path) >= min_rows


def complete_handoff(path: Path, carriers: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    return len(data.get("carriers", {})) >= carriers


def append_state(row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("a") as f:
        f.write(json.dumps({"ts": utc_now(), **row}, ensure_ascii=True) + "\n")


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


@dataclass
class Task:
    tier: str
    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path = ROOT
    expected: Path | None = None
    complete: Callable[[Path], bool] = complete_json_exists
    python: str | None = None

    def log_name(self) -> str:
        raw = f"{self.tier}_{self.name}"
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)[:180] + ".log"

    def should_skip(self) -> bool:
        return bool(self.expected and self.complete(self.expected))


def py_task(
    tier: str,
    name: str,
    script: str,
    env: dict[str, str],
    expected: str | None = None,
    complete: Callable[[Path], bool] = complete_json_exists,
    python: str = "python",
    args: list[str] | None = None,
) -> Task:
    return Task(
        tier=tier,
        name=name,
        cmd=[python, script, *(args or [])],
        env=env,
        expected=rel(expected) if expected else None,
        complete=complete,
    )


def run_task(task: Task) -> bool:
    if task.should_skip():
        append_state({"event": "skip", "tier": task.tier, "task": task.name, "expected": str(task.expected)})
        return True
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / task.log_name()
    env = BASE_ENV.copy()
    env.update(task.env)
    append_state(
        {
            "event": "start",
            "tier": task.tier,
            "task": task.name,
            "cmd": quote_cmd(task.cmd),
            "log": str(log_path.relative_to(ROOT)),
            "expected": str(task.expected.relative_to(ROOT)) if task.expected else "",
        }
    )
    with log_path.open("a") as log:
        log.write(f"\n\n===== {utc_now()} START {task.tier} / {task.name} =====\n")
        log.write(f"$ {quote_cmd(task.cmd)}\n")
        log.flush()
        proc = subprocess.run(task.cmd, cwd=task.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"===== {utc_now()} END rc={proc.returncode} =====\n")
    ok = proc.returncode == 0 and (not task.expected or task.complete(task.expected))
    append_state(
        {
            "event": "end",
            "tier": task.tier,
            "task": task.name,
            "returncode": proc.returncode,
            "ok": ok,
            "log": str(log_path.relative_to(ROOT)),
        }
    )
    return ok


def wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def run_gptr_100() -> bool:
    expected = OUT / "framework_gptr_100_gpt-4o-mini.json"
    if complete_json(expected):
        append_state({"event": "skip", "tier": "P2-agentic", "task": "gpt-researcher-100", "expected": str(expected)})
        return True
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server_log = LOG_DIR / "P2-agentic_gptr_mock_server_100.log"
    driver_log = LOG_DIR / "P2-agentic_gpt-researcher_100.log"
    env = BASE_ENV.copy()
    env.update({"SCENARIOS": SC100, "PORT": "8377"})
    append_state({"event": "start", "tier": "P2-agentic", "task": "gpt-researcher-server"})
    with server_log.open("a") as slog:
        server = subprocess.Popen(
            [".venv-gptr/bin/python", "experiments/gptr_mock_search.py"],
            cwd=ROOT,
            env=env,
            stdout=slog,
            stderr=subprocess.STDOUT,
        )
    try:
        if not wait_port(8377):
            append_state({"event": "end", "tier": "P2-agentic", "task": "gpt-researcher-server", "ok": False})
            return False
        task = Task(
            tier="P2-agentic",
            name="gpt-researcher-100",
            cmd=[".venv-gptr/bin/python", "experiments/framework_gptr.py"],
            env={
                "SCENARIOS": SC100,
                "SCENARIO_TAG": "100",
                "CONDS": "void,reachable",
                "MODEL": "gpt-4o-mini",
                "JUDGE_MODEL": "gpt-4.1-mini",
                "ENDPOINT": "http://127.0.0.1:8377/search",
            },
            expected=expected,
            complete=lambda p: complete_json(p, 100),
        )
        ok = run_task(task)
        return ok
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        append_state({"event": "end", "tier": "P2-agentic", "task": "gpt-researcher-server", "ok": True})


def research_task(tier: str, model: str, tag: str = "canonical100", cot: bool = False) -> Task:
    env = {
        "SCENARIOS": SC100,
        "SCENARIO_TAG": tag,
        "CONDS": "trace,flagged,solid",
        "MODEL": model,
        "JUDGE_MODEL": "gpt-4.1-mini",
        "RESUME_ROWS": "1",
    }
    if cot:
        env["COT"] = "1"
    cot_part = "_cot" if cot else ""
    expected = OUT / f"research_citation_multi_{tag}{cot_part}_{safe(model)}.json"
    return py_task(
        tier,
        f"{tag}{cot_part}-{model}",
        "experiments/research_citation_multi.py",
        env,
        expected=str(expected),
        complete=lambda p: complete_json(p, 100),
    )


def tasks() -> list[Task | str]:
    out: list[Task | str] = []

    # P1: open-weight main text.
    out.append(research_task("P1-open", "qwen3:30b-a3b-instruct-2507-q4_K_M"))

    # P1: harness and mechanism.
    for model in ("gpt-4o-mini", "gpt-4.1-mini"):
        out.append(research_task("P1-cot", model, cot=True))
    for model in ("o4-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano"):
        out.append(research_task("P1-cheap-reasoning", model))
    for model, slug in (
        ("gpt-4o-mini", "gpt4omini"),
        ("gpt-4.1-mini", "gpt41mini"),
        ("claude-haiku-4-5", "claude_haiku"),
    ):
        out.append(
            py_task(
                "P1-mechanism",
                f"diagnostic-{model}",
                "experiments/provenance_mechanism_diagnostic.py",
                {
                    "SCENARIOS": SC100,
                    "LIMIT": "0",
                    "MODELS": model,
                    "JUDGE_MODEL": "gpt-4.1-mini",
                    "ARMS": "raw,instruction,diagnose_answer,self_structured,graph,roots,type,license",
                    "OUT": f"outputs/society/provenance_mechanism_activation_canonical100_{slug}.json",
                },
                expected=f"outputs/society/provenance_mechanism_activation_canonical100_{slug}.json",
            )
        )
    out.append(
        py_task(
            "P1-mechanism",
            "compression-pipeline-canonical100",
            "experiments/provenance_pipeline.py",
            {
                "SCENARIOS": SC100,
                "SCENARIO_TAG": "canonical100",
                "LIMIT": "100",
                "MODELS": "gpt-4o-mini,claude-opus-4-8,gpt-5.5,o3",
                "JUDGE_MODEL": "gpt-4.1-mini",
            },
            expected="outputs/society/prov_pipeline_canonical100_o3.json",
        )
    )
    out.append(
        py_task(
            "P1-mechanism",
            "type-once-carry-canonical100",
            "experiments/provenance_handoff.py",
            {
                "SCENARIOS": SC100,
                "SCENARIO_TAG": "canonical100",
                "LIMIT": "100",
                "TYPER": "claude-opus-4-8",
                "CARRIERS": "gpt-4o-mini,gpt-4.1-mini,qwen2.5:32b-instruct,qwen3:30b-a3b-instruct-2507-q4_K_M",
                "JUDGE_MODEL": "gpt-4.1-mini",
                "OUT": "outputs/society/prov_handoff_canonical100.json",
            },
            expected="outputs/society/prov_handoff_canonical100.json",
            complete=lambda p: complete_handoff(p, 4),
        )
    )

    # P1: complete-context framework RAG.
    for model in ("gpt-4o-mini", "claude-haiku-4-5"):
        out.append(
            py_task(
                "P1-framework-complete",
                f"langchain-100-{model}",
                "experiments/framework_langchain.py",
                {
                    "SCENARIOS": SC100,
                    "SCENARIO_TAG": "100",
                    "CONDS": "trace,flagged,solid",
                    "MODEL": model,
                    "JUDGE_MODEL": "gpt-4.1-mini",
                },
                expected=f"outputs/society/framework_langchain_100_{safe(model)}.json",
                complete=lambda p: complete_json(p, 100),
                python=".venv-frameworks/bin/python",
            )
        )
        for fw in ("haystack", "llamaindex"):
            out.append(
                py_task(
                    "P1-framework-complete",
                    f"{fw}-100-{model}",
                    f"experiments/framework_{fw}.py",
                    {
                        "SCENARIOS": SC100,
                        "MODE": "complete",
                        "CONDS": "trace,flagged,solid",
                        "MODEL": model,
                        "JUDGE_MODEL": "gpt-4.1-mini",
                    },
                    expected=f"outputs/society/framework_{fw}_complete_100_{safe(model)}.json",
                    complete=lambda p: complete_json(p, 100),
                    python=".venv-agentfw/bin/python",
                )
            )

    # P2: agentic retrieval.
    for fw in ("haystack", "llamaindex"):
        out.append(
            py_task(
                "P2-agentic",
                f"{fw}-agentic-100-gpt-4o-mini",
                f"experiments/framework_{fw}.py",
                {
                    "SCENARIOS": SC100,
                    "MODE": "agentic",
                    "CONDS": "void,reachable",
                    "MODEL": "gpt-4o-mini",
                    "JUDGE_MODEL": "gpt-4.1-mini",
                },
                expected=f"outputs/society/framework_{fw}_agentic_100_gpt-4o-mini.json",
                complete=lambda p: complete_json(p, 100),
                python=".venv-agentfw/bin/python",
            )
        )
    out.append("RUN_GPTR_100")

    # P2: external seeded and native panels.
    for bench in ("expertqa", "alce", "ragtruth"):
        raw = f"outputs/society/citation_scenarios_{bench}_100.jsonl"
        v2 = f"outputs/society/citation_scenarios_{bench}_100_v2.jsonl"
        out.append(
            py_task(
                "P2-external-generate",
                f"seeded-{bench}-100",
                "experiments/citation_seeded.py",
                {
                    "SEEDS": f"outputs/society/seeds_{bench}.jsonl",
                    "N": "100",
                    "OUT": raw,
                    "ID_PREFIX": "E",
                    "APPEND": "1",
                },
                expected=raw,
                complete=lambda p: complete_json_scenarios(p, 100),
            )
        )
        out.append(
            py_task(
                "P2-external-generate",
                f"solid-{bench}-100",
                "experiments/citation_solid.py",
                {"IN": raw, "OUT": v2, "RESUME": "1", "MAX_TRIES": "3"},
                expected=v2,
                complete=lambda p: complete_json_scenarios(p, 100),
            )
        )
        for model in ("gpt-4o-mini", "claude-opus-4-8"):
            out.append(
                py_task(
                    "P2-external-seeded",
                    f"{bench}100-{model}",
                    "experiments/research_citation_multi.py",
                    {
                        "SCENARIOS": v2,
                        "SCENARIO_TAG": f"{bench}100",
                        "CONDS": "trace,flagged,solid",
                        "MODEL": model,
                        "JUDGE_MODEL": "gpt-4.1-mini",
                        "RESUME_ROWS": "1",
                    },
                    expected=f"outputs/society/research_citation_multi_{bench}100_{safe(model)}.json",
                    complete=lambda p: complete_json(p, 100),
                )
            )
        out.append(
            py_task(
                "P2-external-handoff",
                f"handoff-{bench}100",
                "experiments/provenance_handoff.py",
                {
                    "SCENARIOS": v2,
                    "SCENARIO_TAG": f"{bench}100",
                    "LIMIT": "100",
                    "TYPER": "claude-opus-4-8",
                    "CARRIERS": "gpt-4o-mini",
                    "JUDGE_MODEL": "gpt-4.1-mini",
                    "OUT": f"outputs/society/prov_handoff_{bench}100.json",
                },
                expected=f"outputs/society/prov_handoff_{bench}100.json",
                complete=lambda p: complete_handoff(p, 1),
            )
        )
        native = f"outputs/society/native_cases_{bench}_100.jsonl"
        out.append(
            py_task(
                "P2-external-native",
                f"prepare-native-{bench}100",
                "experiments/external_native_prepare.py",
                {},
                expected=native,
                complete=lambda p: complete_json_scenarios(p, 100),
                args=["--benchmark", bench, "--n", "100", "--output", native],
            )
        )
        for model in ("gpt-4o-mini", "claude-opus-4-8"):
            for render in ("plain", "citation_wrapper"):
                suffix = "" if render == "plain" else "_citation_wrapper"
                out.append(
                    py_task(
                        "P2-external-native",
                        f"native-{bench}100-{render}-{model}",
                        "experiments/external_native_eval.py",
                        {"NATIVE_RENDER_STYLE": render},
                        expected=f"outputs/society/external_native_{bench}100{suffix}_{safe(model)}.json",
                        args=["--cases", native, "--model", model, "--judge-model", "gpt-4.1-mini", "--tag", f"{bench}100"],
                    )
                )

    # P3: smaller open-weight controls.
    for model in ("qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b", "gemma2:9b"):
        out.append(research_task("P3-open-small", model))

    # Summaries.
    out.extend(
        [
            py_task("summary", "build-canonical100-summary", "experiments/build_canonical100_summary.py", {}),
            py_task("summary", "build-agent-framework-summary", "experiments/build_agent_framework_summary.py", {}),
            py_task("summary", "build-external100-summary", "experiments/build_external100_summary.py", {}),
            py_task("summary", "build-open-model-summary", "experiments/open_model_analyze.py", {}),
            py_task("summary", "build-all-results-overview", "experiments/build_all_results_overview.py", {}),
        ]
    )
    return out


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    append_state(
        {
            "event": "run_start",
            "openai_key": "set" if BASE_ENV.get("OPENAI_API_KEY") else "missing",
            "anthropic_key": "set" if BASE_ENV.get("ANTHROPIC_API_KEY") else "missing",
            "google_key": "set" if BASE_ENV.get("GOOGLE_API_KEY") else "missing",
        }
    )
    failures = 0
    for task in tasks():
        if task == "RUN_GPTR_100":
            ok = run_gptr_100()
        else:
            ok = run_task(task)
        if not ok:
            failures += 1
            append_state({"event": "failure_recorded", "failures": failures})
            if os.environ.get("STOP_ON_FAIL", "0") == "1":
                break
    append_state({"event": "run_end", "failures": failures})
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
