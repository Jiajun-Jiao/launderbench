#!/usr/bin/env python3
"""Cheap-only continuation runner for the $50 GPT budget."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "society"
LOG_DIR = OUT / "budget50_cheap_logs"
STATE = OUT / "budget50_cheap_state.jsonl"
SC100 = "outputs/society/citation_scenarios_100_v2.jsonl"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def load_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if env.get("OPENAI_API_KEY_2"):
        env["OPENAI_API_KEY"] = env["OPENAI_API_KEY_2"]
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


BASE_ENV = load_env()


def write_state(row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATE.open("a") as f:
        f.write(json.dumps({"ts": utc(), **row}) + "\n")


def json_done(path: Path, min_n: int = 100) -> bool:
    if not path.exists():
        return False
    try:
        return int(json.loads(path.read_text()).get("config", {}).get("n_scenarios") or 0) >= min_n
    except Exception:  # noqa: BLE001
        return False


def file_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 20


@dataclass
class Task:
    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    expected: Path | None = None
    complete: Callable[[Path], bool] = file_done
    cwd: Path = ROOT

    @property
    def log_path(self) -> Path:
        clean = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.name)
        return LOG_DIR / f"{clean[:180]}.log"


def run(task: Task) -> bool:
    if task.expected and task.complete(task.expected):
        write_state({"event": "skip", "task": task.name, "expected": str(task.expected.relative_to(ROOT))})
        return True
    env = BASE_ENV.copy()
    env.update(task.env)
    write_state({"event": "start", "task": task.name, "cmd": " ".join(shlex.quote(x) for x in task.cmd)})
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with task.log_path.open("a") as log:
        log.write(f"\n\n===== {utc()} START {task.name} =====\n")
        proc = subprocess.run(task.cmd, cwd=task.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"===== {utc()} END rc={proc.returncode} =====\n")
    ok = proc.returncode == 0 and (not task.expected or task.complete(task.expected))
    write_state({"event": "end", "task": task.name, "ok": ok, "returncode": proc.returncode, "log": str(task.log_path.relative_to(ROOT))})
    return ok


def research_task(model: str, cot: bool = False) -> Task:
    tag = "canonical100"
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
    return Task(
        name=f"research-{tag}{cot_part}-{model}",
        cmd=["python", "experiments/research_citation_multi.py"],
        env=env,
        expected=OUT / f"research_citation_multi_{tag}{cot_part}_{safe(model)}.json",
        complete=lambda p: json_done(p, 100),
    )


def framework_task(framework: str) -> Task:
    py = ".venv-frameworks/bin/python" if framework == "langchain" else ".venv-agentfw/bin/python"
    env = {
        "SCENARIOS": SC100,
        "SCENARIO_TAG": "100",
        "CONDS": "trace,flagged,solid",
        "MODEL": "gpt-4o-mini",
        "JUDGE_MODEL": "gpt-4.1-mini",
    }
    if framework in {"haystack", "llamaindex"}:
        env["MODE"] = "complete"
    return Task(
        name=f"framework-complete-100-{framework}-gpt-4o-mini",
        cmd=[py, f"experiments/framework_{framework}.py"],
        env=env,
        expected=OUT / f"framework_{framework}_{'complete_' if framework != 'langchain' else ''}100_gpt-4o-mini.json",
        complete=lambda p: json_done(p, 100),
    )


def tasks() -> list[Task]:
    out: list[Task] = []
    # Local/open-weight: answer calls are free; only the gpt-4.1-mini judge is paid.
    for model in ("qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b", "gemma2:9b"):
        out.append(research_task(model))
    # Cheap paid reasoning controls. Intentionally skip o4-mini, o3, gpt-5.5, and all Claude/Opus generation.
    for model in ("gpt-5.4-nano", "gpt-5-mini", "gpt-5.4-mini"):
        out.append(research_task(model))
    # Cheap ecological complete-context framework checks with gpt-4o-mini only.
    for framework in ("langchain", "haystack", "llamaindex"):
        out.append(framework_task(framework))
    # Summaries only.
    out.extend(
        [
            Task("summary-canonical100", ["python", "experiments/build_canonical100_summary.py"]),
            Task("summary-framework", ["python", "experiments/build_agent_framework_summary.py"]),
            Task("summary-open-model", ["python", "experiments/open_model_analyze.py"]),
            Task("summary-overview", ["python", "experiments/build_all_results_overview.py"]),
        ]
    )
    return out


def main() -> int:
    write_state(
        {
            "event": "run_start",
            "policy": "budget50 cheap-only; no Claude Opus, no GPT-5.5, no o3/o4-mini continuation, no external generation",
            "openai_key": "set" if BASE_ENV.get("OPENAI_API_KEY") else "missing",
        }
    )
    failures = 0
    for task in tasks():
        if not run(task):
            failures += 1
    write_state({"event": "run_end", "failures": failures})
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
