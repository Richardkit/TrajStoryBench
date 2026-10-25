"""Two model responses surrounding exactly one complete program execution."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import PROTOCOL
from .prompts import SYSTEM, CODE_INSTRUCTIONS, ANSWER_INSTRUCTIONS

MODELS = {
    "gpt": ("gpt-6-luna", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "qwen": ("qwen3.8-flash", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    request_limit: int = 83
    code_timeout: int = 180
    observation_chars: int = 60000
    max_completion_tokens: int = 16000

    def __post_init__(self):
        if self.request_limit < 2:
            raise ValueError("CodeGen requires room for two model requests")
        if min(self.code_timeout, self.max_completion_tokens) < 1 or self.observation_chars < 100:
            raise ValueError("Invalid execution/output limit")


def parse_object(text):
    """Accept plain/fenced JSON only; no content repair or semantic extraction."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def parse_code(text):
    value = parse_object(text)
    if set(value) != {"code"} or not isinstance(value["code"], str) or not value["code"].strip():
        raise ValueError("Expected exactly one nonempty code string")
    return value["code"]


def parse_answer(text):
    value = parse_object(text)
    if set(value) != {"insights", "summary"} or not isinstance(value["insights"], list):
        raise ValueError("Expected insights list and summary only")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("Missing nonempty summary")
    if any(not isinstance(s, str) or not s.strip() or len(s.split()) > 180 for s in value["insights"]):
        raise ValueError("Insights must be nonempty strings of at most 180 words")
    return value


def request(model_key, messages, config):
    body = dict(model=MODELS[model_key][0], messages=messages, temperature=0,
                n=1, response_format={"type": "json_object"})
    if model_key == "gpt":
        body.update(reasoning_effort="none", max_completion_tokens=config.max_completion_tokens)
    else:
        body.update(max_tokens=config.max_completion_tokens, extra_body={"enable_thinking": False})
    return body


def prepare(root, case_id, model_key, out, config, offline=False):
    from baseline_support.case_adapter import load_trajstory_case

    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Use a fresh empty output directory")
    catalog = root / "trajstory_examples/insight/manifest.json"
    matches = [c for c in read(catalog)["cases"] if c["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate case ID")
    case = matches[0]
    case_path = root / case["json_path"]
    bundle = load_trajstory_case(case_path, root, input_csvs=True)
    tables, sources = [], []
    for index, item in enumerate(bundle.files, 1):
        source = item.path.resolve()
        if not source.is_relative_to(root):
            raise ValueError("Input must be inside the benchmark repository")
        if not source.name.startswith("input_") or source.suffix.lower() != ".csv":
            raise ValueError("Only strict projected input_ CSVs are allowed")
        with source.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        name = f"{index:02d}_{source.name}"
        tables.append(dict(file_id=item.file_id, role=item.role, path="input/" + name, columns=columns))
        sources.append(dict(source=str(source.relative_to(root)), staged_path="input/" + name, sha256=sha(source)))
    if not sources:
        raise ValueError("No input tables")
    v5_path = root / "results/v5_result" / model_key / "experiment/manifest.json"
    parity = None
    if v5_path.exists():
        frozen = {k.replace("\\", "/"): v for k, v in read(v5_path)["input_sha256"].items()}
        for item in sources:
            if frozen.get(item["source"].replace("\\", "/")) != item["sha256"]:
                raise ValueError("Projected input differs from frozen v5 input")
        parity = dict(manifest=str(v5_path), sha256=sha(v5_path), verified=True)
    model_input = dict(goal=bundle.goal, dataset_description=bundle.context,
                       analyst_role=bundle.role, tables=tables)
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=(
        CODE_INSTRUCTIONS + "\nACTUAL TASK:\n" + json.dumps(model_input, ensure_ascii=False)))]
    files = [*sorted((root / "baselines/codegen_baseline").glob("*.py")),
             root / "baselines/scripts/run_codegen_baseline.py", root / "baselines/scripts/evaluate_codegen_baseline.py",
             *(root / "baselines/baseline_support" / n for n in ("case_adapter.py", "input_paths.py", "models.py"))]
    from baseline_support.runtime import source_files as support_sources
    files.extend(support_sources())
    manifest = dict(protocol=PROTOCOL, created_at=now(), case_id=case_id, json_path=case["json_path"],
        generator=model_key, model=MODELS[model_key][0], validation_mode="offline_fake" if offline else "benchmark",
        config=config.__dict__, request_limit=config.request_limit, planned_model_responses=2,
        code_execution_limit=1, whole_case_attempts=1, code_repair=False, format_repair=False,
        fixed_insight_quota=False, projected_input_csvs=True, v5_input_parity=parity,
        network_failure_limit=5, network_failures_consume_budget=False,
        input_sources=sources, input_sha256={s["source"]: s["sha256"] for s in sources},
        case_json_sha256=sha(case_path), model_input_sha256=hashlib.sha256(
            json.dumps(model_input, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        source_sha256={str(p.relative_to(root)): sha(p) for p in files},
        upstream="https://github.com/allenai/discoverybench/blob/main/agents/coder_utils.py",
        adaptation="Two explicit JSON responses; script stdout; independent synthetic code example; multiple insights and summary. No original DiscoveryBench metadata/hypothesis hints.",
        execution_isolation="Separate credential-free process, staged inputs and Python audit guard; not an OS sandbox",
        judge="Separate two independent gpt-6-luna rounds; temperature 0; reasoning none")
    write(out / "manifest.json", manifest)
    write(out / "model_input.json", model_input)
    write(out / "code_request.json", request(model_key, messages, config))
    for source in files:
        dest = out / "source_snapshot" / source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    write(out / "status.json", dict(state="prepared", protocol=PROTOCOL, eligible_for_scoring=False))
    return manifest


class ConsecutiveNetworkFailures(RuntimeError):
    pass


class ModelSession:
    def __init__(self, client, out, config, retry_delay=1):
        self.client, self.out, self.config = client, Path(out), config
        self.retry_delay = retry_delay
        self.requests_used = self.network_failures = self.streak = 0
        self.attempts, self.responses = [], []

    def summary(self):
        known = [r["usage"] for r in self.responses if r.get("usage")]
        usage_expected = sum(a.get("state") != "network_error" for a in self.attempts)
        return dict(requests_used=self.requests_used, request_limit=self.config.request_limit,
                    network_failed_attempts=self.network_failures, consecutive_network_failures=self.streak,
                    model_calls_started=len(self.attempts), successful_responses=len(self.responses),
                    calls_with_usage=len(known), missing_usage_calls=usage_expected - len(known),
                    complete=bool(self.responses) and len(known) == usage_expected,
                    token_usage={k: sum(u.get(k, 0) for u in known) for k in
                                 ("prompt_tokens", "completion_tokens", "total_tokens")})

    def save(self):
        write(self.out / "model_usage.json", dict(summary=self.summary(), attempts=self.attempts))

    def invoke(self, phase, body):
        from openai import APIConnectionError, APITimeoutError

        if self.requests_used >= self.config.request_limit:
            raise RuntimeError("Model request budget exhausted")
        write(self.out / (phase + "_request.json"), body)
        while True:
            row = dict(phase=phase, attempt=len(self.attempts) + 1, started_at=now())
            self.attempts.append(row)
            self.save()
            try:
                response = self.client.chat.completions.create(**body)
                response = response.model_dump() if hasattr(response, "model_dump") else response
            except (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError) as exc:
                self.network_failures += 1
                self.streak += 1
                row.update(state="network_error", error_type=type(exc).__name__, ended_at=now())
                self.save()
                if self.streak >= 5:
                    raise ConsecutiveNetworkFailures("Five consecutive network failures") from None
                time.sleep(self.retry_delay)
                continue
            except Exception as exc:
                self.requests_used += 1
                self.streak = 0
                row.update(state="provider_error", error_type=type(exc).__name__, ended_at=now())
                self.save()
                raise
            self.requests_used += 1
            self.streak = 0
            self.responses.append(response)
            row.update(state="returned", ended_at=now(), usage=response.get("usage"))
            write(self.out / (phase + "_response.json"), response)
            self.save()
            choice = response["choices"][0]
            message = choice["message"]
            reasoning = ((response.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
            if (choice.get("finish_reason") != "stop" or message.get("tool_calls")
                    or message.get("reasoning_content") or reasoning not in (None, 0)):
                raise ValueError("Incomplete/unexpected model response; no retry")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Empty model response; no retry")
            return message["content"]


def make_client(model_key):
    from openai import OpenAI

    _, url, name = MODELS[model_key]
    secret = os.environ.get(name)
    if not secret and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            secret = winreg.QueryValueEx(key, name)[0]
    if not secret:
        raise ValueError("Missing provider credential: " + name)
    return OpenAI(api_key=secret, base_url=url, timeout=180, max_retries=0)


def execute_program(root, out, sources, code, config):
    area = Path(out) / "execution"
    (area / "input").mkdir(parents=True)
    artifacts = area / "artifacts"
    artifacts.mkdir()
    for item in sources:
        source = Path(root) / item["source"]
        if sha(source) != item["sha256"]:
            raise ValueError("Input changed before execution")
        target = area / item["staged_path"]
        shutil.copyfile(source, target)
        if sha(target) != item["sha256"]:
            raise ValueError("Staged input mismatch")
    program = area / "program.py"
    program.write_text(code, encoding="utf-8")
    # Construct an allowlist environment: no provider keys, user Python paths or proxies.
    env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR", "COMSPEC", "SYSTEMDRIVE") if k in os.environ}
    env.update(PATH=os.pathsep.join([str(Path(sys.executable).parent), str(Path(sys.base_prefix) / "Library/bin"),
                                    str(Path(os.environ.get("SystemRoot", "/")) / "System32")]),
               TEMP=str(artifacts.resolve()), TMP=str(artifacts.resolve()),
               MPLCONFIGDIR=str((artifacts / "matplotlib").resolve()), MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stdout_path, stderr_path = area / "stdout.txt", area / "stderr.txt"
    started = time.monotonic()
    write(Path(out) / "execution_status.json", dict(state="running", execution_attempts=1))
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen([sys.executable, "-I", "-X", "utf8",
            str((Path(root) / "baselines/codegen_baseline/worker.py").resolve()), str(program.resolve())],
            cwd=area, env=env, stdout=stdout, stderr=stderr,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        timed_out = False
        try:
            process.wait(timeout=config.code_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    result = dict(success=not timed_out and process.returncode == 0, returncode=process.returncode,
                  timed_out=timed_out, execution_attempts=1, elapsed_seconds=time.monotonic() - started,
                  stdout_path=str(stdout_path), stderr_path=str(stderr_path))
    write(Path(out) / "execution_result.json", result)
    write(Path(out) / "execution_status.json", dict(state="completed", **result))
    return result


def bounded_text(path, limit):
    # Bound in bytes before decoding; keep a deterministic prefix and suffix.
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= limit:
            raw = handle.read()
        else:
            raw = handle.read(limit // 2) + b"\n[...OUTPUT TRUNCATED...]\n"
            handle.seek(-limit // 2, 2)
            raw += handle.read()
    return dict(text=raw.decode("utf-8", errors="replace"), truncated=size > limit, total_bytes=size)


def run(root, out, manifest, session):
    root, out = Path(root), Path(out)
    if (out / "started.json").exists():
        raise FileExistsError("Case already started; no automatic whole-case retry")
    for relative, expected in manifest["source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError("Frozen source changed")
    if sha(root / manifest["json_path"]) != manifest["case_json_sha256"]:
        raise ValueError("Frozen case changed")
    for item in manifest["input_sources"]:
        if sha(root / item["source"]) != item["sha256"]:
            raise ValueError("Frozen input changed")
    config = Config(**manifest["config"])
    if session.config != config:
        raise ValueError("Session and frozen execution configurations differ")
    model_input = read(out / "model_input.json")
    if hashlib.sha256(json.dumps(model_input, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != manifest["model_input_sha256"]:
        raise ValueError("Frozen model input changed")
    expected_messages = [dict(role="system", content=SYSTEM), dict(role="user", content=(
        CODE_INSTRUCTIONS + "\nACTUAL TASK:\n" + json.dumps(model_input, ensure_ascii=False)))]
    expected_request = request(manifest["generator"], expected_messages, config)
    if read(out / "code_request.json") != expected_request:
        raise ValueError("Frozen code request changed")
    write(out / "started.json", dict(started_at=now(), pid=os.getpid()))
    start = time.monotonic()
    state = dict(protocol=PROTOCOL, case_id=manifest["case_id"], state="generating_code",
                 execution_attempts=0, eligible_for_scoring=False, insights_count=0)
    phase = "code"
    try:
        write(out / "status.json", state)
        first_request = expected_request
        raw_code = session.invoke("code", first_request)
        code = parse_code(raw_code)
        state.update(state="executing", execution_attempts=1)
        write(out / "status.json", state)
        phase = "execution"
        execution = execute_program(root, out, manifest["input_sources"], code, config)
        if not execution["success"]:
            raise RuntimeError("Program failed or timed out; no code repair")
        observation = dict(returncode=execution["returncode"],
            stdout=bounded_text(execution["stdout_path"], config.observation_chars * 4 // 5),
            stderr=bounded_text(execution["stderr_path"], config.observation_chars // 5))
        write(out / "model_observation.json", observation)
        state.update(state="generating_answer")
        write(out / "status.json", state)
        phase = "answer"
        messages = first_request["messages"] + [dict(role="assistant", content=raw_code),
            dict(role="user", content=ANSWER_INSTRUCTIONS + "\nEXECUTION OBSERVATION:\n" +
                 json.dumps(observation, ensure_ascii=False))]
        raw_answer = session.invoke("answer", request(manifest["generator"], messages, config))
        answer = parse_answer(raw_answer)
        offline = manifest["validation_mode"] != "benchmark"
        eligible = bool(answer["insights"]) and not offline
        write(out / "parsed_output.json", dict(answer=answer, structure_matches_request=True,
            eligible_for_scoring=eligible, validation_mode=manifest["validation_mode"]))
        state.update(state="returned" if answer["insights"] else "empty", eligible_for_scoring=eligible,
                     insights_count=len(answer["insights"]), summary_present=True)
    except Exception as exc:
        # Preserve responses, source and stdout. Provider exception text may contain credentials.
        state.update(state="failed", failure_phase=phase, error_type=type(exc).__name__)
        write(out / "failure.json", dict(phase=phase, error_type=type(exc).__name__, no_regeneration=True))
    state.update(session.summary(), elapsed_seconds=time.monotonic() - start, ended_at=now())
    write(out / "status.json", state)
    write(out / "job_result.json", dict(case_id=manifest["case_id"], json_path=manifest["json_path"],
        directory=str(out.resolve()), protocol=PROTOCOL, validation_mode=manifest["validation_mode"],
        eligible_for_scoring=state["eligible_for_scoring"], returned_insight_count=state["insights_count"],
        status=state))
    return state


class OfflineClient:
    """Scripted model responses for an explicit, unscorable offline smoke run."""
    def __init__(self, table):
        self.chat = self.completions = self
        self.index = 0
        self.table = table

    def create(self, **kwargs):
        self.index += 1
        if self.index == 1:
            answer = dict(code="import pandas as pd\ndf = pd.read_csv(" + repr(self.table) +
                              ")\nprint('OFFLINE ROW COUNT:', len(df))")
        else:
            answer = dict(insights=["OFFLINE PROTOCOL CHECK ONLY; not a benchmark finding."],
                          summary="OFFLINE TEST ONLY; never eligible for scoring.")
        return dict(model="offline_fake", usage=None, choices=[dict(finish_reason="stop",
            message=dict(role="assistant", content=json.dumps(answer)))])
