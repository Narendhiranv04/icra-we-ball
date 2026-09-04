"""External Fast Downward planning and VAL validation adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Mapping, Sequence

from .artifacts import atomic_write_json, atomic_write_text, sha256_file, sha256_text
from .contracts import (
    GeneratedPDDLProblem,
    PDDLValidationResult,
    SymbolicAction,
    SymbolicPlan,
    ValidationStage,
)
from .domains.registry import DomainDefinition


class PlannerError(RuntimeError):
    """Base class for expected symbolic-planning failures."""


class PlannerInfrastructureError(PlannerError):
    """The configured external tool cannot be safely invoked."""


class PlannerTimeoutError(PlannerError):
    """The symbolic planning budget was exhausted."""


class TranslatorError(PlannerError):
    """Fast Downward rejected the domain/problem during translation."""

    def __init__(self, result: PDDLValidationResult):
        super().__init__("Fast Downward translation failed")
        self.result = result


class NoPlanError(PlannerError):
    """Search completed without producing a plan."""


class PlanFormatError(PlannerError):
    """A planner output is not a valid baseline symbolic plan."""


class PlanValidationError(PlannerError):
    """VAL rejected the selected plan."""

    def __init__(self, result: PDDLValidationResult):
        super().__init__("VAL rejected the selected plan")
        self.result = result


@dataclass(frozen=True)
class ToolIdentity:
    path: Path
    version: str
    sha256: str


@dataclass(frozen=True)
class PlanningResult:
    plan: SymbolicPlan
    translation: PDDLValidationResult
    plan_validation: PDDLValidationResult


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    elapsed_seconds: float


@dataclass(frozen=True)
class _PlanCandidate:
    path: Path
    actions: tuple[tuple[int, str, tuple[str, ...]], ...]
    cost: float | None
    normalized_text: str


class VALAdapter:
    """Invoke an external VAL executable and retain its complete diagnostics."""

    def __init__(
        self,
        executable: str | Path,
        *,
        expected_version: str | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.expected_version = expected_version

    def validate_executable(self, output_root: str | Path) -> ToolIdentity:
        return _probe_tool(
            self.executable,
            expected_version=self.expected_version,
            output_root=Path(output_root),
            label="val_version",
        )

    def validate(
        self,
        *,
        domain_path: str | Path,
        problem_path: str | Path,
        plan_path: str | Path,
        output_root: str | Path,
        timeout_seconds: float,
    ) -> PDDLValidationResult:
        destination = Path(output_root)
        self.validate_executable(destination / "version")
        process = _run_process(
            [
                str(self.executable),
                str(Path(domain_path)),
                str(Path(problem_path)),
                str(Path(plan_path)),
            ],
            output_root=destination / "run",
            timeout_seconds=timeout_seconds,
        )
        output = process.stdout_path.read_text(encoding="utf-8").lower()
        valid_marker = "plan valid" in output or "successful plans: 1" in output
        invalid_marker = "plan failed" in output or "plan invalid" in output
        valid = process.returncode == 0 and valid_marker and not invalid_marker
        diagnostics = () if valid else ("VAL did not report a valid plan",)
        return PDDLValidationResult(
            valid=valid,
            stage=ValidationStage.PLAN_VAL,
            diagnostics=diagnostics,
            exit_code=process.returncode,
            stdout_artifact=str(process.stdout_path),
            stderr_artifact=str(process.stderr_path),
            elapsed_seconds=process.elapsed_seconds,
        )


class FastDownwardPlanner:
    """Translate, search, normalize, and independently validate a PDDL plan."""

    def __init__(
        self,
        executable: str | Path,
        val_adapter: VALAdapter,
        *,
        expected_version: str = "24.06",
        search_alias: str = "lama-first",
        timeout_seconds: float = 200.0,
    ) -> None:
        if not search_alias.strip():
            raise ValueError("search_alias must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.executable = Path(executable)
        self.val_adapter = val_adapter
        self.expected_version = expected_version
        self.search_alias = search_alias
        self.timeout_seconds = timeout_seconds

    def plan(
        self,
        *,
        problem: GeneratedPDDLProblem,
        domain: DomainDefinition,
        output_root: str | Path,
        initial_goal_satisfied: bool = False,
    ) -> PlanningResult:
        destination = Path(output_root)
        planner_root = destination / "planner"
        domain_path = atomic_write_text(destination / "domain.pddl", domain.text)
        problem_path = atomic_write_text(destination / "problem.pddl", problem.problem_text)
        identity = _probe_tool(
            self.executable,
            expected_version=self.expected_version,
            output_root=planner_root / "version",
            label="fast_downward_version",
        )
        started = time.perf_counter()
        sas_path = planner_root / "output.sas"
        translation = _run_process(
            [
                str(self.executable),
                "--translate",
                "--sas-file",
                str(sas_path),
                str(domain_path),
                str(problem_path),
            ],
            output_root=planner_root / "translate",
            timeout_seconds=self._remaining(started),
        )
        translation_result = PDDLValidationResult(
            valid=translation.returncode == 0 and sas_path.is_file(),
            stage=ValidationStage.TRANSLATOR,
            diagnostics=(
                ()
                if translation.returncode == 0 and sas_path.is_file()
                else ("Fast Downward translation failed or produced no SAS file",)
            ),
            exit_code=translation.returncode,
            stdout_artifact=str(translation.stdout_path),
            stderr_artifact=str(translation.stderr_path),
            elapsed_seconds=translation.elapsed_seconds,
        )
        atomic_write_json(planner_root / "translation_validation.json", translation_result.to_dict())
        if not translation_result.valid:
            raise TranslatorError(translation_result)

        plan_prefix = planner_root / "sas_plan"
        search = _run_process(
            [
                str(self.executable),
                "--alias",
                self.search_alias,
                "--plan-file",
                str(plan_prefix),
                str(sas_path),
            ],
            output_root=planner_root / "search",
            timeout_seconds=self._remaining(started),
        )
        if search.returncode != 0:
            raise NoPlanError(f"Fast Downward search exited with {search.returncode}")

        plan_paths = tuple(sorted(planner_root.glob("sas_plan*"), key=_plan_path_key))
        if not plan_paths:
            raise NoPlanError("Fast Downward search produced no sas_plan files")
        candidates = tuple(
            parse_plan_file(
                path,
                domain=domain,
                declared_objects=problem.declared_objects,
                allow_empty=initial_goal_satisfied,
            )
            for path in plan_paths
        )
        selected = min(candidates, key=_candidate_key)
        normalized_path = atomic_write_text(
            planner_root / "selected_plan.plan", selected.normalized_text
        )
        validation = self.val_adapter.validate(
            domain_path=domain_path,
            problem_path=problem_path,
            plan_path=normalized_path,
            output_root=planner_root / "val",
            timeout_seconds=self._remaining(started),
        )
        atomic_write_json(planner_root / "plan_validation.json", validation.to_dict())
        if not validation.valid:
            raise PlanValidationError(validation)

        actions = tuple(
            SymbolicAction(
                action_index=index,
                action_instance_id=(
                    f"vilain_{problem.attempt_index:02d}_{index + 1:03d}_"
                    f"{operator.replace('-', '_')}"
                ),
                operator=operator,
                arguments=arguments,
            )
            for index, operator, arguments in selected.actions
        )
        plan = SymbolicPlan(
            attempt_index=problem.attempt_index,
            planner_name="Fast Downward",
            planner_version=identity.version,
            search_configuration=self.search_alias,
            actions=actions,
            plan_cost=selected.cost,
            planner_time_seconds=time.perf_counter() - started,
            raw_plan_artifacts=tuple(str(path) for path in plan_paths),
            plan_sha256=sha256_text(selected.normalized_text),
        )
        atomic_write_json(planner_root / "symbolic_plan.json", plan.to_dict())
        return PlanningResult(plan, translation_result, validation)

    def _remaining(self, started: float) -> float:
        remaining = self.timeout_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            raise PlannerTimeoutError("symbolic planning timeout exhausted")
        return remaining


def parse_plan_file(
    path: str | Path,
    *,
    domain: DomainDefinition,
    declared_objects: Sequence[str] | Mapping[str, object],
    allow_empty: bool = False,
) -> _PlanCandidate:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    actions: list[tuple[int, str, tuple[str, ...]]] = []
    seen_indices: set[int] = set()
    cost: float | None = None
    next_index = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cost_match = re.match(r";\s*cost\s*=\s*([-+0-9.eE]+)", stripped, re.I)
        if cost_match:
            try:
                cost = float(cost_match.group(1))
            except ValueError as error:
                raise PlanFormatError(f"invalid plan cost in {source}") from error
            continue
        if stripped.startswith(";"):
            continue
        action_match = re.fullmatch(
            r"(?:(\d+)\s*:\s*)?\(\s*([^\s()]+)((?:\s+[^\s()]+)*)\s*\)"
            r"(?:\s*\[[^\]]+\])?",
            stripped,
        )
        if action_match is None:
            raise PlanFormatError(f"unrecognized plan line in {source}: {stripped!r}")
        explicit_index = action_match.group(1)
        index = int(explicit_index) if explicit_index is not None else next_index
        if index in seen_indices:
            raise PlanFormatError(f"duplicate action index {index} in {source}")
        if index != next_index:
            raise PlanFormatError(f"non-contiguous action index {index} in {source}")
        seen_indices.add(index)
        next_index += 1
        operator = action_match.group(2).lower()
        arguments = tuple(action_match.group(3).lower().split())
        expected = domain.action_signatures.get(operator)
        if expected is None:
            raise PlanFormatError(f"unknown action {operator!r} in {source}")
        if len(arguments) != len(expected):
            raise PlanFormatError(
                f"action {operator!r} expects {len(expected)} arguments, got {len(arguments)}"
            )
        known_objects = set(declared_objects)
        unknown = tuple(argument for argument in arguments if argument not in known_objects)
        if unknown:
            raise PlanFormatError(f"action {operator!r} uses undeclared objects {unknown!r}")
        actions.append((index, operator, arguments))
    if not actions and not allow_empty:
        raise PlanFormatError("empty plan is only valid when the initial goal is satisfied")
    normalized = "".join(
        f"({operator}{' ' if arguments else ''}{' '.join(arguments)})\n"
        for _, operator, arguments in actions
    )
    if cost is not None:
        normalized += f"; cost = {cost:g}\n"
    return _PlanCandidate(source, tuple(actions), cost, normalized)


def _probe_tool(
    executable: Path,
    *,
    expected_version: str | None,
    output_root: Path,
    label: str,
) -> ToolIdentity:
    resolved = executable.expanduser()
    if not resolved.is_absolute() or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PlannerInfrastructureError(f"{label} executable is missing or not executable: {resolved}")
    process = _run_process(
        [str(resolved), "--version"],
        output_root=output_root,
        timeout_seconds=10.0,
    )
    rendered = "\n".join(
        part.strip()
        for part in (
            process.stdout_path.read_text(encoding="utf-8"),
            process.stderr_path.read_text(encoding="utf-8"),
        )
        if part.strip()
    )
    if process.returncode != 0 or not rendered:
        raise PlannerInfrastructureError(f"could not determine {label}")
    if expected_version is not None and expected_version not in rendered:
        raise PlannerInfrastructureError(
            f"{label} mismatch: expected {expected_version!r}, got {rendered!r}"
        )
    identity = ToolIdentity(resolved, rendered, sha256_file(resolved))
    atomic_write_json(
        output_root / "identity.json",
        {"path": str(resolved), "version": rendered, "sha256": identity.sha256},
    )
    return identity


def _run_process(
    command: Sequence[str],
    *,
    output_root: Path,
    timeout_seconds: float,
) -> _ProcessResult:
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0"})
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_text(error.stdout)
        stderr = _timeout_text(error.stderr)
        elapsed = time.perf_counter() - started
        stdout_path = atomic_write_text(output_root / "stdout.txt", stdout)
        stderr_path = atomic_write_text(output_root / "stderr.txt", stderr)
        atomic_write_json(
            output_root / "command.json",
            {
                "arguments": list(command),
                "elapsed_seconds": elapsed,
                "return_code": None,
                "status": "TIMEOUT",
                "timeout_seconds": timeout_seconds,
            },
        )
        raise PlannerTimeoutError(f"external command timed out after {timeout_seconds:g}s") from error
    except OSError as error:
        raise PlannerInfrastructureError(f"could not launch external tool: {error}") from error
    elapsed = time.perf_counter() - started
    stdout_path = atomic_write_text(
        output_root / "stdout.txt", completed.stdout.decode("utf-8", errors="replace")
    )
    stderr_path = atomic_write_text(
        output_root / "stderr.txt", completed.stderr.decode("utf-8", errors="replace")
    )
    atomic_write_json(
        output_root / "command.json",
        {
            "arguments": list(command),
            "elapsed_seconds": elapsed,
            "return_code": completed.returncode,
            "status": "COMPLETED",
            "timeout_seconds": timeout_seconds,
        },
    )
    return _ProcessResult(completed.returncode, stdout_path, stderr_path, elapsed)


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _plan_path_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("sas_plan")
    if not suffix:
        return (0, path.name)
    if suffix.startswith(".") and suffix[1:].isdigit():
        return (int(suffix[1:]), path.name)
    return (2**31 - 1, path.name)


def _candidate_key(candidate: _PlanCandidate) -> tuple[float, tuple[int, str]]:
    cost = candidate.cost if candidate.cost is not None else float("inf")
    return (cost, _plan_path_key(candidate.path))
