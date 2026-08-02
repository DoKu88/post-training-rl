"""Loading of `config/verifier.yaml` and `config/reward.yaml`.

Plain extraction of named attributes, with no schema. The module stage is *exploring* and
the attribute set is still moving, so validating it through Pydantic now would pin a shape
that has not settled. When it stops moving, this is where the schema goes.

Required keys are read directly and a missing one raises immediately, naming both the key
and the file it was missing from. Nothing is defaulted into existence — a config value that
appears by default is a value nobody chose, and it will not appear in the diff between two
runs that behaved differently.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SandboxConfig:
    backend: str
    timeout_seconds: float
    memory_limit_gib: int
    max_processes: int
    max_open_files: int
    max_file_size_mib: int
    stdout_cap_bytes: int
    stderr_excerpt_bytes: int
    worker_threads: int
    kill_after_seconds: float
    backstop_slack_seconds: float
    reap_timeout_seconds: float


@dataclass(frozen=True)
class ComparatorConfig:
    absolute_float_tolerance: float


@dataclass(frozen=True)
class StartupConfig:
    self_test_timeout_seconds: float


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int | None  # None disables preamble injection


@dataclass(frozen=True)
class TestsConfig:
    max_tests_per_rollout: int
    min_tests_required: int


@dataclass(frozen=True)
class ExtractionConfig:
    prefill: str


@dataclass(frozen=True)
class VerifierConfig:
    sandbox: SandboxConfig
    comparator: ComparatorConfig
    startup: StartupConfig
    determinism: DeterminismConfig
    tests: TestsConfig
    extraction: ExtractionConfig


@dataclass(frozen=True)
class RewardEntry:
    name: str
    weight: float


@dataclass(frozen=True)
class RewardShapes:
    """The numbers that define each reward function.

    A change to one of these defines a *new* reward rather than tuning an existing one — a
    run using different values is not comparable to one that did not. They are loaded rather
    than hardcoded so two runs that behaved differently can be diffed as files.
    """

    pass_rate_threshold: float
    code_r1_no_code: float
    code_r1_wrong: float
    code_r1_correct: float
    ladder_parses: float
    ladder_runs: float
    ladder_pass_weight: float
    parse_term: float
    fence_terms: Mapping[str, float]  # keyed by Fence value


@dataclass(frozen=True)
class RewardConfig:
    functions: tuple[RewardEntry, ...]
    shadow_log: tuple[str, ...]
    shapes: RewardShapes


def load_verifier_config(path: Path) -> VerifierConfig:
    """Read the verifier and sandbox configuration from `path`."""
    document = _read_yaml(path)
    return VerifierConfig(
        sandbox=SandboxConfig(
            backend=_require(document, "sandbox.backend", path),
            timeout_seconds=_require(document, "sandbox.timeout_seconds", path),
            memory_limit_gib=_require(document, "sandbox.memory_limit_gib", path),
            max_processes=_require(document, "sandbox.max_processes", path),
            max_open_files=_require(document, "sandbox.max_open_files", path),
            max_file_size_mib=_require(document, "sandbox.max_file_size_mib", path),
            stdout_cap_bytes=_require(document, "sandbox.stdout_cap_bytes", path),
            stderr_excerpt_bytes=_require(document, "sandbox.stderr_excerpt_bytes", path),
            worker_threads=_require(document, "sandbox.worker_threads", path),
            kill_after_seconds=_require(document, "sandbox.kill_after_seconds", path),
            backstop_slack_seconds=_require(
                document, "sandbox.backstop_slack_seconds", path
            ),
            reap_timeout_seconds=_require(document, "sandbox.reap_timeout_seconds", path),
        ),
        comparator=ComparatorConfig(
            absolute_float_tolerance=_require(
                document, "comparator.absolute_float_tolerance", path
            ),
        ),
        startup=StartupConfig(
            self_test_timeout_seconds=_require(
                document, "startup.self_test_timeout_seconds", path
            ),
        ),
        determinism=DeterminismConfig(
            seed=_require(document, "determinism.seed", path),
        ),
        tests=TestsConfig(
            max_tests_per_rollout=_require(document, "tests.max_tests_per_rollout", path),
            min_tests_required=_require(document, "tests.min_tests_required", path),
        ),
        extraction=ExtractionConfig(
            prefill=_require(document, "extraction.prefill", path),
        ),
    )


def load_reward_config(path: Path) -> RewardConfig:
    """Read which reward functions run, at what weight, and which are shadow-logged."""
    document = _read_yaml(path)
    entries = _require(document, "functions", path)
    return RewardConfig(
        functions=tuple(
            RewardEntry(
                name=_require(entry, "name", path),
                weight=_require(entry, "weight", path),
            )
            for entry in entries
        ),
        shadow_log=tuple(_require(document, "shadow_log", path)),
        shapes=RewardShapes(
            pass_rate_threshold=_require(document, "shapes.pass_rate_threshold", path),
            code_r1_no_code=_require(document, "shapes.code_r1.no_code", path),
            code_r1_wrong=_require(document, "shapes.code_r1.wrong", path),
            code_r1_correct=_require(document, "shapes.code_r1.correct", path),
            ladder_parses=_require(document, "shapes.ladder.parses", path),
            ladder_runs=_require(document, "shapes.ladder.runs", path),
            ladder_pass_weight=_require(document, "shapes.ladder.pass_weight", path),
            parse_term=_require(document, "shapes.extractability.parse_term", path),
            fence_terms=dict(_require(document, "shapes.extractability.fence", path)),
        ),
    )


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _require(document: Mapping[str, Any], dotted_key: str, path: Path) -> Any:
    """Read a required value, naming both the key and the file when it is absent.

    A bare `KeyError` names the key but not where it should have been, which for a project
    with six config files sends the reader looking through all of them.
    """
    value: Any = document
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"missing required key {dotted_key!r} in {path}")
        value = value[part]
    return value
