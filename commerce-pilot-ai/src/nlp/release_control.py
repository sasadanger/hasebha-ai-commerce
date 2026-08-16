"""Fail-closed internal-test release controls (V4-IR-02).

`execution_control.PartitionStore`/`ReleaseAuthorization` (V4 evidence, left
unmodified so its existing tests keep passing unchanged) is self-asserted:
any caller can construct `ReleaseAuthorization(authorized=True, ...)` with
empty strings and release the internal test; the ledger path is caller
-controlled; the ledger is written (as a "success") *before* the evaluator
runs, so an evaluator exception leaves a falsely-successful ledger while rows
remain resident; and the evaluator receives the real row tuple directly.

This module supersedes it for any future real execution:

* authorization must be loaded from an external evidence *file* via
  `load_external_authorization`, never constructed ad hoc by the release
  caller, and is schema-validated and immutable once loaded;
* authorization is cross-checked against an independently supplied, frozen
  winner record (`FrozenWinnerRecord`) covering compound_id, configuration
  fingerprint, split hash, run id, and experiment id -- not merely an
  `authorized=True` flag;
* the ledger path is computed *internally* from a validated run_id/
  experiment_id pair anchored under a fixed artifact root -- callers cannot
  choose an arbitrary ledger location;
* release state flips to "consumed" and the store's own row reference is
  cleared *before* the evaluator is invoked, so a second release attempt (or
  a second `ProtectedInternalTestPartition` instance pointed at the same
  run/experiment, since the ledger file itself is the durable guard) is
  always denied, regardless of what the evaluator does or raises;
* the ledger unconditionally records the true outcome (success vs. evaluator
  exception) via a `finally` block, instead of a pre-written "success" state.

Residual, explicitly documented limitation: once row data is handed to an
evaluator callback within the same Python process, nothing in this module (or
any pure-Python API) can prevent that specific callback from copying the data
it was given into its own memory -- there is no in-process memory isolation
in CPython. What this module *does* guarantee is that no *other* API path
returns internal-test rows, that the store's own reference is irrecoverably
cleared before the evaluator runs, and that release happens at most once per
run/experiment as durably recorded by the ledger file.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

RELEASE_LEDGER_SCHEMA = "nlp-internal-test-release-v2"
AUTHORIZATION_EVIDENCE_SCHEMA = "nlp-internal-test-authorization-v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = _REPO_ROOT / "artifacts" / "experiments" / "nlp" / "phase2c" / "batch1"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ReleaseDenied(PermissionError):
    """Raised whenever release cannot proceed; fail closed."""


class ReleaseEvaluationFailed(RuntimeError):
    """Raised when the evaluator callback itself raises during a release."""


def _is_safe_token(value: Any) -> bool:
    return isinstance(value, str) and bool(_SAFE_TOKEN.match(value)) and ".." not in value


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value.lower()))


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseDenied(f"authorization field {field!r} must be a non-empty string")
    return value


@dataclass(frozen=True)
class FrozenWinnerRecord:
    """An immutable winner identity, independently computed for this run."""
    run_id: str
    experiment_id: str
    compound_id: str
    configuration_fingerprint: str
    split_hash: str

    def __post_init__(self) -> None:
        if not _is_safe_token(self.run_id):
            raise ReleaseDenied("winner run_id must be a non-empty safe token")
        if not _is_safe_token(self.experiment_id):
            raise ReleaseDenied("winner experiment_id must be a non-empty safe token")
        if not isinstance(self.compound_id, str) or not self.compound_id:
            raise ReleaseDenied("winner compound_id must be a non-empty string")
        if not _is_hex64(self.configuration_fingerprint):
            raise ReleaseDenied("winner configuration_fingerprint must be a 64-character hex string")
        if not _is_hex64(self.split_hash):
            raise ReleaseDenied("winner split_hash must be a 64-character hex string")


@dataclass(frozen=True)
class ExternalAuthorizationEvidence:
    """Authorization evidence loaded from an external, immutable file.

    Never construct this directly outside `load_external_authorization`: doing
    so would reintroduce self-asserted authorization.
    """
    schema_version: str
    authorized: bool
    run_id: str
    experiment_id: str
    winning_compound_id: str
    winning_configuration_fingerprint: str
    split_hash: str
    reason: str
    metric_contract: str
    split_policy: str
    evidence_path: str
    evidence_sha256: str


def load_external_authorization(evidence_path: Path) -> ExternalAuthorizationEvidence:
    """Load and schema-validate authorization evidence from an external JSON file.

    Fails closed on: a missing file, malformed/non-object JSON, an
    unrecognized schema_version, or any missing/empty/malformed required
    field. The file's own sha256 is captured so `release_once` can re-hash it
    at call time and detect post-load tampering (TOCTOU defense).
    """
    evidence_path = Path(evidence_path)
    try:
        raw_bytes = evidence_path.read_bytes()
    except OSError as exc:
        raise ReleaseDenied(f"authorization evidence file is unavailable: {evidence_path}") from exc

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseDenied("authorization evidence file is not valid JSON") from exc

    if not isinstance(data, dict):
        raise ReleaseDenied("authorization evidence file must contain a JSON object")

    if data.get("schema_version") != AUTHORIZATION_EVIDENCE_SCHEMA:
        raise ReleaseDenied(f"unsupported authorization evidence schema_version: {data.get('schema_version')!r}")

    if not isinstance(data.get("authorized"), bool):
        raise ReleaseDenied("authorization evidence field 'authorized' must be a boolean")

    run_id = _require_nonempty_str(data.get("run_id"), "run_id")
    experiment_id = _require_nonempty_str(data.get("experiment_id"), "experiment_id")
    winning_compound_id = _require_nonempty_str(data.get("winning_compound_id"), "winning_compound_id")
    winning_configuration_fingerprint = _require_nonempty_str(
        data.get("winning_configuration_fingerprint"), "winning_configuration_fingerprint",
    )
    split_hash = _require_nonempty_str(data.get("split_hash"), "split_hash")
    reason = _require_nonempty_str(data.get("reason"), "reason")
    metric_contract = _require_nonempty_str(data.get("metric_contract"), "metric_contract")
    split_policy = _require_nonempty_str(data.get("split_policy"), "split_policy")

    if not _is_hex64(winning_configuration_fingerprint):
        raise ReleaseDenied("authorization field 'winning_configuration_fingerprint' must be a 64-character hex string")
    if not _is_hex64(split_hash):
        raise ReleaseDenied("authorization field 'split_hash' must be a 64-character hex string")

    return ExternalAuthorizationEvidence(
        schema_version=data["schema_version"],
        authorized=data["authorized"],
        run_id=run_id,
        experiment_id=experiment_id,
        winning_compound_id=winning_compound_id,
        winning_configuration_fingerprint=winning_configuration_fingerprint,
        split_hash=split_hash,
        reason=reason,
        metric_contract=metric_contract,
        split_policy=split_policy,
        evidence_path=str(evidence_path),
        evidence_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _verify_authorization(evidence: ExternalAuthorizationEvidence, winner: FrozenWinnerRecord) -> None:
    if not evidence.authorized:
        raise ReleaseDenied("authorization evidence does not grant release")

    current_bytes = Path(evidence.evidence_path).read_bytes()
    if hashlib.sha256(current_bytes).hexdigest() != evidence.evidence_sha256:
        raise ReleaseDenied("authorization evidence file changed after it was loaded")

    if evidence.run_id != winner.run_id:
        raise ReleaseDenied("authorization run_id does not match the frozen winner record")
    if evidence.experiment_id != winner.experiment_id:
        raise ReleaseDenied("authorization experiment_id does not match the frozen winner record")
    if evidence.winning_compound_id != winner.compound_id:
        raise ReleaseDenied("authorization winning_compound_id does not match the frozen winner record")
    if evidence.winning_configuration_fingerprint != winner.configuration_fingerprint:
        raise ReleaseDenied("authorization configuration fingerprint does not match the frozen winner record")
    if evidence.split_hash != winner.split_hash:
        raise ReleaseDenied("authorization split_hash does not match the frozen winner record")


def _safe_ledger_path(run_id: str, experiment_id: str, artifact_root: Path) -> Path:
    if not _is_safe_token(run_id):
        raise ReleaseDenied(f"unsafe run_id for ledger path: {run_id!r}")
    if not _is_safe_token(experiment_id):
        raise ReleaseDenied(f"unsafe experiment_id for ledger path: {experiment_id!r}")
    root = artifact_root.resolve()
    candidate = (root / experiment_id / run_id / "internal_test_release.json").resolve()
    if root not in candidate.parents:
        raise ReleaseDenied("resolved ledger path escapes the allowed artifact root")
    return candidate


@dataclass(frozen=True)
class ProvenancedRow:
    """A single internal-test row tagged with the provenance it must match to be released."""
    run_id: str
    experiment_id: str
    partition: str
    split_hash: str
    payload: Any


class ProtectedInternalTestPartition:
    """Holds internal-test rows inaccessible except through one audited release."""

    def __init__(self, rows: Sequence[ProvenancedRow]):
        if not rows:
            raise ValueError("internal test partition must not be empty")
        run_ids = {row.run_id for row in rows}
        experiment_ids = {row.experiment_id for row in rows}
        split_hashes = {row.split_hash for row in rows}
        partitions = {row.partition for row in rows}
        if len(run_ids) != 1 or len(experiment_ids) != 1 or len(split_hashes) != 1:
            raise ValueError("internal test rows must share exactly one run_id/experiment_id/split_hash")
        if partitions != {"internal_test"}:
            raise ValueError("all rows supplied to ProtectedInternalTestPartition must be tagged partition='internal_test'")
        self._run_id = next(iter(run_ids))
        self._experiment_id = next(iter(experiment_ids))
        self._split_hash = next(iter(split_hashes))
        self._rows: tuple[ProvenancedRow, ...] | None = tuple(rows)
        self._released = False

    def peek_rows(self) -> None:
        raise ReleaseDenied("internal test rows are not accessible before an audited, authorized release")

    def release_once(
        self, *,
        evidence: ExternalAuthorizationEvidence,
        winner: FrozenWinnerRecord,
        evaluator: Callable[[tuple[Any, ...]], Any],
        artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    ) -> Any:
        """Release rows to exactly one evaluator call, then permanently deny further access."""
        if self._released or self._rows is None:
            raise ReleaseDenied("internal test release is one-time and already consumed")

        if winner.run_id != self._run_id or winner.experiment_id != self._experiment_id:
            raise ReleaseDenied("frozen winner run_id/experiment_id does not match this partition")
        if winner.split_hash != self._split_hash:
            raise ReleaseDenied("frozen winner split_hash does not match this partition")

        _verify_authorization(evidence, winner)

        ledger_path = _safe_ledger_path(self._run_id, self._experiment_id, artifact_root)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # Durably and *atomically* reserve the ledger slot before invoking the
        # evaluator: an exclusive-create ("x" mode -> O_EXCL) closes the
        # check-then-write race a plain `if ledger_path.exists(): ...` would
        # leave open between two concurrent release_once calls (e.g. from two
        # independently constructed partition instances for the same
        # run/experiment). It also protects against a process death or a
        # failed final ledger write after a successful evaluation.
        reservation_payload = json.dumps({
            "schema_version": RELEASE_LEDGER_SCHEMA,
            "run_id": self._run_id,
            "experiment_id": self._experiment_id,
            "release_state": "RESERVED_PENDING_EVALUATION",
            "reserved_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n"
        try:
            with ledger_path.open("x", encoding="utf-8") as fh:
                fh.write(reservation_payload)
        except FileExistsError as exc:
            raise ReleaseDenied("a release ledger already exists for this run/experiment: release is one-time") from exc

        snapshot = self._rows
        self._rows = None
        self._released = True

        outcome_state = "RELEASE_FAILED_EVALUATOR_EXCEPTION"
        error_summary: str | None = None
        result: Any = None
        succeeded = False
        try:
            result = evaluator(tuple(row.payload for row in snapshot))
            outcome_state = "RELEASED_ONCE_SUCCESS"
            succeeded = True
            return result
        except Exception as exc:  # noqa: BLE001 - deliberately wrap any evaluator failure
            error_summary = f"{type(exc).__name__}: {exc}"
            raise ReleaseEvaluationFailed(str(exc)) from exc
        finally:
            ledger_event = {
                "schema_version": RELEASE_LEDGER_SCHEMA,
                "run_id": self._run_id,
                "experiment_id": self._experiment_id,
                "winning_compound_id": winner.compound_id,
                "winning_configuration_fingerprint": winner.configuration_fingerprint,
                "split_hash": self._split_hash,
                "authorization_evidence_path": evidence.evidence_path,
                "authorization_evidence_sha256": evidence.evidence_sha256,
                "authorization_reason": evidence.reason,
                "release_state": outcome_state,
                "error_summary": error_summary,
                "row_count": len(snapshot),
                "released_utc": datetime.now(timezone.utc).isoformat(),
            }
            ledger_path.write_text(json.dumps(ledger_event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not succeeded and error_summary is None:
                # Defensive: should be unreachable, but never leave a silent non-success state undocumented.
                raise ReleaseDenied("release outcome could not be determined")
