"""Read / validate / write the YAML config file from the dashboard.

The editor sends raw YAML text. We:

1. Parse the YAML.
2. Construct an :class:`AppSettings` from the parsed dict — this surfaces
   pydantic validation errors as a structured list.
3. On save, write the raw bytes (preserving the user's formatting and
   comments where possible) to a temp file, rotate the existing
   ``config.yaml`` to ``config.yaml.bak``, then atomically rename.

TRADEOFF: We do *not* round-trip through ``yaml.dump`` on save because
that would destroy comments and reorder keys. The cost is that we cannot
deeply normalise the file — but the editor presents the file as-is and
the validator only requires structural correctness.

TRADEOFF: Secrets (``.env``) are intentionally **off-limits** to the
dashboard editor. The dashboard never reads or writes ``.env`` here; the
Kite-login flow has its own narrow ``KITE_ACCESS_TOKEN`` writer that
preserves every other line in the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from config.settings import AppSettings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    """One pydantic / YAML error suitable for display in the UI."""

    location: str
    message: str
    type: str


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating raw YAML text.

    Attributes:
        ok: ``True`` when both YAML parses and :class:`AppSettings` builds.
        issues: Empty when ``ok``; otherwise one entry per error.
        parsed: The parsed YAML dict when YAML itself was syntactically
            valid (may still be present alongside structural errors).
    """

    ok: bool
    issues: list[ValidationIssue]
    parsed: dict[str, Any] | None


@dataclass(frozen=True)
class SaveResult:
    """Outcome of a save attempt."""

    ok: bool
    validation: ValidationResult
    backup_path: Path | None


def validate_yaml(text: str) -> ValidationResult:
    """Parse ``text`` and try to construct :class:`AppSettings`.

    Args:
        text: Raw YAML config text from the editor.

    Returns:
        A :class:`ValidationResult`. The parsed dict is included when YAML
        itself is valid even if structural validation fails.
    """
    try:
        raw = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as exc:
        return ValidationResult(
            ok=False,
            issues=[
                ValidationIssue(
                    location="(yaml)",
                    message=str(exc),
                    type="yaml_error",
                )
            ],
            parsed=None,
        )

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return ValidationResult(
            ok=False,
            issues=[
                ValidationIssue(
                    location="(root)",
                    message=f"expected mapping at top level, got {type(raw).__name__}",
                    type="yaml_shape",
                )
            ],
            parsed=None,
        )

    try:
        AppSettings.model_validate(raw)
    except ValidationError as exc:
        issues = [
            ValidationIssue(
                location=".".join(str(p) for p in err["loc"]) or "(root)",
                message=err["msg"],
                type=err["type"],
            )
            for err in exc.errors()
        ]
        return ValidationResult(ok=False, issues=issues, parsed=raw)

    return ValidationResult(ok=True, issues=[], parsed=raw)


def save_yaml(
    text: str,
    *,
    config_path: Path,
    backup_suffix: str = ".bak",
) -> SaveResult:
    """Validate then write ``text`` to ``config_path`` atomically.

    The previous file (if any) is rotated to ``{config_path}.bak``. The
    new file is written via ``write + os.replace`` so a half-written file
    is never observable to other readers.

    Args:
        text: New YAML config text.
        config_path: Destination file path.
        backup_suffix: Suffix appended to the previous file's name.

    Returns:
        A :class:`SaveResult`. ``ok=False`` short-circuits before any
        files are touched when validation fails.
    """
    validation = validate_yaml(text)
    if not validation.ok:
        logger.info(
            "dashboard_config_save_rejected",
            issues=[i.message for i in validation.issues],
        )
        return SaveResult(ok=False, validation=validation, backup_path=None)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if config_path.is_file():
        backup_path = config_path.with_name(config_path.name + backup_suffix)
        backup_path.write_bytes(config_path.read_bytes())

    tmp_path = config_path.with_name(config_path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, config_path)
    logger.info(
        "dashboard_config_saved",
        path=str(config_path),
        backup=str(backup_path) if backup_path is not None else None,
    )
    return SaveResult(ok=True, validation=validation, backup_path=backup_path)


def read_config_text(config_path: Path, example_path: Path | None = None) -> str:
    """Read ``config_path`` if present, else fall back to ``example_path``.

    Args:
        config_path: Primary config file (typically ``config/config.yaml``).
        example_path: Optional fallback (typically
            ``config/config.example.yaml``).

    Returns:
        Text contents of whichever file exists, or an empty string if
        neither does.
    """
    if config_path.is_file():
        return config_path.read_text(encoding="utf-8")
    if example_path is not None and example_path.is_file():
        return example_path.read_text(encoding="utf-8")
    return ""
