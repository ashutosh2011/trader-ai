"""Per-artifact LLM tuning service for backtest runs and sweeps.

Operators on ``/backtests/{run_id}`` and ``/backtests/sweep/{sweep_id}``
ask Gemini "this run/sweep is what it is — what params should I try
next?". This service builds the prompt, calls the configured LLM
provider once, parses the JSON response, filters every recommendation
through the strategy schema (drop unknown params + clamp values into
their declared ``[min, max]`` bounds), and persists the result so the
page can render the last analysis without re-calling the LLM on every
refresh.

TRADEOFF: We hard-wire Gemini as the provider for v1 because the user
just dropped a ``GOOGLE_API_KEY`` and that's the only key wired to the
``/llm`` page right now. The constructor accepts a ``provider_factory``
hook so tests can substitute :class:`MockLLMProvider` and so a future
follow-up can let the operator pick the provider the same way
``LLMSettingsService.test_connection`` does.

TRADEOFF: We deliberately do **not** call ``filter_strategy_params``
from :mod:`tuner.validate` for the sweep flow's ``leaders`` /
``next_sweep`` recommendations because those reference a strategy id
the LLM picked, not the active per-symbol config — but we still reuse
the same key-filter + clamp logic locally so the sweep path stays
independent of the global-lookback tuner (which has a different
"apply to active config" semantics we don't want to inherit).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from analyst._parsing import extract_json_object
from analyst.provider import LLMProvider
from analyst.providers.google import GoogleProvider
from config.settings import AppSettings
from dashboard.services.backtest_runner import BacktestRunDetail, BacktestRunner
from dashboard.services.strategy_schemas import (
    STRATEGY_SCHEMAS,
    ParamSpec,
    StrategySchema,
    get_schema,
)
from dashboard.services.sweep_runner import (
    LeaderboardRow,
    SweepConfig,
    SweepRunner,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_TIMEOUT_SEC = 15.0
RAW_PREVIEW_CHARS = 300
MAX_TRADES_IN_PROMPT = 20
MAX_LEADERBOARD_ROWS_IN_PROMPT = 50
MAX_RECOMMENDATIONS = 4
MAX_LEADERS = 3
MAX_NEXT_SWEEP = 4

ScopeKind = Literal["run", "sweep"]
TuneStatus = Literal[
    "ok",
    "fallback_transport",
    "fallback_parse_error",
    "fallback_unexpected",
]

RUN_TUNINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_tunings (
    id VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    target_id VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    provider VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    latency_ms INTEGER NOT NULL,
    error VARCHAR,
    plan_json VARCHAR NOT NULL,
    raw_preview VARCHAR
);
"""

RUN_TUNINGS_SCOPE_TARGET_INDEX = (
    "CREATE INDEX IF NOT EXISTS run_tunings_scope_target_idx "
    "ON run_tunings(scope, target_id)"
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ParamRecommendation(BaseModel):
    """One concrete (strategy, params) suggestion the LLM thinks would do better."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str
    params: dict[str, float | int]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class RunTuningPlan(BaseModel):
    """LLM output for a single backtest run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    diagnosis: list[str] = Field(default_factory=list)
    recommendations: list[ParamRecommendation] = Field(default_factory=list)


class SweepTuningPlan(BaseModel):
    """LLM output for a sweep leaderboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    leaders: list[ParamRecommendation] = Field(default_factory=list)
    next_sweep: list[ParamRecommendation] = Field(default_factory=list)
    discard: list[str] = Field(default_factory=list)


class RunTuneRecord(BaseModel):
    """Persisted result of one per-artifact LLM tuning call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    scope: ScopeKind
    target_id: str
    created_at: datetime
    provider: str
    model: str
    status: TuneStatus
    latency_ms: int
    error: str | None = None
    plan: RunTuningPlan | SweepTuningPlan
    raw_preview: str | None = None


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


ProviderFactory = Callable[[AppSettings], LLMProvider]


def _default_provider_factory(settings: AppSettings) -> LLMProvider:
    """Build a Gemini provider from settings; raises when key is missing."""
    return GoogleProvider(settings.analyst)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RunTunerService:
    """Per-artifact LLM tuner bound to a single dashboard DuckDB connection."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        settings: AppSettings,
        backtest_runner: BacktestRunner,
        sweep_runner: SweepRunner,
        provider_factory: ProviderFactory | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._conn = conn
        self._settings = settings
        self._backtest_runner = backtest_runner
        self._sweep_runner = sweep_runner
        self._provider_factory = provider_factory or _default_provider_factory
        self._timeout = timeout_sec

    def ensure_schema(self) -> None:
        """Create the ``run_tunings`` table + index if missing (idempotent)."""
        self._conn.execute(RUN_TUNINGS_SCHEMA)
        self._conn.execute(RUN_TUNINGS_SCOPE_TARGET_INDEX)

    # ------------------------------------------------------------------
    # Run-level tuning
    # ------------------------------------------------------------------

    async def tune_run(self, run_id: str) -> RunTuneRecord:
        """Ask the LLM for param tweaks on ``run_id`` and persist the result."""
        detail = self._backtest_runner.get_run(run_id)
        if detail is None:
            msg = f"backtest run not found: {run_id}"
            raise KeyError(msg)
        schema = get_schema(detail.summary.strategy)
        if schema is None:
            msg = (
                f"strategy '{detail.summary.strategy}' has no schema; "
                "Gemini cannot suggest tweaks for unknown strategies."
            )
            raise ValueError(msg)

        provider = self._build_provider()
        prompt = build_run_tuning_prompt(detail, schema)
        return await self._call_and_persist(
            scope="run",
            target_id=run_id,
            provider=provider,
            prompt=prompt,
            parser=lambda raw: _parse_run_plan(raw, {schema.id: schema}),
            empty_plan=RunTuningPlan(summary="", diagnosis=[], recommendations=[]),
        )

    # ------------------------------------------------------------------
    # Sweep-level tuning
    # ------------------------------------------------------------------

    async def tune_sweep(self, sweep_id: str) -> RunTuneRecord:
        """Ask the LLM to analyse a sweep leaderboard and persist the result."""
        status = self._sweep_runner.status(sweep_id)
        if status is None:
            msg = f"sweep not found: {sweep_id}"
            raise KeyError(msg)
        if status.status != "done":
            msg = (
                f"sweep status is {status.status!r}; wait for it to complete "
                "before asking Gemini to analyse the leaderboard."
            )
            raise ValueError(msg)
        config = self._sweep_runner.get_config(sweep_id)
        if config is None:  # pragma: no cover - belt-and-braces
            msg = f"sweep config missing: {sweep_id}"
            raise KeyError(msg)
        leaderboard = self._sweep_runner.leaderboard(sweep_id)
        if not leaderboard:
            msg = (
                "sweep produced no completed cells — nothing for Gemini to "
                "analyse."
            )
            raise ValueError(msg)

        strategies_seen = {row.strategy for row in leaderboard}
        schemas: dict[str, StrategySchema] = {}
        for sid in strategies_seen:
            schema = get_schema(sid)
            if schema is not None:
                schemas[sid] = schema

        provider = self._build_provider()
        prompt = build_sweep_tuning_prompt(config, leaderboard, schemas)
        return await self._call_and_persist(
            scope="sweep",
            target_id=sweep_id,
            provider=provider,
            prompt=prompt,
            parser=lambda raw: _parse_sweep_plan(raw, schemas),
            empty_plan=SweepTuningPlan(
                summary="",
                leaders=[],
                next_sweep=[],
                discard=[],
            ),
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def list_for_run(self, run_id: str, *, limit: int = 5) -> list[RunTuneRecord]:
        """Return the most-recent tuning attempts for ``run_id`` (newest first)."""
        return self._list("run", run_id, limit=limit)

    def list_for_sweep(self, sweep_id: str, *, limit: int = 5) -> list[RunTuneRecord]:
        """Return the most-recent tuning attempts for ``sweep_id`` (newest first)."""
        return self._list("sweep", sweep_id, limit=limit)

    def _list(
        self, scope: ScopeKind, target_id: str, *, limit: int
    ) -> list[RunTuneRecord]:
        rows = self._conn.execute(
            "SELECT id, scope, target_id, created_at, provider, model, status, "
            "latency_ms, error, plan_json, raw_preview "
            "FROM run_tunings WHERE scope = ? AND target_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            [scope, target_id, int(limit)],
        ).fetchall()
        out: list[RunTuneRecord] = []
        for row in rows:
            out.append(_row_to_record(row))
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_provider(self) -> LLMProvider:
        if not self._settings.analyst.google_api_key:
            msg = (
                "Gemini API key not configured. Set GOOGLE_API_KEY in .env "
                "or via /llm."
            )
            raise ValueError(msg)
        return self._provider_factory(self._settings)

    async def _call_and_persist(
        self,
        *,
        scope: ScopeKind,
        target_id: str,
        provider: LLMProvider,
        prompt: str,
        parser: Callable[[str], RunTuningPlan | SweepTuningPlan],
        empty_plan: RunTuningPlan | SweepTuningPlan,
    ) -> RunTuneRecord:
        start = time.perf_counter()
        model = self._settings.analyst.model_google
        provider_name = provider.name
        try:
            raw = await asyncio.wait_for(
                provider.complete(prompt),
                timeout=self._timeout,
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "run_tuner_fallback_transport",
                scope=scope,
                target_id=target_id,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=latency_ms,
            )
            return self._persist_record(
                scope=scope,
                target_id=target_id,
                provider=provider_name,
                model=model,
                status="fallback_transport",
                latency_ms=latency_ms,
                error=str(exc),
                plan=empty_plan,
                raw_preview=None,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "run_tuner_fallback_unexpected",
                scope=scope,
                target_id=target_id,
                error=str(exc),
                latency_ms=latency_ms,
            )
            return self._persist_record(
                scope=scope,
                target_id=target_id,
                provider=provider_name,
                model=model,
                status="fallback_unexpected",
                latency_ms=latency_ms,
                error=str(exc),
                plan=empty_plan,
                raw_preview=None,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            plan = parser(raw)
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            logger.warning(
                "run_tuner_fallback_parse_error",
                scope=scope,
                target_id=target_id,
                error=str(exc),
                latency_ms=latency_ms,
            )
            return self._persist_record(
                scope=scope,
                target_id=target_id,
                provider=provider_name,
                model=model,
                status="fallback_parse_error",
                latency_ms=latency_ms,
                error=str(exc),
                plan=empty_plan,
                raw_preview=raw[:RAW_PREVIEW_CHARS] if raw else None,
            )
        return self._persist_record(
            scope=scope,
            target_id=target_id,
            provider=provider_name,
            model=model,
            status="ok",
            latency_ms=latency_ms,
            error=None,
            plan=plan,
            raw_preview=raw[:RAW_PREVIEW_CHARS] if raw else None,
        )

    def _persist_record(
        self,
        *,
        scope: ScopeKind,
        target_id: str,
        provider: str,
        model: str,
        status: TuneStatus,
        latency_ms: int,
        error: str | None,
        plan: RunTuningPlan | SweepTuningPlan,
        raw_preview: str | None,
    ) -> RunTuneRecord:
        record_id = uuid4().hex[:12]
        created_at = datetime.now(tz=IST)
        self._conn.execute(
            "INSERT INTO run_tunings ("
            "id, scope, target_id, created_at, provider, model, status, "
            "latency_ms, error, plan_json, raw_preview"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record_id,
                scope,
                target_id,
                created_at,
                provider,
                model,
                status,
                int(latency_ms),
                error,
                plan.model_dump_json(),
                raw_preview,
            ],
        )
        logger.info(
            "run_tuner_persisted",
            scope=scope,
            target_id=target_id,
            status=status,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )
        return RunTuneRecord(
            id=record_id,
            scope=scope,
            target_id=target_id,
            created_at=created_at,
            provider=provider,
            model=model,
            status=status,
            latency_ms=int(latency_ms),
            error=error,
            plan=plan,
            raw_preview=raw_preview,
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: tuple[Any, ...]) -> RunTuneRecord:
    scope = str(row[1])
    if scope not in {"run", "sweep"}:  # pragma: no cover - defensive
        msg = f"unknown scope persisted in run_tunings: {scope}"
        raise ValueError(msg)
    created_at = row[3]
    if not isinstance(created_at, datetime):
        created_at = datetime.fromisoformat(str(created_at))
    plan_payload = json.loads(str(row[9]))
    plan: RunTuningPlan | SweepTuningPlan
    if scope == "run":
        plan = RunTuningPlan.model_validate(plan_payload)
    else:
        plan = SweepTuningPlan.model_validate(plan_payload)
    status_value = str(row[6])
    # mypy needs an explicit literal cast here; we validate against the
    # known statuses before constructing the Pydantic model.
    if status_value not in {
        "ok",
        "fallback_transport",
        "fallback_parse_error",
        "fallback_unexpected",
    }:
        msg = f"unknown status persisted in run_tunings: {status_value}"
        raise ValueError(msg)
    raw_error = row[8]
    raw_preview = row[10]
    return RunTuneRecord(
        id=str(row[0]),
        scope=scope,  # type: ignore[arg-type]
        target_id=str(row[2]),
        created_at=created_at,
        provider=str(row[4]),
        model=str(row[5]),
        status=status_value,  # type: ignore[arg-type]
        latency_ms=int(row[7]),
        error=str(raw_error) if raw_error is not None else None,
        plan=plan,
        raw_preview=str(raw_preview) if raw_preview is not None else None,
    )


# ---------------------------------------------------------------------------
# Parsing + sanitisation
# ---------------------------------------------------------------------------


def _parse_run_plan(
    raw: str, schemas: dict[str, StrategySchema]
) -> RunTuningPlan:
    payload = extract_json_object(raw)
    data = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"JSON root must be an object, got {type(data).__name__}"
        raise ValueError(msg)
    plan = RunTuningPlan.model_validate(data)
    cleaned = _sanitise_recommendations(
        plan.recommendations,
        schemas=schemas,
        max_items=MAX_RECOMMENDATIONS,
    )
    return RunTuningPlan(
        summary=plan.summary,
        diagnosis=list(plan.diagnosis),
        recommendations=cleaned,
    )


def _parse_sweep_plan(
    raw: str, schemas: dict[str, StrategySchema]
) -> SweepTuningPlan:
    payload = extract_json_object(raw)
    data = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"JSON root must be an object, got {type(data).__name__}"
        raise ValueError(msg)
    plan = SweepTuningPlan.model_validate(data)
    leaders = _sanitise_recommendations(
        plan.leaders, schemas=schemas, max_items=MAX_LEADERS
    )
    next_sweep = _sanitise_recommendations(
        plan.next_sweep, schemas=schemas, max_items=MAX_NEXT_SWEEP
    )
    discard = [str(d) for d in plan.discard if isinstance(d, str) and d.strip()]
    return SweepTuningPlan(
        summary=plan.summary,
        leaders=leaders,
        next_sweep=next_sweep,
        discard=discard,
    )


def _sanitise_recommendations(
    recommendations: list[ParamRecommendation],
    *,
    schemas: dict[str, StrategySchema],
    max_items: int,
) -> list[ParamRecommendation]:
    cleaned: list[ParamRecommendation] = []
    for rec in recommendations[:max_items]:
        schema = schemas.get(rec.strategy) or get_schema(rec.strategy)
        if schema is None:
            # TRADEOFF: We silently drop recommendations that name an
            # unregistered strategy instead of raising, because the LLM
            # occasionally invents plausible-looking ids that we can't
            # run anyway.
            continue
        cleaned_params = _filter_and_clamp_params(rec.params, schema)
        if not cleaned_params:
            # An all-unknown-key payload means we'd run defaults — not
            # useful; skip the recommendation entirely.
            continue
        cleaned.append(
            ParamRecommendation(
                strategy=rec.strategy,
                params=cleaned_params,
                confidence=rec.confidence,
                rationale=rec.rationale,
            )
        )
    return cleaned


def _filter_and_clamp_params(
    params: dict[str, float | int], schema: StrategySchema
) -> dict[str, float | int]:
    """Drop unknown keys; clamp every numeric value into its declared bounds."""
    spec_by_name: dict[str, ParamSpec] = {spec.name: spec for spec in schema.params}
    cleaned: dict[str, float | int] = {}
    for name, raw_value in params.items():
        spec = spec_by_name.get(name)
        if spec is None:
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        bounded = max(spec.min, min(spec.max, numeric))
        if spec.type == "int":
            cleaned[name] = int(round(bounded))
        else:
            cleaned[name] = float(bounded)
    return cleaned


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


_PROMPT_MAX_CHARS = 12_000


def build_run_tuning_prompt(
    detail: BacktestRunDetail, schema: StrategySchema
) -> str:
    """Build the Gemini prompt for one backtest run."""
    summary = detail.summary
    current_params = summary.params.get("strategy", {}) or {}
    if not isinstance(current_params, dict):
        current_params = {}
    spec_lines: list[str] = []
    for spec in schema.params:
        spec_lines.append(
            f"  - {spec.name} ({spec.type}) default={spec.default} "
            f"min={spec.min} max={spec.max} — {spec.help}"
        )
    trades_sample = list(detail.closed_trades[:MAX_TRADES_IN_PROMPT])
    trade_lines: list[str] = []
    for idx, trade in enumerate(trades_sample, start=1):
        entry = trade.get("entry_price")
        exit_ = trade.get("exit_price")
        pnl = trade.get("pnl")
        side = trade.get("side")
        reason = trade.get("exit_reason")
        trade_lines.append(
            f"  {idx}. side={side} entry={entry} exit={exit_} "
            f"pnl={pnl} reason={reason}"
        )
    trade_block = "\n".join(trade_lines) if trade_lines else "  (no closed trades)"
    truncated_note = (
        f"  ... ({len(detail.closed_trades) - MAX_TRADES_IN_PROMPT} more trades omitted)"
        if len(detail.closed_trades) > MAX_TRADES_IN_PROMPT
        else ""
    )
    if truncated_note:
        trade_block = trade_block + "\n" + truncated_note

    body = (
        "You are tuning a single backtest run on Indian equities.\n"
        "Strict JSON output only — no markdown, no code fences, no comments.\n"
        "\n"
        f"Strategy: {schema.id} — {schema.label}\n"
        f"Summary: {schema.summary}\n"
        "\n"
        "Parameter spec:\n"
        + "\n".join(spec_lines)
        + "\n\n"
        "Current run:\n"
        f"  run_id={summary.id}\n"
        f"  symbol={summary.symbol}\n"
        f"  bars_count={summary.bars_count}\n"
        f"  current params={json.dumps(current_params, sort_keys=True)}\n"
        f"  total_pnl={summary.total_pnl}\n"
        f"  sharpe={summary.sharpe}\n"
        f"  win_rate={summary.win_rate}\n"
        f"  mdd_pct={summary.mdd_pct}\n"
        f"  total_trades={summary.total_trades}\n"
        "\n"
        f"Closed trades (first {MAX_TRADES_IN_PROMPT}):\n"
        f"{trade_block}\n"
        "\n"
        "Respond with a JSON object matching this schema:\n"
        "{\n"
        '  "summary": "1-3 sentence executive overview",\n'
        '  "diagnosis": ["bullet observations about what went wrong or right"],\n'
        '  "recommendations": [\n'
        "    {\n"
        f'      "strategy": "{schema.id}",\n'
        '      "params": {<every key MUST be one of the params listed above; '
        'every value MUST be within [min, max]>},\n'
        '      "confidence": 0.0-1.0,\n'
        '      "rationale": "why this beats the current params"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        f"Return between 1 and {MAX_RECOMMENDATIONS} recommendations; the first "
        "is the preferred pick. Every recommendation must include a FULL params "
        "object (not deltas)."
    )
    return _truncate_prompt(body)


def build_sweep_tuning_prompt(
    config: SweepConfig,
    leaderboard: list[LeaderboardRow],
    schemas: dict[str, StrategySchema],
) -> str:
    """Build the Gemini prompt for a sweep leaderboard."""
    rows = leaderboard[:MAX_LEADERBOARD_ROWS_IN_PROMPT]
    leaderboard_lines: list[str] = []
    for row in rows:
        leaderboard_lines.append(
            f"  rank={row.rank} strategy={row.strategy} symbol={row.symbol} "
            f"params={json.dumps(row.params, sort_keys=True)} "
            f"total_pnl={row.total_pnl:.2f} sharpe={row.sharpe:.3f} "
            f"trades={row.total_trades} win_rate={row.win_rate:.2f} "
            f"mdd_pct={row.mdd_pct:.2f}"
        )
    leaderboard_block = (
        "\n".join(leaderboard_lines) if leaderboard_lines else "  (empty)"
    )
    truncated_note = (
        f"  ... ({len(leaderboard) - MAX_LEADERBOARD_ROWS_IN_PROMPT} more rows omitted)"
        if len(leaderboard) > MAX_LEADERBOARD_ROWS_IN_PROMPT
        else ""
    )
    if truncated_note:
        leaderboard_block = leaderboard_block + "\n" + truncated_note

    schemas_block_lines: list[str] = []
    used_ids = sorted(schemas.keys()) or sorted(STRATEGY_SCHEMAS.keys())
    for sid in used_ids:
        schema = schemas.get(sid) or STRATEGY_SCHEMAS[sid]
        spec_summary = ", ".join(
            f"{spec.name}({spec.type}, {spec.min}-{spec.max})"
            for spec in schema.params
        )
        schemas_block_lines.append(f"  - {schema.id}: {spec_summary}")

    symbols_repr = ", ".join(f"{sym}" for sym, _ in config.symbols)
    body = (
        "You are analysing the result of a parameter sweep on Indian equities.\n"
        "Strict JSON output only — no markdown, no code fences, no comments.\n"
        "\n"
        f"Sweep label: {config.label}\n"
        f"Symbols: {symbols_repr}\n"
        f"Timeframe: {config.timeframe}\n"
        f"Date range: {config.from_date.isoformat()} -> {config.to_date.isoformat()}\n"
        f"Quantity: {config.qty}\n"
        "\n"
        "Strategy specs (param ranges):\n"
        + "\n".join(schemas_block_lines)
        + "\n\n"
        "Leaderboard (ranked best -> worst by total_pnl):\n"
        f"{leaderboard_block}\n"
        "\n"
        "Respond with a JSON object matching this schema:\n"
        "{\n"
        '  "summary": "1-3 sentence executive overview of the leaderboard",\n'
        '  "leaders": [\n'
        "    {\n"
        '      "strategy": "<id present in the leaderboard>",\n'
        '      "params": {<repeat verbatim from a leaderboard row>},\n'
        '      "confidence": 0.0-1.0,\n'
        '      "rationale": "why this cell wins"\n'
        "    }\n"
        "  ],\n"
        '  "next_sweep": [\n'
        "    {\n"
        '      "strategy": "<id you want to explore around>",\n'
        '      "params": {<NEW values to try, every key listed in this '
        'strategy\'s spec, every value within [min, max]>},\n'
        '      "confidence": 0.0-1.0,\n'
        '      "rationale": "what this probes"\n'
        "    }\n"
        "  ],\n"
        '  "discard": ["<strategy id that looked hopeless>"]\n'
        "}\n"
        "\n"
        f"Cap leaders at {MAX_LEADERS}, next_sweep at {MAX_NEXT_SWEEP}. Every "
        "recommendation must include a FULL params object (not deltas)."
    )
    return _truncate_prompt(body)


def _truncate_prompt(text: str) -> str:
    if len(text) <= _PROMPT_MAX_CHARS:
        return text
    head = text[: _PROMPT_MAX_CHARS - 200]
    return (
        head
        + "\n\n[TRUNCATED: prompt exceeded "
        + str(_PROMPT_MAX_CHARS)
        + " chars]"
    )


__all__ = [
    "DEFAULT_TIMEOUT_SEC",
    "MAX_LEADERBOARD_ROWS_IN_PROMPT",
    "MAX_LEADERS",
    "MAX_NEXT_SWEEP",
    "MAX_RECOMMENDATIONS",
    "MAX_TRADES_IN_PROMPT",
    "ParamRecommendation",
    "ProviderFactory",
    "RUN_TUNINGS_SCHEMA",
    "RUN_TUNINGS_SCOPE_TARGET_INDEX",
    "RunTuneRecord",
    "RunTunerService",
    "RunTuningPlan",
    "ScopeKind",
    "SweepTuningPlan",
    "TuneStatus",
    "build_run_tuning_prompt",
    "build_sweep_tuning_prompt",
]
