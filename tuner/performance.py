"""Aggregate trade outcomes for the LLM tuner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import structlog

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class TradeOutcome:
    """One closed trade used for tuning context."""

    symbol: str
    strategy_id: str
    side: str
    pnl: float
    entry_price: float
    exit_price: float
    exit_reason: str
    source: str
    run_at: datetime | None


@dataclass(frozen=True)
class StrategySymbolPerformance:
    """Roll-up stats for one (strategy, symbol) pair."""

    symbol: str
    strategy_id: str
    current_params: dict[str, Any]
    trades: tuple[TradeOutcome, ...]

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        return 100.0 * self.win_count / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss <= 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_consecutive_losses(self) -> int:
        streak = 0
        best = 0
        for trade in self.trades:
            if trade.pnl < 0:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return best

    def to_prompt_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary for the LLM prompt."""
        recent = [
            {
                "side": t.side,
                "pnl": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
                "source": t.source,
            }
            for t in self.trades[-5:]
        ]
        return {
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "current_params": self.current_params,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": (
                None
                if self.profit_factor == float("inf")
                else round(self.profit_factor, 2)
            ),
            "max_consecutive_losses": self.max_consecutive_losses,
            "recent_trades": recent,
        }


def collect_performance(
    conn: duckdb.DuckDBPyConnection,
    *,
    lookback_days: int = 30,
    max_runs: int = 50,
    active_configs: dict[str, dict[str, Any]] | None = None,
) -> list[StrategySymbolPerformance]:
    """Build performance roll-ups from persisted backtest runs.

    Args:
        conn: Dashboard DuckDB connection (``backtest_runs`` table).
        lookback_days: Ignore runs older than this many days.
        max_runs: Cap how many recent runs we scan.
        active_configs: Optional ``symbol -> {strategy_id, params}`` from
            :mod:`tuner.active` so symbols with no trades still appear.

    Returns:
        One :class:`StrategySymbolPerformance` per distinct (strategy, symbol)
        key seen in the lookback window, sorted by symbol then strategy.
    """
    cutoff = datetime.now(tz=IST) - timedelta(days=lookback_days)
    rows = conn.execute(
        """
        SELECT strategy, symbol, params, run_at, result_json
        FROM backtest_runs
        WHERE run_at >= ?
        ORDER BY run_at DESC
        LIMIT ?
        """,
        [cutoff, max_runs],
    ).fetchall()

    buckets: dict[tuple[str, str], list[TradeOutcome]] = {}
    params_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        strategy_id = str(row[0])
        symbol = str(row[1])
        params_raw = json.loads(str(row[2]))
        run_at = _coerce_dt(row[3])
        result = json.loads(str(row[4]))
        key = (strategy_id, symbol)
        strat_params = params_raw.get("strategy", {})
        if isinstance(strat_params, dict):
            params_by_key.setdefault(key, strat_params)
        buckets.setdefault(key, [])
        for trade in result.get("closed_trades", []):
            if not isinstance(trade, dict):
                continue
            pnl = float(trade.get("pnl", 0.0))
            buckets.setdefault(key, []).append(
                TradeOutcome(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    side=str(trade.get("side", "")),
                    pnl=pnl,
                    entry_price=float(trade.get("entry_price", 0.0)),
                    exit_price=float(trade.get("exit_price", 0.0)),
                    exit_reason=str(trade.get("exit_reason", "")),
                    source="backtest",
                    run_at=run_at,
                )
            )

    if active_configs:
        for symbol, cfg in active_configs.items():
            sid = str(cfg.get("strategy_id", "ema_crossover"))
            key = (sid, symbol)
            params_by_key.setdefault(key, dict(cfg.get("params", {})))
            buckets.setdefault(key, [])

    performances: list[StrategySymbolPerformance] = []
    for key in sorted(buckets):
        strategy_id, symbol = key
        performances.append(
            StrategySymbolPerformance(
                symbol=symbol,
                strategy_id=strategy_id,
                current_params=params_by_key.get(key, {}),
                trades=tuple(buckets[key]),
            )
        )
    logger.debug(
        "tuner_performance_collected",
        pairs=len(performances),
        trades=sum(p.trade_count for p in performances),
    )
    return performances


def _coerce_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)
    return datetime.fromisoformat(str(value)).astimezone(IST)
