from pathlib import Path

from config.settings import AppSettings, load_settings


def test_load_settings_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "risk:\n  max_open_positions: 5\n  daily_loss_cap_pct: 3.0\n",
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.risk.max_open_positions == 5
    assert settings.risk.daily_loss_cap_pct == 3.0


def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.risk.position_sizing == "atr_based"
