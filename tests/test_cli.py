from click.testing import CliRunner

from orchestrator.main import cli


def test_cli_backtest() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "--bars-count", "200"])
    assert result.exit_code == 0
    assert "trades=" in result.output
    assert "sharpe=" in result.output


def test_cli_ab_test() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ab-test", "--bars-count", "200"])
    assert result.exit_code == 0
    assert "Rules Only" in result.output
    assert "Co-Decide" in result.output


def test_cli_paper() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["paper", "--bars-count", "100"])
    assert result.exit_code == 0
    assert "signals=" in result.output


def test_cli_dashboard_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "localhost" in result.output


def test_cli_dashboard_refuses_non_localhost() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["dashboard", "--host", "0.0.0.0", "--port", "8765"])
    assert result.exit_code == 1
    assert "refusing to bind" in result.output
