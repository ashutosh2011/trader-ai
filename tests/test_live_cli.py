from click.testing import CliRunner

from orchestrator.main import cli


def test_cli_live_dry_run() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live", "--bars-count", "100", "--dry-run"])
    assert result.exit_code == 0
    assert "live signals=" in result.output


def test_cli_live_kill_switch_env() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["live", "--bars-count", "50"],
        env={"KILL_SWITCH": "1"},
    )
    assert result.exit_code == 1
    assert "KILL switch" in result.output
