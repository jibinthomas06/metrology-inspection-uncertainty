from typer.testing import CliRunner

from metinspect.cli import app

runner = CliRunner()


def test_cli_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "metinspect" in r.stdout

