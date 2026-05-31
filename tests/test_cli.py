"""CLI integration tests using typer.testing.CliRunner."""
from typer.testing import CliRunner
from kernel_factory.cli import app

runner = CliRunner()


def test_cli_run_matmul():
    result = runner.invoke(app, [
        "run", "--op", "matmul",
        "--M", "256", "--N", "256", "--K", "128",
        "--tpu", "v5e",
    ])
    assert result.exit_code == 0
    assert "PASSED" in result.output or "matmul" in result.output


def test_cli_run_rmsnorm():
    result = runner.invoke(app, [
        "run", "--op", "rmsnorm",
        "--M", "128", "--N", "256", "--K", "256",
        "--tpu", "v5e",
    ])
    assert result.exit_code == 0


def test_cli_inspect_matmul():
    result = runner.invoke(app, [
        "inspect", "--op", "matmul",
        "--M", "512", "--N", "512", "--K", "256",
        "--tpu", "v5e",
    ])
    assert result.exit_code == 0
    assert "block_m" in result.output or "128" in result.output


def test_cli_run_unsupported_op():
    result = runner.invoke(app, [
        "run", "--op", "attention",
        "--M", "512", "--N", "512", "--K", "256",
        "--tpu", "v5e",
    ])
    assert result.exit_code != 0


def test_cli_run_writes_output_file(tmp_path):
    out = tmp_path / "kernel.py"
    result = runner.invoke(app, [
        "run", "--op", "matmul",
        "--M", "256", "--N", "256", "--K", "128",
        "--tpu", "v5e",
        # Pin an empty corpus so assembly uses the static template deterministically,
        # independent of any .lancedb in the working directory.
        "--rag-path", str(tmp_path / ".lancedb"),
        "--output-file", str(out),
    ])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "run_matmul" in content


def test_cli_seed_command(tmp_path):
    result = runner.invoke(app, [
        "seed", "--rag-path", str(tmp_path / ".lancedb"),
    ])
    assert result.exit_code == 0


def test_cli_v4_hardware():
    result = runner.invoke(app, [
        "run", "--op", "matmul",
        "--M", "64", "--N", "64", "--K", "64",
        "--tpu", "v4",
    ])
    assert result.exit_code == 0
