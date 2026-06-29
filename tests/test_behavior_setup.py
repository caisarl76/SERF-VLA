import os
import stat
import subprocess
from pathlib import Path


def test_dataset_setup_does_not_run_python_from_behavior_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    behavior_root = repo_root / "BEHAVIOR-1K"
    setup_script = behavior_root / "setup.sh"
    calls_file = tmp_path / "python_calls.txt"
    fake_python = tmp_path / "python"

    fake_python.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                f"printf '%s|%s\\n' \"$PWD\" \"$*\" >> {calls_file}",
                f"if [ \"$PWD\" = {behavior_root!s} ]; then",
                "  echo 'python ran from BEHAVIOR-1K root' >&2",
                "  exit 91",
                "fi",
                "exit 0",
                "",
            ]
        )
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["CONDA_PREFIX"] = env.get("CONDA_PREFIX", str(tmp_path / "conda"))

    result = subprocess.run(
        ["bash", str(setup_script), "--dataset", "--accept-dataset-tos"],
        cwd=behavior_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = calls_file.read_text().splitlines()
    assert any("import omnigibson" in call for call in calls)
