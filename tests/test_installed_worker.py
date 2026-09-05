"""Exercise wheel contents outside the checkout, where editable installs cannot hide omissions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_installed_worker_includes_execution_modules_and_default_configs(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
            str(wheels),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheels.glob("*.whl"))) as wheel:
        names = wheel.namelist()
        assert "scripts/train_chart_transform.py" in names
        assert "scripts/train_guitar_v1.py" in names
        assert "configs/guitar_v1.yaml" in names
        assert "configs/onset_classifier.yaml" in names
        assert not any(name.startswith("configs/") and name.endswith(".json") for name in names)
        wheel.extractall(installed)
    environment = dict(os.environ, PYTHONPATH=str(installed), PYTHONNOUSERSITE="1")
    commands = [
        ["-m", "src.worker", "probe", "--json"],
        ["-m", "src.worker", "pipeline", "list", "--json"],
        [str(installed / "scripts" / "train_chart_transform.py"), "--help"],
        [str(installed / "scripts" / "preprocess_guitar_windows.py"), "--help"],
        [str(installed / "scripts" / "train_guitar_v1.py"), "--help"],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        if "--json" in command:
            assert json.loads(completed.stdout)
