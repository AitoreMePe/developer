import os
import subprocess
import sys
from types import SimpleNamespace

import smol_dev.self_heal as self_heal


# tests will patch self_heal._run to simulate script execution


def make_cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["python", "script.py"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_and_fix_installs_missing_package(monkeypatch):
    calls = {"runs": 0, "pip": 0}

    def fake_run(entry, python_exec, container_runtime=None):
        calls["runs"] += 1
        if calls["runs"] == 1:
            return make_cp(1, stderr="ModuleNotFoundError: No module named 'foo'")
        return make_cp(0, stdout="done")

    def fake_subprocess_run(cmd, capture_output=True, text=True):
        calls["pip"] += 1
        assert "install" in cmd
        return make_cp(0, stdout="installed")

    monkeypatch.setattr(self_heal, "_run", fake_run)
    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=fake_subprocess_run))

    result = self_heal.run_and_fix("entry.py")
    assert result["stdout"] == "done"
    assert result["pip_stdout"] == "installed"
    assert result["error"]["error_type"] is None
    assert calls["runs"] == 2
    assert calls["pip"] == 1


def test_run_and_fix_syntax_error(monkeypatch):
    def fake_run(entry, python_exec, container_runtime=None):
        return make_cp(1, stderr="SyntaxError: invalid syntax")

    def should_not_call(*a, **k):
        raise AssertionError("pip install should not run")

    monkeypatch.setattr(self_heal, "_run", fake_run)
    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=should_not_call))

    result = self_heal.run_and_fix("entry.py")
    assert result["error"]["error_type"] == "SyntaxError"
    assert "pip_stdout" not in result


def test_run_and_fix_two_missing_packages(monkeypatch):
    calls = {"runs": 0, "pip": []}

    def fake_run(entry, python_exec, container_runtime=None):
        calls["runs"] += 1
        if calls["runs"] == 1:
            return make_cp(1, stderr="ModuleNotFoundError: No module named 'foo'")
        if calls["runs"] == 2:
            return make_cp(1, stderr="ModuleNotFoundError: No module named 'bar'")
        return make_cp(stdout="done")

    def fake_subprocess_run(cmd, capture_output=True, text=True):
        pkg = cmd[-1]
        calls["pip"].append(pkg)
        return make_cp(stdout=f"installed {pkg}\n")

    monkeypatch.setattr(self_heal, "_run", fake_run)
    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=fake_subprocess_run))

    result = self_heal.run_and_fix("entry.py", retries=2)

    assert result["stdout"] == "done"
    assert calls["pip"] == ["foo", "bar"]
    assert calls["runs"] == 3
    assert "foo" in result.get("pip_stdout", "")
    assert "bar" in result.get("pip_stdout", "")


def test_container_runtime_launch(monkeypatch):
    cmd_capture = {}

    def fake_run(cmd, capture_output=True, text=True):
        cmd_capture["cmd"] = cmd
        return make_cp(stdout="ok")

    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=fake_run))

    result = self_heal.run_and_fix("entry.py", container_runtime="docker")

    cwd = os.getcwd()
    expected = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{cwd}:{cwd}",
        "-w",
        cwd,
        "python:3",
        sys.executable,
        "entry.py",
    ]

    assert cmd_capture["cmd"] == expected
    assert result["stdout"] == "ok"

