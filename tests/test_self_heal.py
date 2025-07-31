import subprocess
from types import SimpleNamespace

import smol_dev.self_heal as self_heal


# tests will patch self_heal._run to simulate script execution


def make_cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["python", "script.py"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_and_fix_installs_missing_package(monkeypatch):
    calls = {"runs": 0, "pip": 0}

    def fake_run(entry, python_exec):
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
    calls = {"gen": 0, "runs": 0}

    def fake_run(entry, python_exec):
        calls["runs"] += 1
        return make_cp(1, stderr="SyntaxError: invalid syntax")

    def fake_generate(*a, **k):
        calls["gen"] += 1
        return "bad"

    def should_not_call(*a, **k):
        raise AssertionError("pip install should not run")

    monkeypatch.setattr(self_heal, "_run", fake_run)
    monkeypatch.setattr(self_heal.prompts, "generate_code_sync", fake_generate)
    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=should_not_call))

    result = self_heal.run_and_fix("entry.py", fix_retries=1)
    assert result["error"]["error_type"] == "SyntaxError"
    assert "pip_stdout" not in result
    assert calls["gen"] == 1
    assert calls["runs"] == 2


def test_run_and_fix_two_missing_packages(monkeypatch):
    calls = {"runs": 0, "pip": []}

    def fake_run(entry, python_exec):
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


def test_run_and_fix_llm_fix(monkeypatch, tmp_path):
    file_path = tmp_path / "bad.py"
    file_path.write_text("print(\n")

    calls = {"runs": 0, "gen": 0}

    def fake_run(entry, python_exec):
        calls["runs"] += 1
        if calls["runs"] == 1:
            return make_cp(1, stderr="IndentationError: unexpected indent")
        return make_cp(stdout="ok")

    def fake_generate(prompt, plan, current_file, **kwargs):
        calls["gen"] += 1
        fp = kwargs.get("file_prompt", "")
        assert "unexpected indent" in fp
        assert "print(" in fp
        return "print('fixed')\n"

    monkeypatch.setattr(self_heal, "_run", fake_run)
    monkeypatch.setattr(self_heal.prompts, "generate_code_sync", fake_generate)
    monkeypatch.setattr(self_heal, "subprocess", SimpleNamespace(run=lambda *a, **k: make_cp()))

    result = self_heal.run_and_fix(str(file_path), fix_retries=1)

    assert result["stdout"] == "ok"
    assert file_path.read_text() == "print('fixed')\n"
    assert calls["gen"] == 1
    assert calls["runs"] == 2

