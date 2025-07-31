import os
import re
import subprocess
import sys
from typing import Optional, Dict, Any

import smol_dev.prompts as prompts


def _parse_error(stderr: str) -> Dict[str, Any]:
    """Parse stderr for common Python errors.

    Returns a dict with keys ``error_type`` and ``missing_package`` if detected.
    """
    result: Dict[str, Any] = {"error_type": None, "missing_package": None}
    if not stderr:
        return result

    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        pkg_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
        if pkg_match:
            result["missing_package"] = pkg_match.group(1)
        result["error_type"] = (
            "ModuleNotFoundError" if "ModuleNotFoundError" in stderr else "ImportError"
        )
        return result

    if "SyntaxError" in stderr:
        result["error_type"] = "SyntaxError"
    elif "IndentationError" in stderr:
        result["error_type"] = "IndentationError"

    return result


def _run(entrypoint: str, python_exec: str) -> subprocess.CompletedProcess:
    """Run the entrypoint using the provided python executable."""
    return subprocess.run([python_exec, entrypoint], capture_output=True, text=True)


def _fix_file_with_llm(file_path: str, error_message: str) -> str:
    """Use the LLM to fix a file given an error message."""
    original = ""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            original = fh.read()
    except Exception:
        pass
    file_prompt = (
        f"The following file has an error:\n{error_message}\n\n" f"{original}\n"
        "Please return the corrected file contents only."
    )
    return prompts.generate_code_sync(
        "fix file", "fix", file_path, file_prompt=file_prompt
    )


def run_and_fix(
    entrypoint: str,
    env_path: Optional[str] = None,
    container_runtime: Optional[str] = None,
    retries: int = 1,
    fix_retries: int = 1,
) -> Dict[str, Any]:
    """Execute ``entrypoint`` and attempt to fix missing dependencies.

    Parameters
    ----------
    entrypoint:
        The script to execute.
    env_path:
        Optional path to a virtualenv. If provided, ``python`` and ``pip`` from
        this environment will be used.
    container_runtime:
        Optional container runtime. Currently unused but reserved for future
        support.
    retries:
        Number of times to retry execution after installing dependencies.
    fix_retries:
        Number of times to attempt LLM-based fixes for syntax errors.
    """
    python_exec = (
        os.path.join(env_path, "bin", "python") if env_path else sys.executable
    )
    pip_exec = (
        os.path.join(env_path, "bin", "pip")
        if env_path
        else [sys.executable, "-m", "pip"]
    )

    if isinstance(pip_exec, list):
        pip_cmd = pip_exec
    else:
        pip_cmd = [pip_exec]

    attempt = _run(entrypoint, python_exec)
    error_info = _parse_error(attempt.stderr)

    pip_stdout: list[str] = []
    pip_stderr: list[str] = []

    while attempt.returncode != 0 and (
        (error_info.get("missing_package") and retries > 0)
        or (
            error_info.get("error_type") in ("SyntaxError", "IndentationError")
            and fix_retries > 0
        )
    ):
        if error_info.get("missing_package") and retries > 0:
            package = error_info["missing_package"]
            install_proc = subprocess.run(
                pip_cmd + ["install", package], capture_output=True, text=True
            )
            pip_stdout.append(install_proc.stdout)
            pip_stderr.append(install_proc.stderr)
            retries -= 1
        elif (
            error_info.get("error_type") in ("SyntaxError", "IndentationError")
            and fix_retries > 0
        ):
            fixed = _fix_file_with_llm(entrypoint, attempt.stderr)
            try:
                with open(entrypoint, "w", encoding="utf-8") as fh:
                    fh.write(fixed)
            except Exception:
                pass
            fix_retries -= 1
        else:
            break

        attempt = _run(entrypoint, python_exec)
        error_info = _parse_error(attempt.stderr)

    result = {
        "stdout": attempt.stdout,
        "stderr": attempt.stderr,
        "returncode": attempt.returncode,
        "error": error_info,
    }

    if pip_stdout or pip_stderr:
        result["pip_stdout"] = "".join(pip_stdout)
        result["pip_stderr"] = "".join(pip_stderr)

    return result
