import os
import re
import subprocess
import sys
from typing import Optional, Dict, Any


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


def run_and_fix(
    entrypoint: str,
    env_path: Optional[str] = None,
    container_runtime: Optional[str] = None,
    retries: int = 1,
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

    if attempt.returncode != 0 and error_info.get("missing_package") and retries > 0:
        package = error_info["missing_package"]
        install_proc = subprocess.run(
            pip_cmd + ["install", package], capture_output=True, text=True
        )
        attempt = _run(entrypoint, python_exec)
        error_info = _parse_error(attempt.stderr)
        return {
            "stdout": attempt.stdout,
            "stderr": attempt.stderr,
            "returncode": attempt.returncode,
            "error": error_info,
            "pip_stdout": install_proc.stdout,
            "pip_stderr": install_proc.stderr,
        }

    return {
        "stdout": attempt.stdout,
        "stderr": attempt.stderr,
        "returncode": attempt.returncode,
        "error": error_info,
    }
