import os
import subprocess
import sys
from pathlib import Path
import textwrap


def test_cli_execution(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import runpy
            import sys
            import unittest.mock
            from pathlib import Path
            import smol_dev.prompts as prompts

            def fake_plan(*args, **kwargs):
                return "plan"

            def fake_specify(*args, **kwargs):
                return ["main.py"]

            def fake_generate(*args, **kwargs):
                return "print('hi')"

            with unittest.mock.patch.object(prompts, 'plan', fake_plan), \
                 unittest.mock.patch.object(prompts, 'specify_file_paths', fake_specify), \
                 unittest.mock.patch.object(prompts, 'generate_code_sync', fake_generate):
                sys.argv = ['smol_dev.main', '--prompt', 'x', '--generate_folder_path', str(Path('out'))]
                runpy.run_module('smol_dev.main', run_name='__main__')
            """
        )
    )

    env = dict(**os.environ)
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert (tmp_path / "out" / "main.py").exists()
    assert (tmp_path / "out" / "shared_deps.md").exists()


def test_cli_requires_hf_model(tmp_path):
    runner = tmp_path / "runner_hf.py"
    runner.write_text(
        textwrap.dedent(
            """
            import runpy
            import sys
            import unittest.mock
            from pathlib import Path
            import smol_dev.prompts as prompts

            def fake_plan(*args, **kwargs):
                return "plan"

            def fake_specify(*args, **kwargs):
                return ["main.py"]

            def fake_generate(*args, **kwargs):
                return "print('hi')"

            with unittest.mock.patch.object(prompts, 'plan', fake_plan), \
                 unittest.mock.patch.object(prompts, 'specify_file_paths', fake_specify), \
                 unittest.mock.patch.object(prompts, 'generate_code_sync', fake_generate):
                sys.argv = ['smol_dev.main', '--prompt', 'x', '--generate_folder_path', str(Path('out')), '--backend', 'hf']
                runpy.run_module('smol_dev.main', run_name='__main__')
            """
        )
    )

    env = dict(**os.environ)
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode != 0
