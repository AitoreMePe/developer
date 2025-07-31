import os
import subprocess
import sys
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


def test_cli_watch_triggers_generation(tmp_path):
    runner = tmp_path / "runner_watch.py"
    runner.write_text(
        textwrap.dedent(
            """
            import sys
            import unittest.mock
            from pathlib import Path
            import smol_dev.main as smain
            import time

            def fake_plan(*a, **k):
                return "plan"

            def fake_specify(*a, **k):
                return ["main.py"]

            calls = {"n": 0}

            def fake_generate(*a, **k):
                calls["n"] += 1
                return "print('hi')"

            def fake_watch(path, cb, interval=1.0):
                cb()

            with unittest.mock.patch.object(smain, 'plan', fake_plan), \
                 unittest.mock.patch.object(smain, 'specify_file_paths', fake_specify), \
                 unittest.mock.patch.object(smain, 'generate_code_sync', fake_generate), \
                 unittest.mock.patch.object(smain, 'watch_prompt_file', fake_watch), \
                 unittest.mock.patch.object(time, 'strftime', lambda fmt: 'ts'):
                p = Path('prompt.md')
                p.write_text('x')
                # replicate CLI logic
                def run_once():
                    smain.main(
                        prompt=p.read_text(),
                        generate_folder_path='out_ts',
                        backend='openai',
                    )

                run_once()
                smain.watch_prompt_file(str(p), run_once)

            print('COUNT', calls['n'])
            """
        )
    )

    env = dict(**os.environ)
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "COUNT 2" in result.stdout
