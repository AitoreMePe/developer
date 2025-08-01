import sys
import time
import os
from pathlib import Path

from smol_dev.prompts import plan, specify_file_paths, generate_code_sync
from smol_dev.utils import generate_folder, write_file
from typing import Optional
import argparse

# model = "gpt-3.5-turbo-0613"
defaultmodel = "gpt-4-0613"


def watch_prompt_file(prompt_path: str | Path, callback, interval: float = 1.0) -> None:
    """Watch ``prompt_path`` and call ``callback`` when it changes.

    Uses ``watchdog`` if available, otherwise falls back to polling using
    ``Path.stat``. Runs until interrupted.
    """

    path = Path(prompt_path)
    try:
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import FileSystemEventHandler  # type: ignore

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(event.src_path) == path:
                    callback()

        observer = Observer()
        observer.schedule(Handler(), str(path.parent), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
    except Exception:
        last = path.stat().st_mtime
        try:
            while True:
                time.sleep(interval)
                current = path.stat().st_mtime
                if current != last:
                    last = current
                    callback()
        except KeyboardInterrupt:
            pass


def main(
    prompt,
    generate_folder_path: str = "generated",
    debug: bool = False,
    model: str = defaultmodel,
    backend: str = "openai",
    hf_model: str | None = None,
    file_prompts_dir: str = ".",
    self_heal: bool = False,
    venv_path: Optional[str] = None,
    container_runtime: Optional[str] = None,
):
    if backend == "hf" and hf_model is None:
        raise ValueError("hf_model must be specified when using the hf backend")
    # create generateFolder folder if doesnt exist
    generate_folder(generate_folder_path)

    # plan shared_deps
    if debug:
        print("--------shared_deps---------")
    with open(f"{generate_folder_path}/shared_deps.md", "wb") as f:

        start_time = time.time()

        def stream_handler(chunk):
            f.write(chunk)
            if debug:
                end_time = time.time()

                sys.stdout.write(
                    "\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(
                        stream_handler.count,
                        stream_handler.count / (end_time - start_time),
                    )
                )
                sys.stdout.flush()
                stream_handler.count += len(chunk)

        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write(
            "\033[0m\n"
        )  # remove the stdout line when streaming is complete

        model_name = hf_model if backend == "hf" else model
        shared_deps = plan(prompt, stream_handler, model=model_name, backend=backend)
    if debug:
        print(shared_deps)
    write_file(f"{generate_folder_path}/shared_deps.md", shared_deps)
    if debug:
        print("--------shared_deps---------")

    # specify file_paths
    if debug:
        print("--------specify_filePaths---------")
    file_paths = specify_file_paths(
        prompt, shared_deps, model=model_name, backend=backend
    )
    if debug:
        print(file_paths)
    if debug:
        print("--------file_paths---------")

    # loop through file_paths array and generate code for each file
    for file_path in file_paths:
        prompt_path = f"{file_prompts_dir}/{file_path}.md"
        file_specific_prompt = None
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as fp:
                file_specific_prompt = fp.read()

        file_path = f"{generate_folder_path}/{file_path}"  # just append prefix
        if debug:
            print(f"--------generate_code: {file_path} ---------")

        start_time = time.time()

        def stream_handler(chunk):
            if debug:
                end_time = time.time()
                sys.stdout.write(
                    "\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(
                        stream_handler.count,
                        stream_handler.count / (end_time - start_time),
                    )
                )
                sys.stdout.flush()
                stream_handler.count += len(chunk)

        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write(
            "\033[0m\n"
        )  # remove the stdout line when streaming is complete
        code = generate_code_sync(
            prompt,
            shared_deps,
            file_path,
            stream_handler,
            model=model_name,
            backend=backend,
            file_prompt=file_specific_prompt,
        )
        if debug:
            print(code)
        if debug:
            print(f"--------generate_code: {file_path} ---------")
        # create file with code content
        write_file(file_path, code)

    print("--------smol dev done!---------")

    if self_heal:
        entry = os.path.join(generate_folder_path, "main.py")
        if os.path.exists(entry):
            from smol_dev.self_heal import run_and_fix

            result = run_and_fix(
                entry, env_path=venv_path, container_runtime=container_runtime
            )
            if result.get("stdout"):
                print(result["stdout"])
            if result.get("stderr"):
                print(result["stderr"], file=sys.stderr)
        else:
            print(f"Self-heal requested but {entry} not found.")


# for local testing
# python main.py --prompt "a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG..." --generate_folder_path "generated" --debug True

if __name__ == "__main__":
    prompt = """
  a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG. 
  The left paddle is controlled by the player, following where the mouse goes.
  The right paddle is controlled by a simple AI algorithm, which slowly moves the paddle toward the ball at every frame, with some probability of error.
  Make the canvas a 400 x 400 black square and center it in the app.
  Make the paddles 100px long, yellow and the ball small and red.
  Make sure to render the paddles and name them so they can controlled in javascript.
  Implement the collision detection and scoring as well.
  Every time the ball bouncess off a paddle, the ball should move faster.
  It is meant to run in Chrome browser, so dont use anything that is not supported by Chrome, and don't use the import and export keywords.
  """
    if len(sys.argv) == 2:
        prompt = sys.argv[1]
    else:

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt",
            type=str,
            required=True,
            help="Prompt for the app to be created.",
        )
        parser.add_argument(
            "--generate_folder_path",
            type=str,
            default="generated",
            help="Path of the folder for generated code.",
        )
        parser.add_argument(
            "--debug", type=bool, default=False, help="Enable or disable debug mode."
        )
        parser.add_argument(
            "--backend",
            choices=["openai", "hf"],
            default="openai",
            help="LLM backend to use",
        )
        parser.add_argument(
            "--hf-model",
            type=str,
            help="Local path or HF repo id for transformers model",
        )
        parser.add_argument(
            "--file-prompts-dir",
            type=str,
            default=".",
            help="Directory containing per-file prompt markdown files",
        )
        parser.add_argument(
            "--self-heal",
            action="store_true",
            help="Run generated code and attempt to install missing dependencies",
        )
        parser.add_argument(
            "--venv-path",
            type=str,
            default=None,
            help="Path to virtualenv for executing generated code",
        )
        parser.add_argument(
            "--container-runtime",
            type=str,
            default=None,
            help="Container runtime to use for execution",
        )
        parser.add_argument(
            "--watch",
            action="store_true",
            help="Watch the prompt file and regenerate on changes",
        )
        args = parser.parse_args()
        if args.prompt:
            prompt = args.prompt

    print(prompt)

    def run_once():
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = (
            args.generate_folder_path
            if not args.watch
            else f"{args.generate_folder_path}_{ts}"
        )
        main(
            prompt=(Path(args.prompt).read_text() if args.watch else prompt),
            generate_folder_path=out_dir,
            debug=args.debug,
            backend=args.backend,
            hf_model=args.hf_model,
            file_prompts_dir=args.file_prompts_dir,
            self_heal=args.self_heal,
            venv_path=args.venv_path,
            container_runtime=args.container_runtime,
        )

    run_once()

    if args.watch:
        watch_prompt_file(args.prompt, run_once)
