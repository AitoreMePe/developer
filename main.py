import sys
import time
from pathlib import Path

from smol_dev.main import main, watch_prompt_file
import argparse


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
        args = None
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, help="Prompt for the app to be created."
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gpt-4-0613",
            help="model to use. can also use gpt-3.5-turbo-0613",
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

    # read file from prompt if it ends in a .md filetype
    if len(prompt) < 100 and prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print(prompt)

    if args is None:
        # This is in case we're just calling the main function directly with a prompt
        main(prompt=prompt)
    else:
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
                model=args.model,
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
