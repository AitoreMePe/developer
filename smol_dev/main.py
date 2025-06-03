import sys
import time
import sys
import time
import os
# import openai # No longer directly used in main.py
# import re # No longer directly used in main.py

from smol_dev.prompts import plan, specify_file_paths, generate_code_sync
from smol_dev.utils import generate_folder, write_file, walk_directory, execute_generated_code
from smol_dev.autodebug import get_llm_debugging_suggestions, parse_llm_correction, apply_correction
import argparse

# model = "gpt-3.5-turbo-0613"
defaultmodel = "gpt-4-0613" # Ensure this is a model that supports the required context length and capabilities

# OpenAI API Key check is now handled in autodebug.py

def main(prompt, generate_folder_path="generated", debug=False, model: str = defaultmodel,
         enable_auto_correction=False, max_correction_attempts_arg=3, debug_model_arg=defaultmodel):
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

                sys.stdout.write("\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                sys.stdout.flush()
                stream_handler.count += len(chunk)

        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write("\033[0m\n") # remove the stdout line when streaming is complete

        shared_deps = plan(prompt, stream_handler, model=model)
    if debug:
        print(shared_deps)
    write_file(f"{generate_folder_path}/shared_deps.md", shared_deps)
    if debug:
        print("--------shared_deps---------")

    # specify file_paths
    if debug:
        print("--------specify_filePaths---------")
    file_paths = specify_file_paths(prompt, shared_deps, model=model)
    if debug:
        print(file_paths)
    if debug:
        print("--------file_paths---------")

    # loop through file_paths array and generate code for each file
    for file_path in file_paths:
        file_path = f"{generate_folder_path}/{file_path}"  # just append prefix
        if debug:
            print(f"--------generate_code: {file_path} ---------")

        start_time = time.time()
        def stream_handler(chunk):
            if debug:
                end_time = time.time()
                sys.stdout.write("\r \033[93mChars streamed\033[0m: {}. \033[93mChars per second\033[0m: {:.2f}".format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                sys.stdout.flush()
                stream_handler.count += len(chunk)
        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write("\033[0m\n") # remove the stdout line when streaming is complete
        code = generate_code_sync(prompt, shared_deps, file_path, stream_handler, model=model)
        if debug:
            print(code)
        if debug:
            print(f"--------generate_code: {file_path} ---------")
        # create file with code content
        write_file(file_path, code)

    print("--------smol dev done!---------")

    # Execute the generated code - initial attempt
    project_type = "js_html_css" # TODO: This should ideally be inferred or passed
    execution_result = execute_generated_code(generate_folder_path, project_type)

    print_execution_result(execution_result, "Initial Execution")

    if enable_auto_correction:
        current_max_attempts = max_correction_attempts_arg
        attempts_left = current_max_attempts

        for attempt in range(current_max_attempts):
            if execution_result["success"]:
                print("\nCode executed successfully!")
                break

            if attempts_left == 0: # This check should ideally be at the beginning of the loop or after decrementing
                print("\nMax correction attempts reached. Auto-correction failed.")
                break

            print(f"\n-------- Attempting LLM Auto-Correction (Attempt {attempt + 1}/{current_max_attempts}) ---------")

            debugging_suggestions_text = get_llm_debugging_suggestions(
                original_prompt=prompt,
                generate_folder_path=generate_folder_path,
                error_details=execution_result,
                model=debug_model_arg, # Use the specified debug_model
                walk_directory_fn=walk_directory, # Pass utility function
                write_file_fn=write_file # Pass utility function
            )

            if "Error calling OpenAI API" in debugging_suggestions_text or "OpenAI API key not configured" in debugging_suggestions_text :
            print(f"LLM Debugging API call failed: {debugging_suggestions_text}")
            print("Stopping auto-correction.")
            break

        print("\n--- LLM Response for Correction ---")
        print(debugging_suggestions_text)
        print("--- End LLM Response ---")

        file_to_correct, corrected_content = parse_llm_correction(debugging_suggestions_text)

        if file_to_correct and corrected_content:
            print(f"LLM suggests correcting file: {file_to_correct}")
            if apply_correction(generate_folder_path, file_to_correct, corrected_content, write_file_fn=write_file): # Pass utility function
                print("Re-executing code with correction...")
                execution_result = execute_generated_code(generate_folder_path, project_type)
                print_execution_result(execution_result, f"Execution after attempt {attempt + 1}")
                attempts_left -= 1
            else:
                print("Failed to apply correction. Stopping auto-correction.")
                break
        else:
            print("Could not parse correction from LLM response or LLM indicated no single file fix. Stopping auto-correction.")
            # print(f"DEBUG: LLM response was: '{debugging_suggestions_text}'") # Optional debug
            break
        else: # Executed if the loop completes without break (i.e., all attempts used up and still failing)
            if not execution_result["success"]: # Check success one last time
                 print(f"\nAuto-correction failed after {current_max_attempts} attempts.")
    elif not execution_result["success"]:
        print("\nExecution failed. Auto-correction is disabled.")


def print_execution_result(execution_result, step_name="Execution"):
    """Helper function to print execution results."""
    print(f"\n-------- {step_name} Result ---------")
    print(f"Success: {execution_result['success']}")
    if execution_result['stdout']:
        print(f"Stdout: {execution_result['stdout']}")
    if execution_result['stderr']:
        print(f"Stderr: {execution_result['stderr']}")
    print(f"Exit Code: {execution_result['exit_code']}")
    print("------------------------------------")

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
        
        parser = argparse.ArgumentParser(description="Smol Dev: A tool for generating and optionally auto-correcting code.")
        parser.add_argument("--prompt", type=str, required=True, help="Prompt for the app to be created.")
        parser.add_argument("--generate_folder_path", type=str, default="generated", help="Path of the folder for generated code.")
        parser.add_argument("--debug", type=bool, default=False, help="Enable or disable debug mode printouts.")
        parser.add_argument("--model", type=str, default=defaultmodel, help="OpenAI model to use for initial code generation.")
        parser.add_argument("--enable-auto-correction", action='store_true', help="Enable the auto-correction loop if initial execution fails.")
        parser.add_argument("--max-correction-attempts", type=int, default=3, help="Maximum number of attempts for auto-correction.")
        parser.add_argument("--debug-model", type=str, default=defaultmodel, help="OpenAI model to use for debugging suggestions.")

        args = parser.parse_args()
        # No need to check if args.prompt, as it's required
        prompt = args.prompt
        
    print(f"Using prompt: {prompt}")
    print(f"Generation Model: {args.model}")
    if args.enable_auto_correction:
        print(f"Auto-correction: Enabled")
        print(f"Max Correction Attempts: {args.max_correction_attempts}")
        print(f"Debug Model: {args.debug_model}")
    else:
        print(f"Auto-correction: Disabled")
        
    main(
        prompt=prompt,
        generate_folder_path=args.generate_folder_path,
        debug=args.debug,
        model=args.model, # Pass the generation model
        enable_auto_correction=args.enable_auto_correction,
        max_correction_attempts_arg=args.max_correction_attempts,
        debug_model_arg=args.debug_model
    )
