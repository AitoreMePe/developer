import os
import re
import openai

# utils are not available yet in this context.
# I expect an error here, and will fix the imports in a later step
# from smol_dev.utils import write_file, walk_directory

# Ensure OPENAI_API_KEY is accessible, e.g., set as an environment variable
if os.environ.get("OPENAI_API_KEY"):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
else:
    print("Warning: OPENAI_API_KEY environment variable not set. LLM debugging may not work.")

def get_llm_debugging_suggestions(original_prompt, generate_folder_path, error_details, model, walk_directory_fn, write_file_fn):
    """
    Queries an LLM for debugging suggestions.
    Uses walk_directory_fn and write_file_fn passed as arguments to avoid circular dependency.
    """
    project_files_content = walk_directory_fn(generate_folder_path)

    formatted_files_content = ""
    for file_path, content in project_files_content.items():
        formatted_files_content += f"\n--- File: {file_path} ---\n{content}\n--- End File: {file_path} ---\n"

    system_prompt = """
You are an expert software developer and debugger. The user has provided code that resulted in an error.
Your task is to analyze the original prompt, the generated code, and the error message.
Provide a correction for ONE file.
Your response MUST be in the following format:
FILEPATH: path/to/corrected/file.ext
```[language]
corrected code here
```
If you cannot determine a fix, or if multiple files need changes, please indicate that you cannot provide a single file fix.
"""

    user_prompt = f"""
Original Prompt for Code Generation:
{original_prompt}

Generated Code Files:
{formatted_files_content}

Error Details:
Success: {error_details.get('success')}
Stdout: {error_details.get('stdout')}
Stderr: {error_details.get('stderr')}
Exit Code: {error_details.get('exit_code')}

Based on the error, please provide the necessary correction for ONE file in the specified format.
Focus on the most likely file to fix the error.
If the error is related to a missing file (e.g., index.html not found), your correction should be to create that file.
The FILEPATH should be relative to the generated project's root directory.
For example, if the project is in 'generated_code' and the file is 'generated_code/src/app.js', the FILEPATH should be 'src/app.js'.
"""

    try:
        if not openai.api_key:
            return "OpenAI API key not configured. Cannot get debugging suggestions."

        print("\n--- Querying LLM for debugging suggestions... ---")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2, # Lower temperature for more deterministic fixes
            max_tokens=2048, # Adjust as needed
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API for debugging: {e}"

def parse_llm_correction(llm_response_content: str) -> tuple[str | None, str | None]:
    """
    Parses the LLM response to extract filepath and corrected code.
    """
    if llm_response_content is None:
        return None, None
    # Regex to find FILEPATH: followed by the path, and then a code block
    # It handles optional language specifier in the code block
    match = re.search(r"FILEPATH:\s*(.+?)\s*\n```(?:[a-zA-Z0-9]+)?\n(.*?)\n```", llm_response_content, re.DOTALL | re.IGNORECASE)
    if match:
        file_path = match.group(1).strip()
        corrected_code = match.group(2).strip()
        return file_path, corrected_code
    else:
        # print(f"DEBUG: No match found in LLM response: '{llm_response_content}'") # Optional debug
        return None, None

def apply_correction(generate_folder_path: str, file_path: str, corrected_code: str, write_file_fn):
    """
    Applies the corrected code to the specified file.
    Uses write_file_fn passed as an argument to avoid circular dependency.
    """
    full_path = os.path.join(generate_folder_path, file_path)
    try:
        write_file_fn(full_path, corrected_code)
        print(f"Applied correction to: {full_path}")
        return True
    except Exception as e:
        print(f"Error applying correction to {full_path}: {e}")
        return False
