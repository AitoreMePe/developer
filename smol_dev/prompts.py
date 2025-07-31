import asyncio
import re
import json
from typing import List, Optional, Callable, Any

from smol_dev.llm import generate_chat
try:  # pragma: no cover - openai_function_call is optional
    from openai_function_call import openai_function
except Exception:  # pragma: no cover - simple fallback
    def openai_function(func: Callable) -> Callable:
        func.openai_schema = {"name": func.__name__}
        return func
try:  # pragma: no cover - tenacity is optional
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )
except Exception:  # pragma: no cover - minimal fallback when tenacity missing
    class stop_after_attempt:  # type: ignore
        def __init__(self, max_attempt_number: int):
            self.max_attempt_number = max_attempt_number

    def wait_random_exponential(*args: Any, **kwargs: Any) -> None:
        return None

    def retry(wait: Any = None, stop: Any = None):
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                async def wrapper(*args: Any, **kwargs: Any):
                    attempts = 0
                    max_attempts = getattr(stop, "max_attempt_number", 1)
                    while True:
                        try:
                            return await func(*args, **kwargs)
                        except Exception:
                            attempts += 1
                            if attempts >= max_attempts:
                                raise
            else:
                def wrapper(*args: Any, **kwargs: Any):
                    attempts = 0
                    max_attempts = getattr(stop, "max_attempt_number", 1)
                    while True:
                        try:
                            return func(*args, **kwargs)
                        except Exception:
                            attempts += 1
                            if attempts >= max_attempts:
                                raise
            return wrapper
        return decorator
import logging

logger = logging.getLogger(__name__)


SMOL_DEV_SYSTEM_PROMPT = """
You are a top tier AI developer who is trying to write a program that will generate code for the user based on their intent.
Do not leave any todos, fully implement every feature requested.

When writing code, add comments to explain what you intend to do and why it aligns with the program plan and specific instructions from the original prompt.
"""


@openai_function
def file_paths(files_to_edit: List[str]) -> List[str]:
    """
    Construct a list of strings.
    """
    # print("filesToEdit", files_to_edit)
    return files_to_edit


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def specify_file_paths(prompt: str, plan: str, model: str = 'gpt-3.5-turbo-0613', backend: str = "openai"):
    messages = [
        {
            "role": "system",
            "content": f"""{SMOL_DEV_SYSTEM_PROMPT}
      Given the prompt and the plan, return a list of strings corresponding to the new files that will be generated. Respond with a JSON list of strings.
                  """,
        },
        {"role": "user", "content": f""" I want a: {prompt} """},
        {"role": "user", "content": f""" The plan we have agreed on is: {plan} """},
    ]

    response_text = generate_chat(
        messages,
        model=model,
        backend=backend,
        functions=[file_paths.openai_schema],
        function_call={"name": "file_paths"} if backend == "openai" else None,
    )

    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "files_to_edit" in data:
            return data["files_to_edit"]
        return data
    except Exception:
        return [line.strip("- *") for line in response_text.splitlines() if line.strip()]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def plan(prompt: str, stream_handler: Optional[Callable[[bytes], None]] = None,
         model: str = 'gpt-3.5-turbo-0613',
         extra_messages: Optional[List[Any]] = None,
         backend: str = "openai"):
    extra_messages = extra_messages or []
    messages = [
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

    In response to the user's prompt, write a plan using GitHub Markdown syntax. Begin with a YAML description of the new files that will be created.
  In this plan, please name and briefly describe the structure of code that will be generated, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.
                Respond only with plans following the above schema.
                  """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            *extra_messages,
        ]
    response_text = generate_chat(messages, model=model, backend=backend)
    if stream_handler:
        try:
            stream_handler(response_text.encode("utf-8"))
        except Exception as err:
            logger.info("\nstream_handler error:", err)
    return response_text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def generate_code(prompt: str, plan: str, current_file: str, stream_handler: Optional[Callable[Any, Any]] = None,
                        model: str = 'gpt-3.5-turbo-0613', backend: str = "openai",
                        file_prompt: Optional[str] | None = None) -> str:
    messages = [
            {
                "role": "system",
                "content": f"""{SMOL_DEV_SYSTEM_PROMPT}

  In response to the user's prompt,
  Please name and briefly describe the structure of the app we will generate, including, for each file we are generating, what variables they export, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.

  We have broken up the program into per-file generation.
  Now your job is to generate only the code for the file: {current_file}

  only write valid code for the given filepath and file type, and return only the code.
  do not add any other explanation, only return valid code for that file type.
                  """,
            },
            {
                "role": "user",
                "content": f""" the plan we have agreed on is: {plan} """,
            },
            {
                "role": "user",
                "content": f""" the app prompt is: {prompt} """,
            },
            *([] if file_prompt is None else [{"role": "user", "content": file_prompt}]),
            {
                "role": "user",
                "content": f"""
    Make sure to have consistent filenames if you reference other files we are also generating.

    Remember that you must obey 3 things:
       - you are generating code for the file {current_file}
       - do not stray from the names of the files and the plan we have decided on
       - MOST IMPORTANT OF ALL - every line of code you generate must be valid code. Do not include code fences in your response, for example

    Bad response (because it contains the code fence):
    ```javascript
    console.log("hello world")
    ```

    Good response (because it only contains the code):
    console.log("hello world")

    Begin generating the code now.

    """,
            },
        ]
    response_text = generate_chat(messages, model=model, backend=backend)
    if stream_handler:
        try:
            stream_handler(response_text.encode("utf-8"))
        except Exception as err:
            logger.info("\nstream_handler error:", err)
    code_file = response_text

    pattern = r"```[\w\s]*\n([\s\S]*?)```"  # codeblocks at start of the string, less eager
    code_blocks = re.findall(pattern, code_file, re.MULTILINE)
    return code_blocks[0] if code_blocks else code_file


def generate_code_sync(prompt: str, plan: str, current_file: str,
                       stream_handler: Optional[Callable[Any, Any]] = None,
                       model: str = 'gpt-3.5-turbo-0613', backend: str = "openai",
                       file_prompt: Optional[str] | None = None) -> str:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(generate_code(prompt, plan, current_file, stream_handler, model, backend, file_prompt))
