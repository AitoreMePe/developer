import os
import shutil


def generate_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def write_file(file_path: str, content: str):
    # if filepath doesn't exist, create it
    # ensure that file_path is a str
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        f.write(content)

# Define extensions to skip when reading files for debugging
EXTENSION_TO_SKIP = ['.md', '.txt', '.json', '.yaml', '.yml', '.log', '.git', '.svg', '.ico', '.png', '.jpg', '.jpeg', '.gif']

def read_file(filename: str) -> str | None:
    """Reads the content of a file. Returns None if an error occurs."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def walk_directory(directory: str) -> dict[str, str]:
    """
    Walks through a directory and reads the content of relevant files.
    Skips files with extensions defined in EXTENSION_TO_SKIP.
    Returns a dictionary mapping relative file paths to their content.
    """
    project_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            filename = os.path.join(root, file)
            if not any(filename.endswith(ext) for ext in EXTENSION_TO_SKIP):
                content = read_file(filename)
                if content is not None:
                    # Store with relative path from the input directory
                    relative_path = os.path.relpath(filename, directory)
                    project_files[relative_path] = content
    return project_files

def execute_generated_code(generate_folder_path, project_type):
    """
    Executes the generated code based on the project type and captures results.
    """
    if project_type == "js_html_css":
        index_html_path = os.path.join(generate_folder_path, "index.html")
        if not os.path.exists(index_html_path):
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error: index.html not found in {generate_folder_path}",
                "exit_code": 1,
            }
        # For now, just a placeholder for actual execution
        return {
            "success": True,
            "stdout": "Placeholder: HTML file found.",
            "stderr": "",
            "exit_code": 0,
        }
    elif project_type == "python":
        main_py_path = os.path.join(generate_folder_path, "main.py")
        app_py_path = os.path.join(generate_folder_path, "app.py")
        if not (os.path.exists(main_py_path) or os.path.exists(app_py_path)):
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error: main.py or app.py not found in {generate_folder_path}",
                "exit_code": 1,
            }
        # For now, just a placeholder for actual execution
        return {
            "success": True,
            "stdout": "Placeholder: Python file found.",
            "stderr": "",
            "exit_code": 0,
        }
    else:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error: Unknown project type '{project_type}'",
            "exit_code": 1,
        }
