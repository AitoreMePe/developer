import os
import shutil


def generate_folder(folder_path: str, clear: bool = False):
    """Ensure ``folder_path`` exists.

    Parameters
    ----------
    folder_path : str
        Directory to create.
    clear : bool, optional
        When ``True`` delete ``folder_path`` before creating it.
    """

    if os.path.exists(folder_path):
        if clear:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)


def write_file(file_path: str, content: str):
    # if filepath doesn't exist, create it
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        f.write(content)
