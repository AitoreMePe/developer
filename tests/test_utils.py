from smol_dev.utils import generate_folder


def test_generate_folder_preserves_contents(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    existing = folder / "old.txt"
    existing.write_text("hi")

    generate_folder(str(folder))

    assert existing.exists()


def test_generate_folder_clear(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "old.txt").write_text("hi")

    generate_folder(str(folder), clear=True)

    assert not (folder / "old.txt").exists()
    assert folder.exists()
