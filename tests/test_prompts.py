import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import smol_dev.prompts as prompts


def test_retry_specify_file_paths():
    call_count = {"count": 0}

    def fake_generate_chat(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise ValueError("fail")
        return '["file1.py"]'

    with patch.object(prompts, "generate_chat", side_effect=fake_generate_chat):
        result = prompts.specify_file_paths("prompt", "plan")
    assert result == ["file1.py"]
    assert call_count["count"] == 2


def test_retry_plan():
    call_count = {"count": 0}

    def fake_generate_chat(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise ValueError("fail")
        return "plan-text"

    with patch.object(prompts, "generate_chat", side_effect=fake_generate_chat):
        result = prompts.plan("prompt")
    assert result == "plan-text"
    assert call_count["count"] == 3
