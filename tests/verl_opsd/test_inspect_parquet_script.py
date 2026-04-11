import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_sample_parquet(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Problem: 1+1=?\n\nPlease reason step by step."}],
                "data_source": "math_dapo",
                "reward_model": {"ground_truth": "2"},
                "extra_info": {
                    "index": 0,
                    "problem_id": "opsd-problem-0",
                    "student_prompt_length": 12,
                    "teacher_prompt_length": 34,
                    "teacher_prompt_text": "teacher prompt example",
                },
                "uid": "opsd-0",
                "problem_id": "opsd-problem-0",
                "problem": "1+1=?",
                "solution": "Compute 1+1=2.",
                "answer": "2",
                "source": "unit-test",
            }
        ]
    )
    df.to_parquet(path)


def test_inspect_opsd_parquet_script_prints_summary_and_row(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    _write_sample_parquet(parquet_path)

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_opsd_parquet.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--path", str(parquet_path), "--row", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ROWS: 1" in result.stdout
    assert "problem_id: opsd-problem-0" in result.stdout
    assert "ground_truth: 2" in result.stdout
    assert "teacher prompt example" not in result.stdout


def test_inspect_opsd_parquet_script_can_show_teacher_prompt(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    _write_sample_parquet(parquet_path)

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_opsd_parquet.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--path", str(parquet_path), "--row", "0", "--show-teacher"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "teacher_prompt_text:" in result.stdout
    assert "teacher prompt example" in result.stdout
