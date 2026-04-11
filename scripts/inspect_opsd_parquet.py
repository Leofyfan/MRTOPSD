#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def truncate(text: Any, limit: int = 30000) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def print_file_summary(path: Path) -> None:
    table = pq.read_table(path)
    print(f"FILE: {path}")
    print(f"ROWS: {table.num_rows}")
    print(f"COLUMNS: {table.column_names}")
    print("SCHEMA:")
    print(table.schema)
    print("-" * 80)


def print_row_summary(row_index: int, row: pd.Series, show_teacher: bool) -> None:
    prompt = row.get("prompt") or []
    prompt_role = prompt[0].get("role", "") if prompt else ""
    prompt_text = prompt[0].get("content", "") if prompt else ""
    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}

    print(f"ROW {row_index}")
    print(f"uid: {row.get('uid', '')}")
    print(f"problem_id: {row.get('problem_id', '')}")
    print(f"data_source: {row.get('data_source', '')}")
    print(f"source: {row.get('source', '')}")
    print(f"prompt_role: {prompt_role}")
    print(f"prompt_head: {truncate(prompt_text)}")
    print(f"ground_truth: {reward_model.get('ground_truth', '')}")
    print(f"answer: {row.get('answer', '')}")
    print(f"student_prompt_length: {extra_info.get('student_prompt_length', '')}")
    print(f"teacher_prompt_length: {extra_info.get('teacher_prompt_length', '')}")
    print(f"problem_head: {truncate(row.get('problem', ''))}")
    print(f"solution_head: {truncate(row.get('solution', ''))}")
    if show_teacher:
        print(f"teacher_prompt_text: {truncate(extra_info.get('teacher_prompt_text', ''), limit=120000)}")
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OPSD parquet files.")
    parser.add_argument("--path", required=True, help="Path to the parquet file.")
    parser.add_argument("--row", type=int, action="append", help="Specific row index to inspect. Can be used multiple times.")
    parser.add_argument("--head", type=int, default=1, help="Number of rows to preview when --row is not set.")
    parser.add_argument("--show-teacher", action="store_true", help="Print teacher_prompt_text from extra_info.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing parquet file: {path}")

    print_file_summary(path)
    df = pd.read_parquet(path)

    if len(df) == 0:
        print("No rows found.")
        return

    if args.row:
        row_indices = args.row
    else:
        row_indices = list(range(min(args.head, len(df))))

    for row_index in row_indices:
        if row_index < 0 or row_index >= len(df):
            print(f"Skipping out-of-range row index: {row_index}")
            continue
        print_row_summary(row_index, df.iloc[row_index], show_teacher=args.show_teacher)


if __name__ == "__main__":
    main()
