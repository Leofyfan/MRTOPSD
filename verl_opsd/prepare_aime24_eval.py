from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from verl_opsd.prepare_dataset import build_student_messages, prompt_length


def _first_non_empty(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return None


def _extract_problem(row: dict) -> str:
    problem = _first_non_empty(row, ["problem", "question", "prompt", "Question"])
    if problem is None:
        raise KeyError(f"Unable to find problem text in row keys: {sorted(row.keys())}")
    return problem


def _extract_answer(row: dict) -> str:
    answer = _first_non_empty(row, ["answer", "Answer", "final_answer", "solution"])
    if answer is None:
        raise KeyError(f"Unable to find answer text in row keys: {sorted(row.keys())}")
    return answer


def convert_split(dataset: Dataset, tokenizer, max_student_prompt_length: int) -> list[dict]:
    records = []
    dropped = 0

    for idx, row in enumerate(dataset):
        problem = _extract_problem(row)
        answer = _extract_answer(row)
        raw_problem_id = row.get("problem_id") or row.get("id") or row.get("uid") or idx
        problem_id = f"aime24-{raw_problem_id}"

        student_messages = build_student_messages(problem)
        student_prompt_length, _ = prompt_length(tokenizer, student_messages, enable_thinking=False)
        if student_prompt_length > max_student_prompt_length:
            dropped += 1
            continue

        records.append(
            {
                "prompt": student_messages,
                "data_source": "aime24",
                "reward_model": {"ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "problem_id": problem_id,
                    "prompt_format_version": "opsd_boxed_last_line_v2",
                    "student_prompt_length": student_prompt_length,
                },
                "uid": problem_id,
                "problem_id": problem_id,
                "problem": problem,
                "answer": answer,
                "source": "HuggingFaceH4/aime_2024",
            }
        )

    print(f"Converted {len(records)} AIME24 rows. Dropped {dropped} for student prompt overflow.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIME24 parquet data for integrated verl validation.")
    parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-path", default="/home/shenyl/hf/model/Qwen/Qwen3-1.7B")
    parser.add_argument("--output-file", default="/home/yuanfan/projects/MRTOPSD/data/benchmarks/aime24_eval.parquet")
    parser.add_argument("--max-student-prompt-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset(args.dataset, split=args.split)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    records = convert_split(
        dataset,
        tokenizer,
        max_student_prompt_length=args.max_student_prompt_length,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    print(f"Wrote AIME24 eval parquet to {output_path}")


if __name__ == "__main__":
    main()
