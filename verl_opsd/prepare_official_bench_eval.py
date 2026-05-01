from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from eval.dataset_registry import get_supported_dataset_names, load_registered_dataset
from verl_opsd.official_math_eval import build_official_math_messages, extract_boxed_answer
from verl_opsd.prepare_dataset import prompt_length


def _extract_problem_and_answer(dataset_name: str, row: dict) -> tuple[str, str, str]:
    dataset_name = dataset_name.lower()

    if dataset_name in {"aime24", "aime25", "hmmt25"}:
        problem = row["problem"]
        answer = str(row["answer"])
        problem_idx = row.get("problem_idx", row.get("id", row.get("uid", 0)))
        return problem, answer, str(problem_idx)

    if dataset_name in {"minerva", "amc23"}:
        problem = row["question"]
        answer = str(row["answer"])
        problem_idx = row.get("id", row.get("uid", 0))
        return problem, answer, str(problem_idx)

    if dataset_name in {"math500", "amo-bench"}:
        problem = row["problem"]
        solution = str(row["solution"])
        answer = extract_boxed_answer(solution) or solution
        problem_idx = row.get("id", row.get("uid", 0))
        return problem, answer, str(problem_idx)

    raise ValueError(f"Unsupported official bench dataset for parquet conversion: {dataset_name}")


def convert_split(dataset, dataset_name: str, tokenizer, max_student_prompt_length: int) -> list[dict]:
    records = []
    dropped = 0

    for idx, row in enumerate(dataset):
        problem, answer, raw_problem_id = _extract_problem_and_answer(dataset_name, row)
        problem_id = f"{dataset_name}-{raw_problem_id}"

        student_messages = build_official_math_messages(problem)
        student_prompt_length, _ = prompt_length(tokenizer, student_messages, enable_thinking=True)
        if student_prompt_length > max_student_prompt_length:
            dropped += 1
            continue

        records.append(
            {
                "prompt": student_messages,
                "data_source": dataset_name,
                "reward_model": {"ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "problem_id": problem_id,
                    "prompt_format_version": "official_math_eval_v1",
                    "student_prompt_length": student_prompt_length,
                },
                "uid": problem_id,
                "problem_id": problem_id,
                "problem": problem,
                "answer": answer,
                "source": dataset_name,
            }
        )

    print(f"Converted {len(records)} {dataset_name} rows. Dropped {dropped} for prompt overflow.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official math benchmark parquet data for verl validation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25", "hmmt25"],
        choices=get_supported_dataset_names(),
    )
    parser.add_argument("--model-path", default="/home/shenyl/hf/model/Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="/home/yuanfan/projects/MRTOPSD/data/benchmarks")
    parser.add_argument("--max-student-prompt-length", type=int, default=38912)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset = load_registered_dataset(dataset_name)
        if args.max_samples > 0:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        records = convert_split(
            dataset=dataset,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_student_prompt_length=args.max_student_prompt_length,
        )

        output_path = output_dir / f"{dataset_name}_official_eval.parquet"
        Dataset.from_list(records).to_parquet(str(output_path))
        print(f"Wrote {dataset_name} eval parquet to {output_path}")


if __name__ == "__main__":
    main()
