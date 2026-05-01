from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


FINAL_ANSWER_INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n\n"
    f"{FINAL_ANSWER_INSTRUCTION}"
)


def build_student_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"Problem: {problem}\n\n{FINAL_ANSWER_INSTRUCTION}",
        }
    ]


def build_teacher_messages(problem: str, solution: str) -> list[dict[str, str]]:
    teacher_user_message = (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}"
    )
    return [{"role": "user", "content": teacher_user_message}]


def prompt_length(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> tuple[int, str]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return len(prompt_ids), prompt_text


def convert_split(
    dataset: Dataset,
    tokenizer,
    max_student_prompt_length: int,
    max_teacher_prompt_length: int,
) -> list[dict]:
    records = []
    dropped_student = 0
    dropped_teacher = 0

    for idx, row in enumerate(dataset):
        sample_idx = row.get("global_row_idx", idx)
        problem = row["problem"]
        solution = row["solution"]
        answer = row.get("Answer") or row.get("answer") or ""
        problem_id = f"opsd-problem-{sample_idx}"

        student_messages = build_student_messages(problem)
        teacher_messages = build_teacher_messages(problem, solution)

        student_prompt_length, _ = prompt_length(tokenizer, student_messages, enable_thinking=False)
        if student_prompt_length > max_student_prompt_length:
            dropped_student += 1
            continue

        teacher_prompt_length, teacher_prompt_text = prompt_length(tokenizer, teacher_messages, enable_thinking=True)
        if teacher_prompt_length > max_teacher_prompt_length:
            dropped_teacher += 1
            continue

        records.append(
            {
                "prompt": student_messages,
                "data_source": "math_dapo",
                "reward_model": {"ground_truth": answer},
                "extra_info": {
                    "index": sample_idx,
                    "problem_id": problem_id,
                    "prompt_format_version": "opsd_boxed_last_line_v2",
                    "teacher_prompt_text": teacher_prompt_text,
                    "student_prompt_length": student_prompt_length,
                    "teacher_prompt_length": teacher_prompt_length,
                },
                "uid": f"opsd-{sample_idx}",
                "problem_id": problem_id,
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "source": row.get("source", ""),
            }
        )

    print(
        f"Converted {len(records)} rows. Dropped {dropped_student} for student prompt overflow and "
        f"{dropped_teacher} for teacher prompt overflow."
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OPSD parquet data for verl.")
    parser.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--model-path", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--output-dir", default="/root/MRTOPSD/data/processed")
    parser.add_argument("--train-file", default="opsd_train.parquet")
    parser.add_argument("--val-file", default="opsd_val.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-student-prompt-length", type=int, default=20000)
    parser.add_argument("--max-teacher-prompt-length", type=int, default=38912)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-val-samples", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.add_column("global_row_idx", list(range(len(dataset))))
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_split = split["train"]
    val_split = split["test"]

    if args.max_train_samples > 0:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
    if args.max_val_samples > 0:
        val_split = val_split.select(range(min(args.max_val_samples, len(val_split))))

    train_records = convert_split(
        train_split,
        tokenizer,
        max_student_prompt_length=args.max_student_prompt_length,
        max_teacher_prompt_length=args.max_teacher_prompt_length,
    )
    val_records = convert_split(
        val_split,
        tokenizer,
        max_student_prompt_length=args.max_student_prompt_length,
        max_teacher_prompt_length=args.max_teacher_prompt_length,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / args.train_file
    val_path = output_dir / args.val_file

    Dataset.from_list(train_records).to_parquet(str(train_path))
    Dataset.from_list(val_records).to_parquet(str(val_path))

    print(f"Wrote train parquet to {train_path}")
    print(f"Wrote val parquet to {val_path}")


if __name__ == "__main__":
    main()
