from __future__ import annotations

import os
from functools import lru_cache

from math_verify import parse, verify
from transformers import AutoTokenizer

OFFICIAL_MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DEFAULT_OFFICIAL_EVAL_TOKENIZER_PATH = os.getenv(
    "OFFICIAL_EVAL_TOKENIZER_PATH", "/home/shenyl/hf/model/Qwen/Qwen3-1.7B"
)


def build_official_math_user_message(problem: str) -> str:
    return f"{problem}{OFFICIAL_MATH_SUFFIX}"


def build_official_math_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_official_math_user_message(problem)}]


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False

    try:
        pred = predicted if "$" in predicted else f"${predicted}$"
        gt = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        pred_parsed = parse(pred, fallback_mode="no_fallback")
        gt_parsed = parse(gt, fallback_mode="no_fallback")
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


@lru_cache(maxsize=1)
def _get_debug_tokenizer():
    return AutoTokenizer.from_pretrained(DEFAULT_OFFICIAL_EVAL_TOKENIZER_PATH, trust_remote_code=True)


def _count_retokenized_tokens(text: str) -> int:
    tokenizer = _get_debug_tokenizer()
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def compute_official_math_score(solution_str: str, ground_truth: str) -> dict[str, object]:
    # token_count = _count_retokenized_tokens(solution_str)
    # print(f"== eval response token count(retokenized)==: {token_count}")
    print(f"== eval model response==: \n\n {solution_str[-200:]}")
    predicted = extract_boxed_answer(solution_str)
    print(f"=*****= eval model extracted predicted=***=:  {predicted}")
    formatted = predicted is not None
    acc = grade_answer(predicted, ground_truth)
    print(f"== eval acc==:  {acc}")
    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "pred": predicted,
        "formatted": formatted,
    }
