from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from verl.utils.tokenizer import normalize_token_ids

from verl_opsd.rubric_memory import RolloutObservation, RubricMiningRequest
from verl_opsd.rubric_prompting import DynamicRubricCriterion, DynamicRubricPayload, RubricPayload

__all__ = [
    "build_rollout_observations",
    "build_dynamic_rubric_payload_from_response",
    "build_dynamic_rubric_payload_from_request",
    "build_rubric_payload_from_request",
    "parse_dynamic_rubric_response",
    "parse_rubric_response",
    "select_hard_wrong_observation",
    "summarize_response_text",
]


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _coerce_text(value: Any) -> str:
    value = _as_python_scalar(value)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_json_text(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].lstrip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _load_json_object(value: Any, error_prefix: str) -> dict:
    if isinstance(value, dict):
        return value

    normalized_text = _normalize_json_text(str(value))
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as exc:
        json_start = normalized_text.find("{")
        json_end = normalized_text.rfind("}")
        if json_start >= 0 and json_end > json_start:
            try:
                payload = json.loads(normalized_text[json_start : json_end + 1])
            except json.JSONDecodeError as nested_exc:
                raise ValueError(f"{error_prefix} must be valid JSON.") from nested_exc
        else:
            raise ValueError(f"{error_prefix} must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{error_prefix} must decode to a JSON object.")
    return payload


def select_hard_wrong_observation(
    observations: list[RolloutObservation] | tuple[RolloutObservation, ...],
    min_response_chars: int,
) -> RolloutObservation | None:
    candidates = [
        observation
        for observation in observations
        if observation.acc <= 0.0 and len(observation.response_text.strip()) >= min_response_chars
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda observation: (observation.score, observation.global_step))


def parse_rubric_response(response_text: str) -> RubricPayload:
    payload = _load_json_object(response_text, "Rubric response")

    def require_text(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Rubric response missing required non-empty field: {key}")
        return value.strip()

    free_rule = payload.get("free_rule", "")
    if free_rule is None:
        free_rule = ""
    elif not isinstance(free_rule, str):
        free_rule = str(free_rule)

    return RubricPayload(
        core_correctness_rule=require_text("core_correctness_rule"),
        core_key_steps_rule=require_text("core_key_steps_rule"),
        core_error_avoidance_rule=require_text("core_error_avoidance_rule"),
        free_rule=free_rule.strip(),
    )


def _normalize_dynamic_points(value: Any) -> float:
    try:
        points = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dynamic rubric criterion points must be numeric.") from exc
    if points == 0:
        raise ValueError("Dynamic rubric criterion points must be non-zero.")
    if points > 0:
        return min(max(points, 1.0), 5.0)
    return max(min(points, -1.0), -5.0)


def parse_dynamic_rubric_response(response_text: Any) -> DynamicRubricPayload:
    payload = _load_json_object(response_text, "Dynamic rubric response")

    question_domain = _coerce_text(payload.get("question_domain", "")).strip() or "unknown/unknown"
    raw_rubrics = payload.get("rubrics")
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        raise ValueError("Dynamic rubric response must include a non-empty rubrics list.")

    rubrics: list[DynamicRubricCriterion] = []
    for idx, raw_rubric in enumerate(raw_rubrics):
        if not isinstance(raw_rubric, dict):
            raise ValueError(f"Dynamic rubric at index {idx} must be a JSON object.")
        category = _coerce_text(raw_rubric.get("category", "")).strip()
        criterion = _coerce_text(raw_rubric.get("criterion", "")).strip()
        if not category or not criterion:
            raise ValueError(f"Dynamic rubric at index {idx} requires non-empty category and criterion.")
        rubrics.append(
            DynamicRubricCriterion(
                category=category,
                criterion=criterion,
                points=_normalize_dynamic_points(raw_rubric.get("points")),
            )
        )

    maximum_score = sum(rubric.points for rubric in rubrics if rubric.points > 0)
    minimum_score = sum(rubric.points for rubric in rubrics if rubric.points < 0)
    current_score = float(payload.get("current_score", 0.0) or 0.0)

    return DynamicRubricPayload(
        question_domain=question_domain,
        rubrics=tuple(rubrics),
        maximum_score=maximum_score,
        minimum_score=minimum_score,
        current_score=current_score,
    )


def summarize_response_text(response_text: str, max_chars: int = 9999999) -> str:
    normalized = " ".join(response_text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def build_rubric_payload_from_request(request: RubricMiningRequest) -> RubricPayload:
    target = request.ground_truth.strip() or "the verified answer"
    wrong_summary = summarize_response_text(request.wrong_observation.response_text, max_chars=9999999)
    return RubricPayload(
        core_correctness_rule=(
            f"Match {target} exactly and ensure the final answer resolves the stated problem."
        ),
        core_key_steps_rule=(
            "Preserve the key transformations needed to justify the solution instead of jumping to an answer."
        ),
        core_error_avoidance_rule=(
            f"Avoid repeating the failure pattern from the weak attempt: {wrong_summary}"
        ),
        free_rule="Check the final boxed answer before ending the solution.",
    )


def build_dynamic_rubric_payload_from_request(request: RubricMiningRequest) -> DynamicRubricPayload:
    return build_dynamic_rubric_payload_from_response(
        problem=request.problem,
        ground_truth=request.ground_truth,
        response_text=request.wrong_observation.response_text,
    )


def build_dynamic_rubric_payload_from_response(
    problem: str,
    ground_truth: str,
    response_text: str,
) -> DynamicRubricPayload:
    response_summary = summarize_response_text(response_text, max_chars=9999999)
    target = ground_truth.strip() or "the verified answer"
    rubrics = (
        DynamicRubricCriterion(
            category="Effectiveness of Problem Decomposition & Planning",
            criterion=f"Breaks this problem into the key mathematical subgoals before executing computations: {problem}",
            points=4.0,
        ),
        DynamicRubricCriterion(
            category="Effectiveness of BackTracking / Self-Validation / Error Handling",
            criterion="Checks intermediate results and corrects inconsistent reasoning before the final answer.",
            points=4.0,
        ),
        DynamicRubricCriterion(
            category="Reasoning Clarity & Flow",
            criterion="Makes each inference follow explicitly from prior statements and problem constraints.",
            points=3.0,
        ),
        DynamicRubricCriterion(
            category="Reasoning Focus & Efficiency",
            criterion=f"Repeats or extends the current response's weak pattern without resolving it: {response_summary}",
            points=-3.0,
        ),
        DynamicRubricCriterion(
            category="Other Question-Specific Aspects",
            criterion=f"Verifies that the final answer matches the target value or condition: {target}",
            points=5.0,
        ),
    )
    return DynamicRubricPayload(
        question_domain="math/general",
        rubrics=rubrics,
        maximum_score=sum(rubric.points for rubric in rubrics if rubric.points > 0),
        minimum_score=sum(rubric.points for rubric in rubrics if rubric.points < 0),
        current_score=min(0.0, sum(rubric.points for rubric in rubrics if rubric.points < 0)),
    )


def _extract_ground_truth(reward_model_entry: Any) -> str:
    reward_model_entry = _as_python_scalar(reward_model_entry)
    if isinstance(reward_model_entry, dict):
        return _coerce_text(reward_model_entry.get("ground_truth", ""))
    if reward_model_entry is None:
        return ""
    return _coerce_text(reward_model_entry)


def _valid_response_ids(responses: Any, response_mask: Any | None, pad_token_id: int | None) -> list[int]:
    if response_mask is not None:
        return normalize_token_ids(responses[response_mask.bool()])
    if pad_token_id is not None:
        return normalize_token_ids(responses[responses != pad_token_id])
    return normalize_token_ids(responses)


def build_rollout_observations(
    batch: Any,
    reward_extra_infos_dict: dict[str, Any],
    global_step: int,
    tokenizer: Any,
) -> list[RolloutObservation]:
    score_values = reward_extra_infos_dict.get("score")
    acc_values = reward_extra_infos_dict.get("acc")
    if score_values is None or acc_values is None:
        return []

    responses = batch.batch["responses"]
    prompts = batch.batch.get("prompts")
    response_mask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() else None
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    reward_model_batch = batch.non_tensor_batch.get("reward_model")
    extra_info_batch = batch.non_tensor_batch.get("extra_info")
    problem_id_batch = batch.non_tensor_batch.get("problem_id")
    problem_batch = batch.non_tensor_batch.get("problem")
    data_source_batch = batch.non_tensor_batch.get("data_source")
    uid_batch = batch.non_tensor_batch.get("uid")
    request_id_batch = batch.non_tensor_batch.get("request_id")

    observations: list[RolloutObservation] = []
    for idx in range(len(responses)):
        mask_row = None if response_mask is None else response_mask[idx]
        token_ids = _valid_response_ids(responses[idx], mask_row, pad_token_id)
        response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt_text = ""
        if prompts is not None:
            prompt_ids = _valid_response_ids(prompts[idx], None, pad_token_id)
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        extra_info = _as_python_scalar(None if extra_info_batch is None else extra_info_batch[idx])
        if extra_info is not None and not isinstance(extra_info, dict):
            extra_info = {"value": extra_info}
        problem_id = _coerce_text(None if problem_id_batch is None else problem_id_batch[idx])
        if not problem_id:
            problem_id = _coerce_text((extra_info or {}).get("problem_id"))
        problem = _coerce_text(None if problem_batch is None else problem_batch[idx])
        if not problem:
            problem = _coerce_text((extra_info or {}).get("problem"))
        teacher_prompt_text = _coerce_text((extra_info or {}).get("teacher_prompt_text"))
        data_source = _coerce_text(None if data_source_batch is None else data_source_batch[idx])
        uid = _coerce_text(None if uid_batch is None else uid_batch[idx])
        request_id = _coerce_text(None if request_id_batch is None else request_id_batch[idx])
        reward_model_entry = None if reward_model_batch is None else reward_model_batch[idx]
        if reward_model_entry is None:
            reward_model_entry = (extra_info or {}).get("reward_model")
        ground_truth = _extract_ground_truth(reward_model_entry)
        if not problem_id or not problem:
            continue
        score = float(_as_python_scalar(score_values[idx]))
        acc = float(_as_python_scalar(acc_values[idx]))

        observations.append(
            RolloutObservation(
                problem_id=problem_id,
                problem=problem,
                ground_truth=ground_truth,
                response_text=response_text,
                score=score,
                acc=acc,
                global_step=global_step,
                prompt_text=prompt_text,
                teacher_prompt_text=teacher_prompt_text,
                data_source=data_source,
                uid=uid,
                request_id=request_id,
                extra_info=extra_info,
            )
        )

    return observations
