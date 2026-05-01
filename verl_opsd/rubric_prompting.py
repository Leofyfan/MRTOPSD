from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True, slots=True)
class RubricPayload:
    core_correctness_rule: str
    core_key_steps_rule: str
    core_error_avoidance_rule: str
    free_rule: str = ""


@dataclass(frozen=True, slots=True)
class DynamicRubricCriterion:
    category: str
    criterion: str
    points: float


@dataclass(frozen=True, slots=True)
class DynamicRubricPayload:
    question_domain: str
    rubrics: tuple[DynamicRubricCriterion, ...]
    maximum_score: float
    minimum_score: float
    current_score: float


@dataclass(frozen=True, slots=True)
class DynamicRubricSpec:
    rubric_index: int
    criterion: DynamicRubricCriterion
    weight: float


class GenericRubricFactory:
    """Build generic rubrics for privileged math reasoning prompts."""

    def build_math_rubric(self) -> RubricPayload:
        return RubricPayload(
            core_correctness_rule=(
                "Verify the final answer directly against the problem requirements, "
                "and keep the derivation consistent with the given quantities."
            ),
            core_key_steps_rule=(
                "Track the essential algebra, arithmetic, or logical steps needed "
                "to reach the result without skipping the reasoning bridge."
            ),
            core_error_avoidance_rule=(
                "Avoid sign mistakes, unit mismatches, unsupported leaps, and "
                "answering a related but different question."
            ),
            free_rule=(
                "Prefer concise step-by-step reasoning that makes the decision points "
                "behind the answer easy to audit."
            ),
        )


def format_teacher_scoring_prompt(problem: str, rubric: RubricPayload) -> str:
    sections = [
        "You are a privileged math teacher scoring the student's response token-by-token.",
        f"Problem: {problem}",
        "Use the structured rubric below as privileged guidance while evaluating the student's response.",
        f"- Correctness: {rubric.core_correctness_rule}",
        f"- Key steps: {rubric.core_key_steps_rule}",
        f"- Avoid: {rubric.core_error_avoidance_rule}",
    ]
    if rubric.free_rule:
        sections.append(f"- Extra guidance: {rubric.free_rule}")
    sections.append(
        "Focus on whether each next token stays consistent with this rubric; do not generate an independent solution."
    )
    return "\n\n".join((sections[0], sections[1], "\n".join(sections[2:])))


def format_dynamic_teacher_scoring_prompt(
    problem: str,
    payload: DynamicRubricPayload,
    spec: DynamicRubricSpec,
) -> str:
    criterion = spec.criterion
    if criterion.points < 0:
        criterion_kind = "Penalty criterion. Favor next tokens that avoid this flaw."
    else:
        criterion_kind = "Reward criterion. Favor next tokens that satisfy this requirement."

    sections = [
        "You are a privileged math teacher scoring the student's response token-by-token.",
        f"Problem: {problem}",
        "Use exactly one rubric criterion as privileged guidance while evaluating the student's response.",
        f"Question domain: {payload.question_domain}",
        f"Category: {criterion.category}",
        f"Points: {criterion.points:g}",
        f"Criterion: {criterion.criterion}",
        criterion_kind,
        "Focus on whether each next token stays consistent with this criterion; do not generate an independent solution.",
    ]
    return "\n\n".join((sections[0], sections[1], "\n".join(sections[2:])))


def build_teacher_scoring_messages(prompt_text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt_text}]


def format_rubric_mining_prompt(
    problem: str,
    ground_truth: str,
    correct_example_summary: str,
    wrong_example_summary: str,
) -> str:
    return (
        "Read the paired student attempts and extract a compact rubric as JSON.\n\n"
        f"Problem: {problem}\n"
        f"Ground truth: {ground_truth}\n\n"
        f"Correct attempt summary: {correct_example_summary}\n"
        f"Wrong attempt summary: {wrong_example_summary}\n\n"
        "Return JSON with keys core_correctness_rule, core_key_steps_rule, "
        "core_error_avoidance_rule, and optional free_rule."
    )


def format_dynamic_rubric_mining_prompt(
    problem: str,
    ground_truth: str,
    response_summary: str,
    correct_example_summary: str = "",
) -> str:
    return (
        "You are an expert in educational assessment and rubric design. Your task is to analyze a given "
        "question-answer pair and generate comprehensive evaluation rubrics that can assess response quality "
        "for this question. The answer is only a reference answer from the student and is not necessarily good, "
        "so the rubric system should consider merits already present and room for further improvement.\n\n"
        "# Input Data\n"
        f"[Question]: {problem}\n"
        f"[Ground Truth]: {ground_truth}\n"
        f"[Response]: {response_summary}\n"
        f"[Optional Strong Attempt Summary]: {correct_example_summary}\n\n"
        "# Task Instructions\n"
        "Generate 5 to 15 binary-evaluable criteria covering problem decomposition and planning, "
        "backtracking/self-validation/error handling, reasoning clarity and flow, reasoning focus and "
        "efficiency, and other question-specific aspects. Positive criteria should describe merits or goals "
        "worth rewarding. Negative criteria should describe reasoning flaws or failures worth penalizing. "
        "Do not overfit criteria to this particular response; design them for future responses to the same "
        "question.\n\n"
        "# Output Format\n"
        "Return only a JSON object with keys question_domain, rubrics, maximum_score, minimum_score, and "
        "current_score. rubrics must be a list of objects with category, criterion, and points. points must "
        "be 1 to 5 for reward criteria and -5 to -1 for penalty criteria. maximum_score is the sum of positive "
        "points, minimum_score is the sum of negative points, and current_score should be strictly below the "
        "midpoint of the rubric range."
    )


def _rubric_base_weight(points: float, weight_mode: str, temperature: float) -> float:
    magnitude = abs(float(points))
    if magnitude <= 0:
        return 0.0
    if weight_mode == "softmax_points":
        temperature = max(float(temperature), 1e-6)
        return math.exp(magnitude / temperature)
    if weight_mode != "abs_points":
        raise ValueError(f"Unsupported dynamic rubric weight_mode: {weight_mode}")
    return magnitude


def _deterministic_unit_interval(seed_key: str) -> float:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def select_dynamic_rubric_specs(
    payload: DynamicRubricPayload,
    max_rubrics: int,
    selection_mode: str,
    weight_mode: str,
    weight_temperature: float,
    include_negative: bool,
    seed_key: str,
) -> list[DynamicRubricSpec]:
    candidates = [
        (idx, rubric)
        for idx, rubric in enumerate(payload.rubrics)
        if include_negative or rubric.points > 0
    ]
    if not candidates:
        return []

    if selection_mode == "top_m":
        limit = max(1, int(max_rubrics))
        selected = sorted(candidates, key=lambda item: (abs(item[1].points), -item[0]), reverse=True)[:limit]
    elif selection_mode == "ensemble_all":
        selected = candidates
        if max_rubrics and max_rubrics > 0:
            selected = selected[: int(max_rubrics)]
    elif selection_mode == "sample_one":
        weights = [
            _rubric_base_weight(rubric.points, weight_mode=weight_mode, temperature=weight_temperature)
            for _, rubric in candidates
        ]
        total = sum(weights)
        if total <= 0:
            selected = [candidates[0]]
        else:
            threshold = _deterministic_unit_interval(seed_key) * total
            running = 0.0
            selected = [candidates[-1]]
            for candidate, weight in zip(candidates, weights, strict=True):
                running += weight
                if running >= threshold:
                    selected = [candidate]
                    break
    else:
        raise ValueError(f"Unsupported dynamic rubric selection_mode: {selection_mode}")

    raw_weights = [
        _rubric_base_weight(rubric.points, weight_mode=weight_mode, temperature=weight_temperature)
        for _, rubric in selected
    ]
    weight_sum = sum(raw_weights)
    if weight_sum <= 0:
        normalized_weights = [1.0 / len(selected)] * len(selected)
    else:
        normalized_weights = [weight / weight_sum for weight in raw_weights]

    return [
        DynamicRubricSpec(rubric_index=idx, criterion=rubric, weight=weight)
        for (idx, rubric), weight in zip(selected, normalized_weights, strict=True)
    ]
