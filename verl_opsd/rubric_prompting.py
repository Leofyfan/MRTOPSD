from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RubricPayload:
    core_correctness_rule: str
    core_key_steps_rule: str
    core_error_avoidance_rule: str
    free_rule: str = ""


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
