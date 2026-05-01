from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl_opsd.rubric_prompting import DynamicRubricPayload, RubricPayload

__all__ = [
    "RolloutObservation",
    "RubricEntry",
    "RubricMiningRequest",
    "RubricMemory",
]


@dataclass(frozen=True, slots=True)
class RolloutObservation:
    problem_id: str
    problem: str
    ground_truth: str
    response_text: str
    score: float
    acc: float
    global_step: int
    prompt_text: str = ""
    teacher_prompt_text: str = ""
    data_source: str = ""
    uid: str = ""
    request_id: str = ""
    extra_info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RubricEntry:
    problem_id: str
    rubric_source: str
    rubric_version: int
    updated_step: int
    course_stage: str
    rubric_payload: RubricPayload | DynamicRubricPayload
    correct_example_summary: str
    wrong_example_summary: str


@dataclass(frozen=True, slots=True)
class RubricMiningRequest:
    problem_id: str
    problem: str
    ground_truth: str
    correct_observation: RolloutObservation
    wrong_observation: RolloutObservation


@dataclass(slots=True)
class _ProblemObservations:
    correct_observation: RolloutObservation | None = None
    wrong_observation: RolloutObservation | None = None


class RubricMemory:
    def __init__(self, min_response_chars: int):
        self.min_response_chars = min_response_chars
        self._observations: dict[str, _ProblemObservations] = {}
        self._active_entries: dict[str, RubricEntry] = {}

    @staticmethod
    def _keep_newer(existing: RolloutObservation | None, candidate: RolloutObservation) -> RolloutObservation:
        if existing is None or candidate.global_step >= existing.global_step:
            return candidate
        return existing

    @staticmethod
    def _keep_harder(existing: RolloutObservation | None, candidate: RolloutObservation) -> RolloutObservation:
        if existing is None:
            return candidate
        if (candidate.score, candidate.global_step) >= (existing.score, existing.global_step):
            return candidate
        return existing

    def observe(self, observation: RolloutObservation) -> RubricMiningRequest | None:
        if len(observation.response_text) < self.min_response_chars:
            return None

        problem_observations = self._observations.setdefault(observation.problem_id, _ProblemObservations())
        if observation.acc >= 1.0:
            problem_observations.correct_observation = self._keep_newer(
                problem_observations.correct_observation,
                observation,
            )
        elif observation.acc <= 0.0:
            problem_observations.wrong_observation = self._keep_harder(
                problem_observations.wrong_observation,
                observation,
            )
        else:
            return None

        if problem_observations.correct_observation and problem_observations.wrong_observation:
            request = RubricMiningRequest(
                problem_id=observation.problem_id,
                problem=problem_observations.correct_observation.problem,
                ground_truth=problem_observations.correct_observation.ground_truth,
                correct_observation=problem_observations.correct_observation,
                wrong_observation=problem_observations.wrong_observation,
            )
            del self._observations[observation.problem_id]
            return request

        return None

    def store_active_entry(self, entry: RubricEntry) -> None:
        self._active_entries[entry.problem_id] = entry

    def get_active_entry(self, problem_id: str) -> RubricEntry | None:
        return self._active_entries.get(problem_id)
