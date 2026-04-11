from verl_opsd.rubric_curriculum import RubricCurriculum
from verl_opsd.rubric_memory import RolloutObservation, RubricMemory
from verl_opsd.rubric_prompting import GenericRubricFactory, RubricPayload


def test_generic_rubric_has_required_slots():
    rubric = GenericRubricFactory().build_math_rubric()
    assert isinstance(rubric, RubricPayload)
    assert rubric.core_correctness_rule
    assert rubric.core_key_steps_rule
    assert rubric.core_error_avoidance_rule


def test_generic_rubric_free_rule_defaults_to_empty_string():
    rubric = RubricPayload(
        core_correctness_rule="correct",
        core_key_steps_rule="steps",
        core_error_avoidance_rule="avoidance",
    )
    assert rubric.free_rule == ""


def test_curriculum_mix_stage_is_deterministic_and_respects_boundaries():
    curriculum = RubricCurriculum(warmup_steps=10, mix_steps=10, seed=7)
    mix_step_first = curriculum.should_use_self_mined(problem_id="p1", global_step=12)
    mix_step_second = curriculum.should_use_self_mined(problem_id="p1", global_step=12)

    assert curriculum.should_use_self_mined(problem_id="p1", global_step=9) is False
    assert curriculum.should_use_self_mined(problem_id="p2", global_step=10) is True
    assert curriculum.should_use_self_mined(problem_id="p1", global_step=20) is True
    assert mix_step_first == mix_step_second


def test_memory_returns_mining_request_only_after_cross_step_pair_is_complete():
    memory = RubricMemory(min_response_chars=20)
    wrong_obs = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="wrong path but nontrivial",
        score=-1.0,
        acc=0.0,
        global_step=3,
    )
    correct_obs = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="detailed correct derivation",
        score=1.0,
        acc=1.0,
        global_step=4,
    )

    assert memory.observe(wrong_obs) is None
    request = memory.observe(correct_obs)

    assert request is not None
    assert request.problem_id == "p1"
    assert request.correct_observation.response_text == "detailed correct derivation"
    assert request.wrong_observation.response_text == "wrong path but nontrivial"


def test_memory_keeps_newest_wrong_observation_before_pair_emits():
    memory = RubricMemory(min_response_chars=20)
    newer_wrong_obs = RolloutObservation(
        problem_id="p3",
        problem="Q",
        ground_truth="42",
        response_text="newer wrong but nontrivial",
        score=-1.0,
        acc=0.0,
        global_step=10,
    )
    older_wrong_obs = RolloutObservation(
        problem_id="p3",
        problem="Q",
        ground_truth="42",
        response_text="older wrong path that should not win",
        score=-1.0,
        acc=0.0,
        global_step=3,
    )
    correct_obs = RolloutObservation(
        problem_id="p3",
        problem="Q",
        ground_truth="42",
        response_text="detailed correct derivation",
        score=1.0,
        acc=1.0,
        global_step=11,
    )

    assert memory.observe(newer_wrong_obs) is None
    assert memory.observe(older_wrong_obs) is None
    request = memory.observe(correct_obs)

    assert request is not None
    assert request.wrong_observation.response_text == "newer wrong but nontrivial"


def test_memory_keeps_harder_wrong_observation_before_pair_emits():
    memory = RubricMemory(min_response_chars=20)
    weaker_newer_wrong_obs = RolloutObservation(
        problem_id="p4",
        problem="Q",
        ground_truth="42",
        response_text="newer but weaker wrong attempt",
        score=-0.8,
        acc=0.0,
        global_step=10,
    )
    harder_older_wrong_obs = RolloutObservation(
        problem_id="p4",
        problem="Q",
        ground_truth="42",
        response_text="older but harder wrong attempt",
        score=-0.2,
        acc=0.0,
        global_step=3,
    )
    correct_obs = RolloutObservation(
        problem_id="p4",
        problem="Q",
        ground_truth="42",
        response_text="detailed correct derivation",
        score=1.0,
        acc=1.0,
        global_step=11,
    )

    assert memory.observe(weaker_newer_wrong_obs) is None
    assert memory.observe(harder_older_wrong_obs) is None
    request = memory.observe(correct_obs)

    assert request is not None
    assert request.wrong_observation.response_text == "older but harder wrong attempt"


def test_memory_ignores_short_responses():
    memory = RubricMemory(min_response_chars=20)
    short_wrong_obs = RolloutObservation(
        problem_id="p2",
        problem="Q",
        ground_truth="42",
        response_text="too short",
        score=-1.0,
        acc=0.0,
        global_step=1,
    )
    long_correct_obs = RolloutObservation(
        problem_id="p2",
        problem="Q",
        ground_truth="42",
        response_text="sufficiently detailed correct derivation",
        score=1.0,
        acc=1.0,
        global_step=2,
    )

    assert memory.observe(short_wrong_obs) is None
    assert memory.observe(long_correct_obs) is None
