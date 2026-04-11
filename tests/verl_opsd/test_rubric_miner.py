from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl_opsd.rubric_memory import RolloutObservation
from verl_opsd.rubric_miner import build_rollout_observations, parse_rubric_response, select_hard_wrong_observation


def test_select_hard_wrong_observation_ignores_short_or_empty_responses():
    wrong_short = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="no",
        score=-1.0,
        acc=0.0,
        global_step=1,
    )
    wrong_good = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="nontrivial wrong reasoning with a final answer",
        score=-0.2,
        acc=0.0,
        global_step=2,
    )

    picked = select_hard_wrong_observation([wrong_short, wrong_good], min_response_chars=10)
    assert picked == wrong_good


def test_select_hard_wrong_observation_ignores_partial_credit_samples():
    partial_credit = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="partially correct but still not wrong enough",
        score=1.0,
        acc=0.5,
        global_step=3,
    )
    true_wrong = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="truly wrong reasoning with enough detail",
        score=0.2,
        acc=0.0,
        global_step=4,
    )

    picked = select_hard_wrong_observation([partial_credit, true_wrong], min_response_chars=10)
    assert picked == true_wrong


def test_parse_rubric_response_requires_all_core_slots():
    rubric = parse_rubric_response(
        '{"core_correctness_rule":"match the verified answer","core_key_steps_rule":"keep the key invariant","core_error_avoidance_rule":"avoid sign errors","free_rule":"check boundary cases"}'
    )
    assert rubric.core_correctness_rule == "match the verified answer"
    assert rubric.free_rule == "check boundary cases"


def test_parse_rubric_response_defaults_optional_free_rule_and_rejects_missing_core_field():
    rubric = parse_rubric_response(
        '{"core_correctness_rule":"match the verified answer","core_key_steps_rule":"keep the key invariant","core_error_avoidance_rule":"avoid sign errors"}'
    )
    assert rubric.free_rule == ""

    try:
        parse_rubric_response('{"core_correctness_rule":"match the verified answer","core_error_avoidance_rule":"avoid sign errors"}')
    except ValueError as exc:
        assert "core_key_steps_rule" in str(exc)
    else:
        raise AssertionError("Expected parse_rubric_response to reject missing core_key_steps_rule")


def test_build_rollout_observations_respects_response_mask_and_ignores_junk_tail_tokens():
    batch = SimpleNamespace(
        batch=TensorDict(
            {
                "responses": torch.tensor([[1, 2, 9, 8], [3, 4, 5, 7]], dtype=torch.int64),
                "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
            },
            batch_size=(2,),
        ),
        non_tensor_batch={
            "problem_id": np.array(["p1", "p2"], dtype=object),
            "problem": np.array(["Q1", "Q2"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}, {"ground_truth": "7"}], dtype=object),
        },
    )
    reward_extra_infos_dict = {
        "score": torch.tensor([-0.5, 0.75]),
        "acc": torch.tensor([0.0, 1.0]),
    }

    class DummyTokenizer:
        pad_token_id = 99

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(token)) for token in ids if int(token) != self.pad_token_id)

    observations = build_rollout_observations(
        batch=batch,
        reward_extra_infos_dict=reward_extra_infos_dict,
        global_step=17,
        tokenizer=DummyTokenizer(),
    )

    assert observations == [
        RolloutObservation(
            problem_id="p1",
            problem="Q1",
            ground_truth="42",
            response_text="1 2",
            score=-0.5,
            acc=0.0,
            global_step=17,
        ),
        RolloutObservation(
            problem_id="p2",
            problem="Q2",
            ground_truth="7",
            response_text="3 4 5",
            score=0.75,
            acc=1.0,
            global_step=17,
        ),
    ]


def test_build_rollout_observations_falls_back_to_extra_info_for_problem_fields():
    batch = SimpleNamespace(
        batch=TensorDict(
            {
                "responses": torch.tensor([[1, 2, 9]], dtype=torch.int64),
                "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
            },
            batch_size=(1,),
        ),
        non_tensor_batch={
            "extra_info": np.array(
                [{"problem_id": "p-extra", "problem": "Q-extra", "reward_model": {"ground_truth": "5"}}],
                dtype=object,
            ),
        },
    )
    reward_extra_infos_dict = {
        "score": torch.tensor([0.25]),
        "acc": torch.tensor([1.0]),
    }

    class DummyTokenizer:
        pad_token_id = 99

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(token)) for token in ids if int(token) != self.pad_token_id)

    observations = build_rollout_observations(
        batch=batch,
        reward_extra_infos_dict=reward_extra_infos_dict,
        global_step=9,
        tokenizer=DummyTokenizer(),
    )

    assert observations == [
        RolloutObservation(
            problem_id="p-extra",
            problem="Q-extra",
            ground_truth="5",
            response_text="1 2",
            score=0.25,
            acc=1.0,
            global_step=9,
        )
    ]


def test_build_rollout_observations_returns_empty_when_score_or_acc_are_missing():
    batch = SimpleNamespace(
        batch=TensorDict(
            {
                "responses": torch.tensor([[1, 2]], dtype=torch.int64),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            },
            batch_size=(1,),
        ),
        non_tensor_batch={
            "problem_id": np.array(["p1"], dtype=object),
            "problem": np.array(["Q1"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
        },
    )

    class DummyTokenizer:
        pad_token_id = 99

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(token)) for token in ids if int(token) != self.pad_token_id)

    assert build_rollout_observations(batch=batch, reward_extra_infos_dict={"score": [1.0]}, global_step=3, tokenizer=DummyTokenizer()) == []
    assert build_rollout_observations(batch=batch, reward_extra_infos_dict={"acc": [1.0]}, global_step=3, tokenizer=DummyTokenizer()) == []
