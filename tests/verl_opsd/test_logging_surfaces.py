import json
from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import ValidationGenerationsLogger


class FakeWandbTable:
    def __init__(self, columns=None, data=None):
        self.columns = columns or []
        self.data = list(data or [])

    def add_data(self, *row):
        self.data.append(list(row))


class FakeWandbModule:
    def __init__(self):
        self.run = object()
        self.logged = []

    Table = FakeWandbTable

    def log(self, payload, step):
        self.logged.append((payload, step))


class _FakeWandbAuthError(RuntimeError):
    pass


class FailingWandbModule(FakeWandbModule):
    def log(self, payload, step):
        raise _FakeWandbAuthError("Unable to connect to server to verify API token.")


def test_validation_generations_logger_keeps_tables_separate_per_log_key():
    logger = ValidationGenerationsLogger(project_name="opsd", experiment_name="exp")
    fake_wandb = FakeWandbModule()

    logger._log_generations_to_wandb(
        samples=[("input-a", "output-a", -1.0)],
        step=0,
        wandb=fake_wandb,
        log_key="val/generations",
    )
    logger._log_generations_to_wandb(
        samples=[("input-b", "output-b", -1.0)],
        step=0,
        wandb=fake_wandb,
        log_key="val/error_generations",
    )

    assert len(fake_wandb.logged) == 2
    assert "val/generations" in fake_wandb.logged[0][0]
    assert "val/error_generations" in fake_wandb.logged[1][0]
    assert fake_wandb.logged[0][0]["val/generations"].data == [[0, "input-a", "output-a", -1.0]]
    assert fake_wandb.logged[1][0]["val/error_generations"].data == [[0, "input-b", "output-b", -1.0]]


def test_validation_generations_logger_downgrades_wandb_table_failures(caplog):
    logger = ValidationGenerationsLogger(project_name="opsd", experiment_name="exp")
    failing_wandb = FailingWandbModule()

    logger._log_generations_to_wandb(
        samples=[("input-a", "output-a", -1.0)],
        step=0,
        wandb=failing_wandb,
        log_key="val/generations",
    )

    assert "wandb table logging failed for val/generations at step 0" in caplog.text
    assert "Unable to connect to server to verify API token." in caplog.text


def test_maybe_log_val_generations_emits_error_only_table_for_negative_scores():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 3
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            log_val_generations=4,
            log_val_error_generations=4,
            logger=["console", "wandb"],
        )
    )
    calls = []
    trainer.validation_generations_logger = SimpleNamespace(
        log=lambda loggers, samples, step, log_key="val/generations": calls.append((log_key, list(samples), step))
    )

    trainer._maybe_log_val_generations(
        inputs=["q2", "q1", "q3"],
        outputs=["a2", "a1", "a3"],
        scores=[1.0, -1.0, -0.5],
    )

    assert calls[0] == (
        "val/generations",
        [("q1", "a1", -1.0), ("q2", "a2", 1.0), ("q3", "a3", -0.5)],
        3,
    )
    assert calls[1] == (
        "val/error_generations",
        [("q1", "a1", -1.0), ("q3", "a3", -0.5)],
        3,
    )


class _FakeItem:
    def __init__(self, ground_truth):
        self.non_tensor_batch = {"reward_model": {"ground_truth": ground_truth}}


class _FakeBatch:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "responses": torch.tensor([[5, 6], [7, 8]]),
            "token_level_scores": torch.tensor([[0.0, -1.0], [0.5, 0.5]], dtype=torch.float32),
        }
        self.non_tensor_batch = {"request_id": np.array(["req-1", "req-2"], dtype=object)}
        self._items = [_FakeItem("gt-wrong"), _FakeItem("gt-right")]

    def __iter__(self):
        return iter(self._items)


def test_log_rollout_data_can_dump_error_only_records():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)

    def _batch_decode(batch, **_kwargs):
        if int(batch[0][0]) in {1, 3}:
            return ["prompt-wrong", "prompt-right"]
        return ["output-wrong", "output-right"]

    trainer.tokenizer = SimpleNamespace(batch_decode=_batch_decode)
    calls = []
    trainer._dump_generations = lambda **kwargs: calls.append(kwargs)

    trainer._log_rollout_data(
        batch=_FakeBatch(),
        reward_extra_infos_dict={"acc": [0.0, 1.0]},
        timing_raw={},
        rollout_data_dir="/tmp/all",
        error_rollout_data_dir="/tmp/errors",
    )

    assert len(calls) == 2
    assert calls[0]["dump_path"] == "/tmp/all"
    assert calls[0]["scores"] == [-1.0, 1.0]
    assert calls[1]["dump_path"] == "/tmp/errors"
    assert calls[1]["inputs"] == ["prompt-wrong"]
    assert calls[1]["outputs"] == ["output-wrong"]
    assert calls[1]["scores"] == [-1.0]
    assert calls[1]["reward_extra_infos_dict"]["acc"] == [0.0]


def test_dump_generations_serializes_numpy_scalar_values(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 7

    trainer._dump_generations(
        inputs=["question"],
        outputs=["answer"],
        gts=["gt"],
        scores=[-1.0],
        reward_extra_infos_dict={"acc": [np.bool_(False)]},
        dump_path=str(tmp_path),
    )

    content = (tmp_path / "7.jsonl").read_text()
    record = json.loads(content.strip())
    assert record["acc"] is False
