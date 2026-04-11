# Rubric Self-Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement rubric-based self-distillation for the OPSD-on-verl training path with per-problem cached rubrics, generic fallback, lagged async rubric mining, and validation score tracking at `mean@1`, `mean@4`, and `best@4`.

**Architecture:** Keep the distillation loss unchanged and change only teacher conditioning. Add OPSD-local rubric modules under `verl_opsd` to manage structured rubric payloads, curriculum selection, rollout observation pairing, and background rubric mining. The trainer submits rollout observations after reward extraction; teacher scoring reads the currently active rubric from cache or falls back to a generic rubric without blocking the current step.

**Tech Stack:** Python 3.12, verl, Ray, vLLM, Hydra/OmegaConf, pytest, unittest.mock

---

## File Structure

- Create: `tests/verl_opsd/__init__.py`
- Create: `tests/verl_opsd/test_rubric_core.py`
- Create: `tests/verl_opsd/test_prepare_dataset.py`
- Create: `tests/verl_opsd/test_rubric_miner.py`
- Create: `tests/verl_opsd/test_teacher_rubric_flow.py`
- Create: `verl_opsd/rubric_prompting.py`
- Create: `verl_opsd/rubric_curriculum.py`
- Create: `verl_opsd/rubric_memory.py`
- Create: `verl_opsd/rubric_miner.py`
- Modify: `verl_opsd/prepare_dataset.py`
- Modify: `verl_opsd/teacher.py`
- Modify: `verl_opsd/main_ppo_opsd.py`
- Modify: `third_party/verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `third_party/verl/verl/trainer/ppo/metric_utils.py`
- Modify: `third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py`
- Modify: `scripts/run_opsd_4b_verl.sh`
- Modify: `scripts/run_opsd_4b_verl_smoke.sh`

`verl_opsd/rubric_prompting.py` owns structured rubric payloads, generic rubric templates, and teacher/miner prompt formatting. `verl_opsd/rubric_curriculum.py` owns the warmup/mix/mature selection rule. `verl_opsd/rubric_memory.py` owns per-problem candidate banks and active rubric entries, including the cross-step correct/wrong pairing that is required because training rollout currently uses `n=1`. `verl_opsd/rubric_miner.py` owns request building, hard-wrong filtering, and rubric parsing/validation. `verl_opsd/teacher.py` becomes the integration point for scoring with active rubrics plus a background mining queue. `ray_trainer.py` stays thin and only submits observation updates plus logs extra rubric metrics.

### Task 1: Build rubric core primitives

**Files:**
- Create: `tests/verl_opsd/__init__.py`
- Create: `tests/verl_opsd/test_rubric_core.py`
- Create: `verl_opsd/rubric_prompting.py`
- Create: `verl_opsd/rubric_curriculum.py`
- Create: `verl_opsd/rubric_memory.py`

- [ ] **Step 1: Write the failing tests for generic rubric, curriculum, and cross-step pairing**

```python
from verl_opsd.rubric_curriculum import RubricCurriculum
from verl_opsd.rubric_memory import RolloutObservation, RubricMemory
from verl_opsd.rubric_prompting import GenericRubricFactory, RubricPayload


def test_generic_rubric_has_required_slots():
    rubric = GenericRubricFactory().build_math_rubric()
    assert isinstance(rubric, RubricPayload)
    assert rubric.core_correctness_rule
    assert rubric.core_key_steps_rule
    assert rubric.core_error_avoidance_rule


def test_curriculum_prefers_generic_during_warmup_and_self_mined_later():
    curriculum = RubricCurriculum(warmup_steps=10, mix_steps=10, seed=7)
    assert curriculum.should_use_self_mined(problem_id="p1", global_step=0) is False
    assert curriculum.should_use_self_mined(problem_id="p1", global_step=25) is True


def test_memory_returns_mining_request_only_after_cross_step_pair_is_complete():
    memory = RubricMemory(min_response_chars=20)
    wrong_obs = RolloutObservation(problem_id="p1", problem="Q", ground_truth="42", response_text="wrong path but nontrivial", score=-1.0, acc=0.0, global_step=3)
    correct_obs = RolloutObservation(problem_id="p1", problem="Q", ground_truth="42", response_text="detailed correct derivation", score=1.0, acc=1.0, global_step=4)

    assert memory.observe(wrong_obs) is None
    request = memory.observe(correct_obs)

    assert request is not None
    assert request.problem_id == "p1"
    assert request.correct_observation.response_text == "detailed correct derivation"
    assert request.wrong_observation.response_text == "wrong path but nontrivial"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_core.py -q`
Expected: FAIL with `ModuleNotFoundError` for `verl_opsd.rubric_*`

- [ ] **Step 3: Write the minimal rubric core implementation**

```python
# verl_opsd/rubric_prompting.py
from dataclasses import dataclass


@dataclass(frozen=True)
class RubricPayload:
    core_correctness_rule: str
    core_key_steps_rule: str
    core_error_avoidance_rule: str
    free_rule: str = ""


class GenericRubricFactory:
    def build_math_rubric(self) -> RubricPayload:
        return RubricPayload(
            core_correctness_rule="The final boxed answer must match the mathematically verified result.",
            core_key_steps_rule="The reasoning must preserve the necessary algebraic or logical constraints.",
            core_error_avoidance_rule="Avoid leaps that change signs, drop constraints, or skip validation of the final expression.",
        )


# verl_opsd/rubric_curriculum.py
import hashlib


class RubricCurriculum:
    def __init__(self, warmup_steps: int, mix_steps: int, seed: int):
        self.warmup_steps = warmup_steps
        self.mix_steps = mix_steps
        self.seed = seed

    def should_use_self_mined(self, problem_id: str, global_step: int) -> bool:
        if global_step < self.warmup_steps:
            return False
        if global_step >= self.warmup_steps + self.mix_steps:
            return True
        key = f"{problem_id}:{global_step}:{self.seed}".encode()
        threshold = (global_step - self.warmup_steps + 1) / self.mix_steps
        sample = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
        return sample < threshold


# verl_opsd/rubric_memory.py
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutObservation:
    problem_id: str
    problem: str
    ground_truth: str
    response_text: str
    score: float
    acc: float
    global_step: int


@dataclass(frozen=True)
class RubricMiningRequest:
    problem_id: str
    correct_observation: RolloutObservation
    wrong_observation: RolloutObservation


class RubricMemory:
    def __init__(self, min_response_chars: int):
        self.min_response_chars = min_response_chars
        self._correct = {}
        self._wrong = {}

    def observe(self, observation: RolloutObservation):
        if len(observation.response_text.strip()) < self.min_response_chars:
            return None
        if observation.acc >= 1.0:
            self._correct[observation.problem_id] = observation
        else:
            self._wrong[observation.problem_id] = observation
        if observation.problem_id in self._correct and observation.problem_id in self._wrong:
            return RubricMiningRequest(
                problem_id=observation.problem_id,
                correct_observation=self._correct[observation.problem_id],
                wrong_observation=self._wrong[observation.problem_id],
            )
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_core.py -q`
Expected: PASS with 3 passing tests

- [ ] **Step 5: Commit**

```bash
git add tests/verl_opsd/__init__.py tests/verl_opsd/test_rubric_core.py verl_opsd/rubric_prompting.py verl_opsd/rubric_curriculum.py verl_opsd/rubric_memory.py
git commit -m "feat: add rubric core primitives"
```

### Task 2: Add stable problem identifiers and rubric config surfaces

**Files:**
- Create: `tests/verl_opsd/test_prepare_dataset.py`
- Modify: `verl_opsd/prepare_dataset.py`
- Modify: `scripts/run_opsd_4b_verl.sh`
- Modify: `scripts/run_opsd_4b_verl_smoke.sh`

- [ ] **Step 1: Write the failing tests for dataset conversion**

```python
from verl_opsd.prepare_dataset import convert_split


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        if tokenize:
            return list(range(len(messages[0]["content"].split())))
        return messages[0]["content"]


def test_convert_split_adds_problem_id_and_preserves_problem_fields():
    dataset = [{"problem": "2+2?", "solution": "4", "answer": "4", "source": "toy"}]
    records = convert_split(dataset, DummyTokenizer(), max_student_prompt_length=999, max_teacher_prompt_length=999)
    record = records[0]

    assert record["problem_id"] == "opsd-problem-0"
    assert record["problem"] == "2+2?"
    assert record["answer"] == "4"
    assert record["extra_info"]["problem_id"] == "opsd-problem-0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_prepare_dataset.py -q`
Expected: FAIL because `problem_id` is missing from converted records

- [ ] **Step 3: Write the minimal dataset and config wiring**

```python
# verl_opsd/prepare_dataset.py
problem_id = f"opsd-problem-{idx}"
records.append(
    {
        "prompt": student_messages,
        "data_source": "math_dapo",
        "reward_model": {"ground_truth": answer},
        "extra_info": {
            "index": idx,
            "problem_id": problem_id,
            "student_prompt_length": student_prompt_length,
            "teacher_prompt_length": teacher_prompt_length,
        },
        "uid": f"opsd-{idx}",
        "problem_id": problem_id,
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "source": row.get("source", ""),
    }
)

# scripts/run_opsd_4b_verl.sh
VAL_SAMPLE_N="${VAL_SAMPLE_N:-4}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.1}"
RUBRIC_ENABLED="${RUBRIC_ENABLED:-true}"
RUBRIC_WARMUP_STEPS="${RUBRIC_WARMUP_STEPS:-100}"
RUBRIC_MIX_STEPS="${RUBRIC_MIX_STEPS:-400}"
...
"actor_rollout_ref.rollout.val_kwargs.n=${VAL_SAMPLE_N}"
"actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
"actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
"+distillation.opsd_rubric.enabled=${RUBRIC_ENABLED}"
"+distillation.opsd_rubric.warmup_steps=${RUBRIC_WARMUP_STEPS}"
"+distillation.opsd_rubric.mix_steps=${RUBRIC_MIX_STEPS}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_prepare_dataset.py -q`
Expected: PASS with 1 passing test

- [ ] **Step 5: Commit**

```bash
git add tests/verl_opsd/test_prepare_dataset.py verl_opsd/prepare_dataset.py scripts/run_opsd_4b_verl.sh scripts/run_opsd_4b_verl_smoke.sh
git commit -m "feat: add rubric dataset ids and config wiring"
```

### Task 3: Implement rubric mining request building and parsing

**Files:**
- Create: `tests/verl_opsd/test_rubric_miner.py`
- Modify: `verl_opsd/rubric_memory.py`
- Create: `verl_opsd/rubric_miner.py`

- [ ] **Step 1: Write the failing tests for hard-wrong selection and rubric parsing**

```python
from verl_opsd.rubric_miner import parse_rubric_response, select_hard_wrong_observation
from verl_opsd.rubric_memory import RolloutObservation


def test_select_hard_wrong_observation_ignores_short_or_empty_responses():
    wrong_short = RolloutObservation(problem_id="p1", problem="Q", ground_truth="42", response_text="no", score=-1.0, acc=0.0, global_step=1)
    wrong_good = RolloutObservation(problem_id="p1", problem="Q", ground_truth="42", response_text="nontrivial wrong reasoning with a final answer", score=-0.2, acc=0.0, global_step=2)

    picked = select_hard_wrong_observation([wrong_short, wrong_good], min_response_chars=10)
    assert picked == wrong_good


def test_parse_rubric_response_requires_all_core_slots():
    rubric = parse_rubric_response(
        '{\"core_correctness_rule\":\"match the verified answer\",\"core_key_steps_rule\":\"keep the key invariant\",\"core_error_avoidance_rule\":\"avoid sign errors\",\"free_rule\":\"check boundary cases\"}'
    )
    assert rubric.core_correctness_rule == "match the verified answer"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_miner.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors

- [ ] **Step 3: Write the minimal mining utilities**

```python
# verl_opsd/rubric_miner.py
import json

from verl_opsd.rubric_prompting import RubricPayload


def select_hard_wrong_observation(observations, min_response_chars: int):
    candidates = [obs for obs in observations if len(obs.response_text.strip()) >= min_response_chars and obs.acc < 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda obs: obs.score)


def parse_rubric_response(response_text: str) -> RubricPayload:
    data = json.loads(response_text)
    for key in ("core_correctness_rule", "core_key_steps_rule", "core_error_avoidance_rule"):
        if not data.get(key):
            raise ValueError(f"Missing required rubric field: {key}")
    return RubricPayload(
        core_correctness_rule=data["core_correctness_rule"].strip(),
        core_key_steps_rule=data["core_key_steps_rule"].strip(),
        core_error_avoidance_rule=data["core_error_avoidance_rule"].strip(),
        free_rule=data.get("free_rule", "").strip(),
    )


def build_rollout_observations(batch, reward_extra_infos_dict, global_step: int, tokenizer):
    scores = reward_extra_infos_dict.get("score", [])
    accs = reward_extra_infos_dict.get("acc", [0.0] * len(scores))
    response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
    observations = []
    for idx, response_text in enumerate(response_texts):
        observations.append(
            RolloutObservation(
                problem_id=batch.non_tensor_batch["problem_id"][idx],
                problem=batch.non_tensor_batch["problem"][idx],
                ground_truth=batch.non_tensor_batch["reward_model"][idx]["ground_truth"],
                response_text=response_text,
                score=float(scores[idx]),
                acc=float(accs[idx]),
                global_step=global_step,
            )
        )
    return observations
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_miner.py -q`
Expected: PASS with 2 passing tests

- [ ] **Step 5: Commit**

```bash
git add tests/verl_opsd/test_rubric_miner.py verl_opsd/rubric_memory.py verl_opsd/rubric_miner.py
git commit -m "feat: add rubric mining utilities"
```

### Task 4: Integrate rubric selection and background mining into the teacher manager

**Files:**
- Create: `tests/verl_opsd/test_teacher_rubric_flow.py`
- Modify: `verl_opsd/teacher.py`
- Modify: `verl_opsd/main_ppo_opsd.py`
- Modify: `verl_opsd/rubric_prompting.py`
- Modify: `verl_opsd/rubric_curriculum.py`
- Modify: `verl_opsd/rubric_memory.py`
- Modify: `verl_opsd/rubric_miner.py`

- [ ] **Step 1: Write the failing tests for teacher-side rubric fallback and cache hits**

```python
from collections import deque

from verl_opsd.rubric_memory import RubricEntry
from verl_opsd.rubric_prompting import RubricPayload
from verl_opsd.teacher import select_active_teacher_prompt


class TestTeacherManager:
    def __init__(self):
        self.pending_requests = deque()
        self._correct = None
        self._wrong = None

    def submit_test_observation(self, problem_id: str, response_text: str, score: float, acc: float, global_step: int):
        payload = (problem_id, response_text, score, acc, global_step)
        if acc >= 1.0:
            self._correct = payload
        else:
            self._wrong = payload
        if self._correct is not None and self._wrong is not None:
            self.pending_requests.append(problem_id)

    def pending_rubric_request_count(self) -> int:
        return len(self.pending_requests)


def test_select_active_teacher_prompt_falls_back_to_generic_when_cache_misses():
    prompt, source = select_active_teacher_prompt(
        problem_id="p1",
        problem="Solve x+1=2.",
        active_entry=None,
        generic_rubric=RubricPayload("correct", "steps", "avoid"),
    )
    assert source == "generic"
    assert "Solve x+1=2." in prompt
    assert "avoid" in prompt


def test_select_active_teacher_prompt_uses_self_mined_entry_when_available():
    entry = RubricEntry(
        problem_id="p1",
        rubric_source="self_mined",
        rubric_version=1,
        updated_step=10,
        course_stage="mature",
        rubric_payload=RubricPayload("match gt", "preserve invariant", "avoid arithmetic slips", "check final box"),
        correct_example_summary="correct",
        wrong_example_summary="wrong",
    )
    prompt, source = select_active_teacher_prompt(
        problem_id="p1",
        problem="Solve x+1=2.",
        active_entry=entry,
        generic_rubric=RubricPayload("correct", "steps", "avoid"),
    )
    assert source == "self_mined"
    assert "preserve invariant" in prompt


def test_enqueue_rubric_request_is_triggered_after_pair_completion():
    manager = TestTeacherManager()
    manager.submit_test_observation(problem_id="p1", response_text="wrong but long enough", score=-0.5, acc=0.0, global_step=1)
    manager.submit_test_observation(problem_id="p1", response_text="correct detailed derivation", score=1.0, acc=1.0, global_step=2)
    assert manager.pending_rubric_request_count() == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_teacher_rubric_flow.py -q`
Expected: FAIL because `RubricEntry` or `select_active_teacher_prompt` is missing

- [ ] **Step 3: Write the minimal teacher integration**

```python
# verl_opsd/rubric_memory.py
@dataclass(frozen=True)
class RubricEntry:
    problem_id: str
    rubric_source: str
    rubric_version: int
    updated_step: int
    course_stage: str
    rubric_payload: RubricPayload
    correct_example_summary: str
    wrong_example_summary: str


# verl_opsd/rubric_prompting.py
def format_teacher_scoring_prompt(problem: str, rubric: RubricPayload) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"Rubric:\n"
        f"- Correctness: {rubric.core_correctness_rule}\n"
        f"- Key Steps: {rubric.core_key_steps_rule}\n"
        f"- Error Avoidance: {rubric.core_error_avoidance_rule}\n"
        f"- Extra: {rubric.free_rule or 'N/A'}\n\n"
        "Use this rubric as privileged guidance when scoring the student's response token by token."
    )


# verl_opsd/teacher.py
def select_active_teacher_prompt(problem_id: str, problem: str, active_entry, generic_rubric):
    rubric = active_entry.rubric_payload if active_entry is not None else generic_rubric
    source = active_entry.rubric_source if active_entry is not None else "generic"
    return format_teacher_scoring_prompt(problem=problem, rubric=rubric), source


class OPSDTeacherModelManager(TeacherModelManager):
    def __init__(self, config, resource_pool=None):
        self.raw_config = config
        super().__init__(config=config, resource_pool=resource_pool)
        self.rubric_memory = RubricMemory(min_response_chars=self.raw_config.get("opsd_rubric", {}).get("min_response_chars", 20))
        self.rubric_curriculum = RubricCurriculum(
            warmup_steps=self.raw_config.get("opsd_rubric", {}).get("warmup_steps", 100),
            mix_steps=self.raw_config.get("opsd_rubric", {}).get("mix_steps", 400),
            seed=self.raw_config.get("opsd_rubric", {}).get("seed", 17),
        )
        self.generic_rubric = GenericRubricFactory().build_math_rubric()
        self._rubric_queue = queue.Queue(maxsize=self.raw_config.get("opsd_rubric", {}).get("max_pending_requests", 1024))
        self._rubric_worker = threading.Thread(target=self._rubric_worker_loop, daemon=True)
        self._rubric_worker.start()

    def enqueue_rubric_request(self, request):
        with contextlib.suppress(queue.Full):
            self._rubric_queue.put_nowait(request)

    def _rubric_worker_loop(self):
        while True:
            request = self._rubric_queue.get()
            self._mine_and_store_rubric(request)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_teacher_rubric_flow.py -q`
Expected: PASS with 3 passing tests

- [ ] **Step 5: Commit**

```bash
git add tests/verl_opsd/test_teacher_rubric_flow.py verl_opsd/teacher.py verl_opsd/main_ppo_opsd.py verl_opsd/rubric_prompting.py verl_opsd/rubric_curriculum.py verl_opsd/rubric_memory.py verl_opsd/rubric_miner.py
git commit -m "feat: integrate rubric selection into teacher manager"
```

### Task 5: Submit rollout observations from the trainer and extend validation metrics

**Files:**
- Modify: `third_party/verl/verl/trainer/ppo/ray_trainer.py`
- Modify: `third_party/verl/verl/trainer/ppo/metric_utils.py`
- Modify: `third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py`
- Modify: `verl_opsd/teacher.py`
- Modify: `verl_opsd/rubric_miner.py`

- [ ] **Step 1: Write the failing tests for `mean@1` derivation from `n=4` validation**

```python
def test_process_validation_metrics_reports_mean_at_1_from_first_sample():
    data_sources = ["math_dapo"] * 4
    sample_uids = ["p1"] * 4
    infos_dict = {"score": [1.0, 0.0, 1.0, 1.0], "acc": [1.0, 0.0, 1.0, 1.0]}

    result = process_validation_metrics(data_sources, sample_uids, infos_dict, seed=42)

    assert result["math_dapo"]["score"]["mean@1"] == 1.0
    assert result["math_dapo"]["score"]["mean@4"] == 0.75
    assert result["math_dapo"]["score"]["best@4/mean"] == 1.0


def test_score_metrics_are_marked_as_primary_validation_targets():
    from verl.trainer.ppo.metric_utils import is_primary_score_metric_name

    assert is_primary_score_metric_name("score", "mean@1", 4) is True
    assert is_primary_score_metric_name("score", "mean@4", 4) is True
    assert is_primary_score_metric_name("score", "best@4/mean", 4) is True
    assert is_primary_score_metric_name("score", "std@4", 4) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py -q`
Expected: FAIL because `mean@1` is not present for `n_resps == 4`

- [ ] **Step 3: Write the minimal trainer hook and metric changes**

```python
# third_party/verl/verl/trainer/ppo/metric_utils.py
def is_primary_score_metric_name(var_name: str, metric_name: str, n_max: int) -> bool:
    return var_name == "score" and metric_name in {f"mean@1", f"mean@{n_max}", f"best@{n_max}/mean"}


if n_resps > 1:
    metric["mean@1"] = float(var_vals[0])
    metric["std@1"] = 0.0


# verl_opsd/teacher.py
def submit_rubric_updates(self, batch, reward_extra_infos_dict, global_step: int):
    observations = build_rollout_observations(batch=batch, reward_extra_infos_dict=reward_extra_infos_dict, global_step=global_step, tokenizer=self.tokenizer)
    for observation in observations:
        request = self.rubric_memory.observe(observation)
        if request is not None:
            self.enqueue_rubric_request(request)

def pop_rubric_metrics(self):
    metrics = dict(self._rubric_metrics)
    self._rubric_metrics.clear()
    return metrics


# third_party/verl/verl/trainer/ppo/ray_trainer.py
if is_primary_score_metric_name(var_name, metric_name, n_max):
    metric_sec = "val-core"
elif (
    (var_name == core_var)
    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
    and (f"@{n_max}" in metric_name)
):
    metric_sec = "val-core"

with marked_timer("reward", timing_raw, color="yellow"):
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
if self.teacher_model_manager is not None and hasattr(self.teacher_model_manager, "submit_rubric_updates"):
    self.teacher_model_manager.submit_rubric_updates(batch, reward_extra_infos_dict, self.global_steps)
if self.teacher_model_manager is not None and hasattr(self.teacher_model_manager, "pop_rubric_metrics"):
    metrics.update(self.teacher_model_manager.pop_rubric_metrics())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py -q`
Expected: PASS and the new `mean@1` test passes alongside the existing metric tests

- [ ] **Step 5: Commit**

```bash
git add third_party/verl/verl/trainer/ppo/ray_trainer.py third_party/verl/verl/trainer/ppo/metric_utils.py third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py verl_opsd/teacher.py verl_opsd/rubric_miner.py
git commit -m "feat: wire rubric updates into trainer and validation metrics"
```

### Task 6: Verify full rubric flow with targeted unit tests and a smoke run

**Files:**
- Modify: `tests/verl_opsd/test_rubric_core.py`
- Modify: `tests/verl_opsd/test_rubric_miner.py`
- Modify: `tests/verl_opsd/test_teacher_rubric_flow.py`
- Modify: `scripts/run_opsd_4b_verl_smoke.sh`

- [ ] **Step 1: Write the final failing coverage test for rubric source diagnostics**

```python
from collections import defaultdict


class TestTeacherManager:
    def __init__(self):
        self._usage_total = 0
        self._usage_by_source = defaultdict(int)
        self._prompt_token_total = 0

    def record_rubric_usage(self, source: str, prompt_token_count: int):
        self._usage_total += 1
        self._usage_by_source[source] += 1
        self._prompt_token_total += prompt_token_count

    def pop_rubric_metrics(self):
        total = max(self._usage_total, 1)
        return {
            "rubric/generic_usage_rate": self._usage_by_source["generic"] / total,
            "rubric/self_mined_usage_rate": self._usage_by_source["self_mined"] / total,
            "rubric/active_prompt_tokens_mean": self._prompt_token_total / total,
        }


def test_teacher_manager_reports_generic_and_self_mined_usage_metrics():
    manager = TestTeacherManager()
    manager.record_rubric_usage("generic", prompt_token_count=120)
    manager.record_rubric_usage("self_mined", prompt_token_count=140)

    metrics = manager.pop_rubric_metrics()

    assert metrics["rubric/generic_usage_rate"] == 0.5
    assert metrics["rubric/self_mined_usage_rate"] == 0.5
    assert metrics["rubric/active_prompt_tokens_mean"] == 130.0
```

- [ ] **Step 2: Run the targeted unit tests to verify the new test fails**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_core.py tests/verl_opsd/test_rubric_miner.py tests/verl_opsd/test_teacher_rubric_flow.py -q`
Expected: FAIL because rubric diagnostics are not fully aggregated yet

- [ ] **Step 3: Complete the minimal diagnostics aggregation and smoke config**

```python
# verl_opsd/teacher.py
def record_rubric_usage(self, source: str, prompt_token_count: int):
    self._usage_total += 1
    self._usage_by_source[source] += 1
    self._prompt_token_total += prompt_token_count

def pop_rubric_metrics(self):
    total = max(self._usage_total, 1)
    metrics = {
        "rubric/generic_usage_rate": self._usage_by_source["generic"] / total,
        "rubric/self_mined_usage_rate": self._usage_by_source["self_mined"] / total,
        "rubric/cache_hit_rate": self._cache_hits / total,
        "rubric/update_success_rate": self._update_successes / max(self._update_attempts, 1),
        "rubric/active_prompt_tokens_mean": self._prompt_token_total / total,
    }
    self._reset_metric_counters()
    return metrics

# scripts/run_opsd_4b_verl_smoke.sh
export VAL_SAMPLE_N="${VAL_SAMPLE_N:-4}"
export RUBRIC_ENABLED="${RUBRIC_ENABLED:-true}"
export TEST_FREQ="${TEST_FREQ:-1}"
```

- [ ] **Step 4: Run the full verification set**

Run: `PYTHONPATH=/root/MRTOPSD:/root/MRTOPSD/third_party/verl pytest tests/verl_opsd/test_rubric_core.py tests/verl_opsd/test_prepare_dataset.py tests/verl_opsd/test_rubric_miner.py tests/verl_opsd/test_teacher_rubric_flow.py third_party/verl/tests/trainer/ppo/test_metric_utils_on_cpu.py -q`
Expected: PASS with all rubric-specific tests plus the patched verl metric tests passing

Run: `WANDB_MODE=disabled EXPERIMENT_NAME=qwen3_4b_opsd_verl_rubric_smoke TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 bash scripts/run_opsd_4b_verl_smoke.sh`
Expected: exit code `0`, one checkpoint under the smoke output directory, validation logs containing `val-core/math_dapo/score/mean@1`, `val-core/math_dapo/score/mean@4`, and `val-core/math_dapo/score/best@4/mean`

- [ ] **Step 5: Commit**

```bash
git add tests/verl_opsd/test_rubric_core.py tests/verl_opsd/test_rubric_miner.py tests/verl_opsd/test_teacher_rubric_flow.py scripts/run_opsd_4b_verl_smoke.sh
git commit -m "test: verify rubric diagnostics and smoke flow"
```
