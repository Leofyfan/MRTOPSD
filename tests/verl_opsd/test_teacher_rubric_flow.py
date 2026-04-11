import queue
import threading
from types import SimpleNamespace
from threading import RLock

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.workers.config import DistillationConfig
from verl_opsd.rubric_curriculum import RubricCurriculum
from verl_opsd.rubric_memory import RubricEntry, RubricMiningRequest, RolloutObservation
from verl_opsd.rubric_prompting import RubricPayload
from verl_opsd.teacher import AsyncOPSDTeacherLLMServerManager, OPSDTeacherModelManager


class RecordingTokenizer:
    def __init__(self):
        self.apply_chat_template_calls = []
        self.tokenizer_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.apply_chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        rendered = f"CHAT::{messages[0]['content']}"
        if tokenize:
            return [len(rendered), len(rendered) + 1]
        return rendered

    def __call__(self, text, add_special_tokens, return_attention_mask):
        self.tokenizer_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "return_attention_mask": return_attention_mask,
            }
        )
        return {"input_ids": [len(text), len(text) + 1]}


class RubricMiningTokenizer:
    def __init__(self):
        self.apply_chat_template_calls = []
        self.tokenizer_calls = []
        self.decode_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.apply_chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return f"CHAT::{messages[0]['content']}"

    def __call__(self, text, add_special_tokens, return_attention_mask):
        self.tokenizer_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "return_attention_mask": return_attention_mask,
            }
        )
        return {"input_ids": [101, 202, 303]}

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
            }
        )
        return (
            '{"core_correctness_rule":"match the verified answer",'
            '"core_key_steps_rule":"keep the essential derivation",'
            '"core_error_avoidance_rule":"avoid sign errors",'
            '"free_rule":"check the final answer"}'
        )


class FakeRubricServerManager:
    def __init__(self, output_token_ids):
        self.output_token_ids = output_token_ids
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": sampling_params,
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        return SimpleNamespace(token_ids=self.output_token_ids, extra_fields={})


def build_server_manager(
    tokenizer,
    *,
    rubric_enabled: bool,
    active_entry: RubricEntry | None = None,
    curriculum: RubricCurriculum | None = None,
) -> AsyncOPSDTeacherLLMServerManager:
    return AsyncOPSDTeacherLLMServerManager(
        config={},
        servers=[],
        load_balancer_handle=None,
        distillation_config=DistillationConfig(),
        pad_token_id=0,
        tokenizer=tokenizer,
        active_rubric_lookup=(lambda problem_id: active_entry),
        generic_rubric=RubricPayload("generic correctness", "generic steps", "generic avoid"),
        use_rubric_prompts=rubric_enabled,
        rubric_curriculum=curriculum or RubricCurriculum(warmup_steps=0, mix_steps=0, seed=0),
    )


def test_flag_off_keeps_legacy_teacher_prompt_text_path():
    tokenizer = RecordingTokenizer()
    manager = build_server_manager(tokenizer, rubric_enabled=False)

    prompt_ids = manager._teacher_prompt_ids_from_context(
        extra_info={"teacher_prompt_text": "legacy prompt"},
        problem_id="p1",
        problem="Solve x+1=2.",
        global_step=3,
    )

    assert prompt_ids == [13, 14]
    assert tokenizer.apply_chat_template_calls == []
    assert tokenizer.tokenizer_calls[-1]["text"] == "legacy prompt"


def test_teacher_prompt_ids_take_precedence_over_all_other_prompt_sources():
    tokenizer = RecordingTokenizer()
    entry = RubricEntry(
        problem_id="p1",
        rubric_source="self_mined",
        rubric_version=1,
        updated_step=10,
        course_stage="mature",
        rubric_payload=RubricPayload("self correctness", "self steps", "self avoid"),
        correct_example_summary="correct example",
        wrong_example_summary="wrong example",
    )
    manager = build_server_manager(tokenizer, rubric_enabled=True, active_entry=entry)

    prompt_ids = manager._teacher_prompt_ids_from_context(
        extra_info={"teacher_prompt_ids": [7, 8, 9], "teacher_prompt_text": "legacy prompt"},
        problem_id="p1",
        problem="Solve x+1=2.",
        global_step=20,
    )

    assert prompt_ids == [7, 8, 9]
    assert tokenizer.apply_chat_template_calls == []
    assert tokenizer.tokenizer_calls == []


def test_curriculum_gating_falls_back_to_generic_until_self_mined_is_allowed():
    entry = RubricEntry(
        problem_id="p1",
        rubric_source="self_mined",
        rubric_version=1,
        updated_step=10,
        course_stage="mature",
        rubric_payload=RubricPayload("self correctness", "self steps", "self avoid"),
        correct_example_summary="correct example",
        wrong_example_summary="wrong example",
    )
    curriculum = RubricCurriculum(warmup_steps=5, mix_steps=0, seed=0)

    early_tokenizer = RecordingTokenizer()
    early_manager = build_server_manager(
        early_tokenizer,
        rubric_enabled=True,
        active_entry=entry,
        curriculum=curriculum,
    )
    early_manager._teacher_prompt_ids_from_context(
        extra_info={},
        problem_id="p1",
        problem="Solve x+1=2.",
        global_step=4,
    )

    late_tokenizer = RecordingTokenizer()
    late_manager = build_server_manager(
        late_tokenizer,
        rubric_enabled=True,
        active_entry=entry,
        curriculum=curriculum,
    )
    late_manager._teacher_prompt_ids_from_context(
        extra_info={},
        problem_id="p1",
        problem="Solve x+1=2.",
        global_step=5,
    )

    assert "generic steps" in early_tokenizer.tokenizer_calls[-1]["text"]
    assert "self steps" not in early_tokenizer.tokenizer_calls[-1]["text"]
    assert "self steps" in late_tokenizer.tokenizer_calls[-1]["text"]
    assert "correct example" not in late_tokenizer.tokenizer_calls[-1]["text"]
    assert "wrong example" not in late_tokenizer.tokenizer_calls[-1]["text"]


def test_rubric_prompts_are_rendered_through_chat_template_before_tokenization():
    tokenizer = RecordingTokenizer()
    manager = build_server_manager(tokenizer, rubric_enabled=True)

    manager._teacher_prompt_ids_from_context(
        extra_info={},
        problem_id="p1",
        problem="Solve x+1=2.",
        global_step=2,
    )

    assert len(tokenizer.apply_chat_template_calls) == 1
    template_call = tokenizer.apply_chat_template_calls[0]
    assert template_call["add_generation_prompt"] is True
    assert template_call["enable_thinking"] is True
    assert template_call["tokenize"] is False
    assert tokenizer.tokenizer_calls[-1]["text"].startswith("CHAT::")
    assert "Solve x+1=2." in tokenizer.tokenizer_calls[-1]["text"]
    assert "token-by-token" in tokenizer.tokenizer_calls[-1]["text"]
    assert "boxed" not in tokenizer.tokenizer_calls[-1]["text"]
    assert "solve the problem" not in tokenizer.tokenizer_calls[-1]["text"].lower()


def test_rubric_worker_loop_records_failures_and_keeps_processing():
    manager = OPSDTeacherModelManager.__new__(OPSDTeacherModelManager)
    manager.pending_rubric_requests = queue.Queue()
    manager._rubric_worker_stop = threading.Event()
    manager.rubric_mining_failures = 0
    manager.rubric_mining_last_error = None
    processed = []

    def fake_mine(request):
        if request == "bad":
            raise RuntimeError("boom")
        processed.append(request)

    manager._mine_and_store_rubric = fake_mine
    manager.pending_rubric_requests.put("bad")
    manager.pending_rubric_requests.put("good")
    manager.pending_rubric_requests.put(None)

    worker = threading.Thread(target=manager._rubric_worker_loop, daemon=True)
    worker.start()
    worker.join(timeout=2.0)

    assert processed == ["good"]
    assert manager.rubric_mining_failures == 1
    assert "boom" in manager.rubric_mining_last_error


def test_mine_and_store_rubric_uses_teacher_backed_json_mining_by_default(monkeypatch):
    manager = OPSDTeacherModelManager.__new__(OPSDTeacherModelManager)
    manager._rubric_miner_fn = None
    manager.rubric_memory = type(
        "RecordingMemory",
        (),
        {
            "__init__": lambda self: setattr(self, "stored", []),
            "get_active_entry": lambda self, problem_id: None,
            "store_active_entry": lambda self, entry: self.stored.append(entry),
        },
    )()
    manager.rubric_curriculum = RubricCurriculum(warmup_steps=0, mix_steps=0, seed=0)
    manager.server_manager = FakeRubricServerManager(output_token_ids=[9001, 9002])
    manager.tokenizer = RubricMiningTokenizer()
    manager._teacher_service_lock = RLock()
    lifecycle_calls = []
    manager.wake_up = lambda: lifecycle_calls.append("wake")
    manager.sleep = lambda: lifecycle_calls.append("sleep")

    def fail_if_heuristic_used(request):
        raise AssertionError("heuristic fallback should not be used when server_manager is available")

    monkeypatch.setattr("verl_opsd.teacher.build_rubric_payload_from_request", fail_if_heuristic_used)

    request = RubricMiningRequest(
        problem_id="p1",
        problem="Solve x + 1 = 2.",
        ground_truth="1",
        correct_observation=RolloutObservation(
            problem_id="p1",
            problem="Solve x + 1 = 2.",
            ground_truth="1",
            response_text="Careful derivation ending at 1",
            score=1.0,
            acc=1.0,
            global_step=3,
        ),
        wrong_observation=RolloutObservation(
            problem_id="p1",
            problem="Solve x + 1 = 2.",
            ground_truth="1",
            response_text="Missed the minus one adjustment",
            score=-0.3,
            acc=0.0,
            global_step=2,
        ),
    )

    entry = manager._mine_and_store_rubric(request)

    assert len(manager.server_manager.calls) == 1
    assert lifecycle_calls == ["wake", "sleep"]
    assert manager.server_manager.calls[0]["prompt_ids"] == [101, 202, 303]
    assert manager.tokenizer.apply_chat_template_calls[0]["messages"][0]["content"].startswith("Read the paired student attempts")
    assert manager.tokenizer.decode_calls[0]["token_ids"] == [9001, 9002]
    assert entry.rubric_payload.core_correctness_rule == "match the verified answer"
    assert entry.rubric_payload.core_key_steps_rule == "keep the essential derivation"
    assert entry.rubric_payload.core_error_avoidance_rule == "avoid sign errors"
    assert entry.rubric_payload.free_rule == "check the final answer"
    assert manager.rubric_memory.stored == [entry]


def test_enqueue_rubric_request_drops_when_bounded_queue_is_full():
    manager = OPSDTeacherModelManager.__new__(OPSDTeacherModelManager)
    manager.opsd_rubric_enabled = True
    manager.pending_rubric_requests = queue.Queue(maxsize=1)
    manager.dropped_rubric_requests = 0

    manager.enqueue_rubric_request("first")
    manager.enqueue_rubric_request("second")

    assert manager.pending_rubric_request_count() == 1
    assert manager.pending_rubric_requests.get_nowait() == "first"
    assert manager.dropped_rubric_requests == 1


def test_submit_rubric_updates_prefers_hardest_wrong_from_current_rollout(monkeypatch):
    manager = OPSDTeacherModelManager.__new__(OPSDTeacherModelManager)
    manager.opsd_rubric_enabled = True
    manager.tokenizer = object()

    weak_wrong = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="weak wrong attempt",
        score=-0.9,
        acc=0.0,
        global_step=6,
    )
    hard_wrong = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="hard wrong attempt",
        score=-0.1,
        acc=0.0,
        global_step=7,
    )
    correct_obs = RolloutObservation(
        problem_id="p1",
        problem="Q",
        ground_truth="42",
        response_text="correct attempt",
        score=1.0,
        acc=1.0,
        global_step=8,
    )
    observations = [weak_wrong, correct_obs, hard_wrong]
    enqueued = []

    class RecordingMemory:
        def __init__(self):
            self.seen = []

        def observe(self, observation):
            self.seen.append(observation)
            return "request-1" if observation.response_text == "hard wrong attempt" else None

    manager.rubric_memory = RecordingMemory()
    manager.enqueue_rubric_request = lambda request: enqueued.append(request)

    monkeypatch.setattr(
        "verl_opsd.teacher.build_rollout_observations",
        lambda batch, reward_extra_infos_dict, global_step, tokenizer: observations,
    )

    manager.submit_rubric_updates(
        batch="batch",
        reward_extra_infos_dict={"score": [1.0, -1.0], "acc": [1.0, 0.0]},
        global_step=7,
    )

    assert hard_wrong in manager.rubric_memory.seen
    assert correct_obs in manager.rubric_memory.seen
    assert weak_wrong not in manager.rubric_memory.seen
    assert enqueued == ["request-1"]


def test_compute_logprobs_uses_shared_teacher_service_lock(monkeypatch):
    manager = OPSDTeacherModelManager.__new__(OPSDTeacherModelManager)
    events = []

    class RecordingLock:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    manager._teacher_service_lock = RecordingLock()

    def fake_super_compute_logprobs(self, data):
        events.append(("compute", data))
        return "ok"

    monkeypatch.setattr("verl_opsd.teacher.TeacherModelManager.compute_logprobs", fake_super_compute_logprobs)

    result = OPSDTeacherModelManager.compute_logprobs(manager, data="batch-1")

    assert result == "ok"
    assert events == ["enter", ("compute", "batch-1"), "exit"]


def test_agent_loop_streaming_teacher_holds_shared_lock_across_wake_generate_sleep(monkeypatch):
    events = []

    class RecordingLock:
        def __init__(self):
            self.active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            assert self.active
            events.append("exit")
            self.active = False

    lock = RecordingLock()

    class RecordingTeacherManager:
        def __init__(self):
            self._teacher_service_lock = lock

        async def wake_up(self):
            assert lock.active
            events.append("wake")

        async def sleep(self):
            assert lock.active
            events.append("sleep")

    async def remote_generate(chunk):
        assert lock.active
        events.append(("worker", chunk))
        return SimpleNamespace(meta_info={"metrics": [{"generate_sequences": 1.0, "tool_calls": 0.0, "num_preempted": 0}]})

    combined_output = SimpleNamespace(meta_info={})

    class FakeDataProto:
        @staticmethod
        def concat(outputs):
            events.append(("concat", len(outputs)))
            return combined_output

    class FakePrompts:
        def chunk(self, size):
            events.append(("chunk", size))
            return ["chunk-1"]

    dummy_manager = SimpleNamespace(
        stream_teacher_with_rollout=True,
        teacher_model_manager=RecordingTeacherManager(),
        agent_loop_workers=[SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote_generate))],
        _performance_metrics=lambda metrics, output: {"agent_loop/generate_sequences/max": 1.0},
    )

    monkeypatch.setattr("verl.experimental.agent_loop.agent_loop.DataProto", FakeDataProto)

    result = AgentLoopManager.generate_sequences(dummy_manager, FakePrompts())

    assert result is combined_output
    assert combined_output.meta_info == {"timing": {"agent_loop/generate_sequences/max": 1.0}}
    assert events == [
        "enter",
        "wake",
        ("chunk", 1),
        ("worker", "chunk-1"),
        "sleep",
        "exit",
        ("concat", 1),
    ]
