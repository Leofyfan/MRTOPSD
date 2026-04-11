from __future__ import annotations

import asyncio
import queue
import threading
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.experimental.teacher_loop.teacher_model import TeacherModelManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl_opsd.rubric_curriculum import RubricCurriculum
from verl_opsd.rubric_memory import RubricEntry, RubricMemory, RubricMiningRequest
from verl_opsd.rubric_miner import (
    build_rollout_observations,
    build_rubric_payload_from_request,
    parse_rubric_response,
    select_hard_wrong_observation,
    summarize_response_text,
)
from verl_opsd.rubric_prompting import (
    GenericRubricFactory,
    RubricPayload,
    build_teacher_scoring_messages,
    format_teacher_scoring_prompt,
    format_rubric_mining_prompt,
)


def _teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict:
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM prompt_logprobs requires temperature=1.0 for teacher scoring.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _valid_response_ids(item: DataProto) -> list[int]:
    responses = item.batch["responses"][0]
    response_mask = item.batch["response_mask"][0].bool()
    return normalize_token_ids(responses[response_mask])


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def select_active_teacher_prompt(
    problem_id: str,
    problem: str,
    active_entry: RubricEntry | None,
    generic_rubric: RubricPayload,
) -> tuple[str, str]:
    del problem_id
    if active_entry is None:
        return format_teacher_scoring_prompt(problem=problem, rubric=generic_rubric), "generic"

    return format_teacher_scoring_prompt(problem=problem, rubric=active_entry.rubric_payload), active_entry.rubric_source


def _pad_teacher_response_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_width = prompt_width + response_width

    teacher_ids = teacher_ids[-response_length:] if response_length > 0 else teacher_ids[:0]
    teacher_logprobs = teacher_logprobs[-response_length:] if response_length > 0 else teacher_logprobs[:0]

    ids_shape = (total_width,) + tuple(teacher_ids.shape[1:])
    logprob_shape = (total_width,) + tuple(teacher_logprobs.shape[1:])

    padded_ids = torch.full(ids_shape, pad_token_id, dtype=teacher_ids.dtype if teacher_ids.numel() else torch.int32)
    padded_logprobs = torch.zeros(
        logprob_shape,
        dtype=teacher_logprobs.dtype if teacher_logprobs.numel() else torch.float32,
    )

    if response_length > 0:
        start = prompt_width
        end = prompt_width + response_length
        padded_ids[start:end] = teacher_ids
        padded_logprobs[start:end] = teacher_logprobs

    return padded_ids.unsqueeze(0), padded_logprobs.unsqueeze(0)


class AsyncOPSDTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher client that scores student responses under privileged OPSD prompts."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
        tokenizer,
        active_rubric_lookup=None,
        generic_rubric: RubricPayload | None = None,
        use_rubric_prompts: bool = False,
        rubric_curriculum: RubricCurriculum | None = None,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.active_rubric_lookup = active_rubric_lookup
        self.generic_rubric = generic_rubric or GenericRubricFactory().build_math_rubric()
        self.use_rubric_prompts = use_rubric_prompts
        self.rubric_curriculum = rubric_curriculum or RubricCurriculum(warmup_steps=0, mix_steps=0, seed=0)

    def _render_chat_template_prompt(self, prompt_text: str) -> str:
        messages = build_teacher_scoring_messages(prompt_text)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _teacher_prompt_ids_from_context(
        self,
        extra_info: dict | None,
        problem_id: str,
        problem: str,
        global_step: int,
    ) -> list[int]:
        extra_info = extra_info or {}

        teacher_prompt_ids = extra_info.get("teacher_prompt_ids")
        if teacher_prompt_ids is not None:
            return normalize_token_ids(teacher_prompt_ids)

        teacher_prompt_text = extra_info.get("teacher_prompt_text")
        if self.use_rubric_prompts and problem:
            active_entry = None
            if (
                problem_id
                and self.active_rubric_lookup is not None
                and self.rubric_curriculum.should_use_self_mined(problem_id, global_step)
            ):
                active_entry = self.active_rubric_lookup(problem_id)
            rubric_prompt_text, _ = select_active_teacher_prompt(
                problem_id=problem_id,
                problem=problem,
                active_entry=active_entry,
                generic_rubric=self.generic_rubric,
            )
            teacher_prompt_text = self._render_chat_template_prompt(rubric_prompt_text)

        if teacher_prompt_text is None:
            raise KeyError(
                "Missing problem text or `extra_info.teacher_prompt_text` for OPSD teacher prompt construction."
            )

        tokenized = self.tokenizer(
            teacher_prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return normalize_token_ids(tokenized)

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multi_modal_data = multi_modal_data or {}
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        extra_info_batch = data.non_tensor_batch.get("extra_info")
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        problem_id_batch = data.non_tensor_batch.get("problem_id")
        problem_batch = data.non_tensor_batch.get("problem")
        global_step = int(data.meta_info.get("global_steps", 0))
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        tasks = []
        response_lengths = []
        for i in range(len(data)):
            item = data[i : i + 1]
            response_ids = _valid_response_ids(item)
            extra_info = None if extra_info_batch is None else extra_info_batch[i]
            problem_id = _coerce_text(None if problem_id_batch is None else problem_id_batch[i])
            if not problem_id:
                problem_id = _coerce_text((extra_info or {}).get("problem_id"))
            problem = _coerce_text(None if problem_batch is None else problem_batch[i])
            if not problem:
                problem = _coerce_text((extra_info or {}).get("problem"))
            teacher_prompt_ids = self._teacher_prompt_ids_from_context(extra_info, problem_id, problem, global_step)
            sequence_ids = teacher_prompt_ids + response_ids
            response_lengths.append(len(response_ids))
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        multi_modal_data=multi_modal_data,
                    )
                )
            )

        outputs = await asyncio.gather(*tasks)

        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), response_length in zip(outputs, response_lengths, strict=True):
            padded_ids, padded_logprobs = _pad_teacher_response_outputs(
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)


class OPSDTeacherModelManager(TeacherModelManager):
    """Teacher model manager that scores student rollouts under privileged OPSD prompts."""

    def __init__(self, config: DictConfig, resource_pool=None):
        self.raw_config = config
        self._initialize_rubric_state(config)
        super().__init__(config=config, resource_pool=resource_pool)
        self._start_rubric_worker()

    def _initialize_rubric_state(self, raw_config: DictConfig | dict | DistillationConfig) -> None:
        rubric_config = {}
        if hasattr(raw_config, "get"):
            rubric_config = raw_config.get("opsd_rubric", {}) or {}
        elif hasattr(raw_config, "opsd_rubric"):
            rubric_config = getattr(raw_config, "opsd_rubric") or {}

        self.opsd_rubric_enabled = bool(getattr(rubric_config, "get", lambda *_: None)("enabled", False))
        get_value = rubric_config.get if hasattr(rubric_config, "get") else lambda key, default=None: default
        self.rubric_memory = RubricMemory(min_response_chars=int(get_value("min_response_chars", 32)))
        self.rubric_curriculum = RubricCurriculum(
            warmup_steps=int(get_value("warmup_steps", 0)),
            mix_steps=int(get_value("mix_steps", 0)),
            seed=int(get_value("seed", 0)),
        )
        self.max_pending_rubric_requests = int(get_value("max_pending_requests", 128))
        self.generic_rubric_factory = GenericRubricFactory()
        self.generic_rubric = self.generic_rubric_factory.build_math_rubric()
        self.pending_rubric_requests: queue.Queue[RubricMiningRequest | None] = queue.Queue(
            maxsize=self.max_pending_rubric_requests
        )
        self._rubric_worker_stop = threading.Event()
        self._rubric_worker_thread: threading.Thread | None = None
        self._rubric_miner_fn = None
        self.rubric_mining_failures = 0
        self.rubric_mining_last_error: str | None = None
        self.dropped_rubric_requests = 0
        self._teacher_service_lock = threading.RLock()

    def submit_rubric_updates(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict[str, list],
        global_step: int,
    ) -> None:
        if not self.opsd_rubric_enabled:
            return

        observations = build_rollout_observations(
            batch=batch,
            reward_extra_infos_dict=reward_extra_infos_dict,
            global_step=global_step,
            tokenizer=self.tokenizer,
        )
        selected_hard_wrong_ids = self._selected_hard_wrong_ids_by_problem(observations)
        for observation in observations:
            if observation.acc <= 0.0 and id(observation) != selected_hard_wrong_ids.get(observation.problem_id):
                continue
            request = self.rubric_memory.observe(observation)
            if request is not None:
                self.enqueue_rubric_request(request)

    def _selected_hard_wrong_ids_by_problem(self, observations):
        grouped: dict[str, list] = {}
        for observation in observations:
            grouped.setdefault(observation.problem_id, []).append(observation)

        selected = {}
        min_response_chars = getattr(self.rubric_memory, "min_response_chars", 0)
        for problem_id, problem_observations in grouped.items():
            hard_wrong = select_hard_wrong_observation(problem_observations, min_response_chars=min_response_chars)
            if hard_wrong is not None:
                selected[problem_id] = id(hard_wrong)
        return selected

    def _start_rubric_worker(self) -> None:
        if not self.opsd_rubric_enabled or self._rubric_worker_thread is not None:
            return
        self._rubric_worker_thread = threading.Thread(
            target=self._rubric_worker_loop,
            name="opsd-rubric-worker",
            daemon=True,
        )
        self._rubric_worker_thread.start()

    def compute_logprobs(self, data):
        with self._teacher_service_lock:
            return super().compute_logprobs(data)

    def _initialize_async_server_manager(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(server_actor_ids=self.server_addresses)
        self.server_manager = AsyncOPSDTeacherLLMServerManager(
            config=self.config,
            servers=list(zip(self.server_addresses, self.server_handles, strict=True)),
            load_balancer_handle=self.load_balancer_handle,
            distillation_config=self.config,
            pad_token_id=self.pad_token_id,
            tokenizer=self.tokenizer,
            active_rubric_lookup=self.rubric_memory.get_active_entry,
            generic_rubric=self.generic_rubric,
            use_rubric_prompts=self.opsd_rubric_enabled,
            rubric_curriculum=self.rubric_curriculum,
        )

    def enqueue_rubric_request(self, request: RubricMiningRequest) -> None:
        if not self.opsd_rubric_enabled:
            return
        try:
            self.pending_rubric_requests.put_nowait(request)
        except queue.Full:
            self.dropped_rubric_requests += 1

    def pending_rubric_request_count(self) -> int:
        return self.pending_rubric_requests.qsize()

    def _rubric_worker_loop(self) -> None:
        while not self._rubric_worker_stop.is_set():
            try:
                request = self.pending_rubric_requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                self.pending_rubric_requests.task_done()
                break
            try:
                self._mine_and_store_rubric(request)
            except Exception as exc:
                self.rubric_mining_failures += 1
                self.rubric_mining_last_error = str(exc)
            finally:
                self.pending_rubric_requests.task_done()

    def _mine_and_store_rubric(self, request: RubricMiningRequest) -> RubricEntry:
        rubric_payload = self._mine_rubric_payload(request)
        existing_entry = self.rubric_memory.get_active_entry(request.problem_id)
        updated_step = max(request.correct_observation.global_step, request.wrong_observation.global_step)
        entry = RubricEntry(
            problem_id=request.problem_id,
            rubric_source="self_mined",
            rubric_version=1 if existing_entry is None else existing_entry.rubric_version + 1,
            updated_step=updated_step,
            course_stage=self.rubric_curriculum.course_stage(updated_step),
            rubric_payload=rubric_payload,
            correct_example_summary=summarize_response_text(request.correct_observation.response_text),
            wrong_example_summary=summarize_response_text(request.wrong_observation.response_text),
        )
        self.rubric_memory.store_active_entry(entry)
        return entry

    def _mine_rubric_payload(self, request: RubricMiningRequest):
        if self._rubric_miner_fn is not None:
            return self._rubric_miner_fn(request)

        if getattr(self, "server_manager", None) is None or not hasattr(self.tokenizer, "decode"):
            return build_rubric_payload_from_request(request)

        with self._teacher_service_lock:
            self.wake_up()
            try:
                return asyncio.run(self._mine_rubric_payload_with_teacher_service(request))
            finally:
                self.sleep()

    def _render_rubric_mining_prompt(self, request: RubricMiningRequest) -> str:
        mining_prompt = format_rubric_mining_prompt(
            problem=request.problem,
            ground_truth=request.ground_truth,
            correct_example_summary=summarize_response_text(request.correct_observation.response_text),
            wrong_example_summary=summarize_response_text(request.wrong_observation.response_text),
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": mining_prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return mining_prompt

    def _rubric_mining_prompt_ids(self, request: RubricMiningRequest) -> list[int]:
        prompt_text = self._render_rubric_mining_prompt(request)
        tokenized = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return normalize_token_ids(tokenized)

    async def _mine_rubric_payload_with_teacher_service(self, request: RubricMiningRequest):
        prompt_ids = self._rubric_mining_prompt_ids(request)
        teacher_output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 256, "temperature": 0.0},
        )
        response_token_ids = normalize_token_ids(teacher_output.token_ids)
        response_text = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        return parse_rubric_response(response_text)

    def shutdown_rubric_worker(self) -> None:
        if self._rubric_worker_thread is None:
            return
        self._rubric_worker_stop.set()
        self.pending_rubric_requests.put(None)
        self._rubric_worker_thread.join(timeout=1.0)
