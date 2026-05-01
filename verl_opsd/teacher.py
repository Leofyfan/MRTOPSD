from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
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
    build_dynamic_rubric_payload_from_response,
    build_dynamic_rubric_payload_from_request,
    build_rollout_observations,
    build_rubric_payload_from_request,
    parse_dynamic_rubric_response,
    parse_rubric_response,
    select_hard_wrong_observation,
    summarize_response_text,
)
from verl_opsd.rubric_prompting import (
    DynamicRubricPayload,
    GenericRubricFactory,
    RubricPayload,
    build_teacher_scoring_messages,
    format_dynamic_rubric_mining_prompt,
    format_dynamic_teacher_scoring_prompt,
    format_teacher_scoring_prompt,
    format_rubric_mining_prompt,
    select_dynamic_rubric_specs,
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


def _ground_truth_from_context(reward_model_entry, extra_info: dict | None) -> str:
    if isinstance(reward_model_entry, dict):
        ground_truth = _coerce_text(reward_model_entry.get("ground_truth", ""))
    else:
        ground_truth = _coerce_text(reward_model_entry)
    if ground_truth:
        return ground_truth

    extra_reward_model = (extra_info or {}).get("reward_model")
    if isinstance(extra_reward_model, dict):
        return _coerce_text(extra_reward_model.get("ground_truth", ""))
    return _coerce_text(extra_reward_model)


def select_active_teacher_prompt(
    problem_id: str,
    problem: str,
    active_entry: RubricEntry | None,
    generic_rubric: RubricPayload,
) -> tuple[str, str]:
    del problem_id
    if active_entry is None:
        return format_teacher_scoring_prompt(problem=problem, rubric=generic_rubric), "generic"

    rubric_payload = active_entry.rubric_payload
    if isinstance(rubric_payload, DynamicRubricPayload):
        rubric_payload = _dynamic_payload_to_single_rubric(rubric_payload)

    return format_teacher_scoring_prompt(problem=problem, rubric=rubric_payload), active_entry.rubric_source


def _dynamic_payload_to_single_rubric(payload: DynamicRubricPayload) -> RubricPayload:
    positive = [rubric.criterion for rubric in payload.rubrics if rubric.points > 0]
    negative = [rubric.criterion for rubric in payload.rubrics if rubric.points < 0]
    return RubricPayload(
        core_correctness_rule=positive[0] if positive else "Satisfy the strongest applicable rubric criteria.",
        core_key_steps_rule=(
            "; ".join(positive[1:3])
            if len(positive) > 1
            else "Preserve the essential reasoning steps required by the problem."
        ),
        core_error_avoidance_rule=(
            "; ".join(negative[:3]) if negative else "Avoid unsupported leaps, arithmetic errors, and irrelevant paths."
        ),
        free_rule=f"Question domain: {payload.question_domain}",
    )


def _pad_teacher_response_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    response_start: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_width = prompt_width + response_width

    if response_length > 0:
        teacher_slice_start = response_start - 1
        teacher_slice_end = teacher_slice_start + response_length
        teacher_ids = teacher_ids[teacher_slice_start:teacher_slice_end]
        teacher_logprobs = teacher_logprobs[teacher_slice_start:teacher_slice_end]
    else:
        teacher_ids = teacher_ids[:0]
        teacher_logprobs = teacher_logprobs[:0]

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
        dynamic_rubric_enabled: bool = False,
        dynamic_max_rubrics_per_sample: int = 1,
        dynamic_selection_mode: str = "top_m",
        dynamic_weight_mode: str = "abs_points",
        dynamic_weight_temperature: float = 1.0,
        dynamic_include_negative: bool = True,
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
        self.dynamic_rubric_enabled = dynamic_rubric_enabled
        self.dynamic_max_rubrics_per_sample = dynamic_max_rubrics_per_sample
        self.dynamic_selection_mode = dynamic_selection_mode
        self.dynamic_weight_mode = dynamic_weight_mode
        self.dynamic_weight_temperature = dynamic_weight_temperature
        self.dynamic_include_negative = dynamic_include_negative

    def _render_chat_template_prompt(self, prompt_text: str) -> str:
        messages = build_teacher_scoring_messages(prompt_text)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _tokenize_teacher_prompt_text(self, prompt_text: str) -> list[int]:
        tokenized = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return normalize_token_ids(tokenized)

    def _active_rubric_entry(
        self,
        problem_id: str,
        global_step: int,
    ) -> RubricEntry | None:
        if (
            problem_id
            and self.active_rubric_lookup is not None
            and self.rubric_curriculum.should_use_self_mined(problem_id, global_step)
        ):
            return self.active_rubric_lookup(problem_id)
        return None

    async def _mine_dynamic_rubric_payload_from_response(
        self,
        problem: str,
        ground_truth: str,
        response_text: str,
    ) -> DynamicRubricPayload:
        if getattr(self, "_server_id_to_handle", None) and self._load_balancer is not None and hasattr(
            self.tokenizer, "decode"
        ):
            prompt_text = format_dynamic_rubric_mining_prompt(
                problem=problem,
                ground_truth=ground_truth,
                response_summary=summarize_response_text(response_text),
            )
            prompt_ids = self._tokenize_teacher_prompt_text(self._render_chat_template_prompt(prompt_text))
            try:
                teacher_output = await self.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "max_tokens": self.dynamic_rubric_mining_max_tokens,
                        "temperature": 0.0,
                    },
                )
                response_token_ids = normalize_token_ids(teacher_output.token_ids)
                rubric_response_text = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
                return parse_dynamic_rubric_response(rubric_response_text)
            except Exception as exc:
                print(f"[OPSD dynamic rubric] teacher mining failed, using heuristic rubric: {exc}")

        return build_dynamic_rubric_payload_from_response(
            problem=problem,
            ground_truth=ground_truth,
            response_text=response_text,
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
        print(f"=== current prompt ===  \n\n {teacher_prompt_text}")
        print(f"#########  use_rubric_prompt: {self.use_rubric_prompts} \n\n problem: {problem}")
        if self.use_rubric_prompts and problem:
            active_entry = self._active_rubric_entry(problem_id, global_step)
            rubric_prompt_text, _ = select_active_teacher_prompt(
                problem_id=problem_id,
                problem=problem,
                active_entry=active_entry,
                generic_rubric=self.generic_rubric,
            )
            teacher_prompt_text = self._render_chat_template_prompt(rubric_prompt_text)
            print(f"=== rubric prompt ===  \n\n {rubric_prompt_text}")
        if teacher_prompt_text is None:
            raise KeyError(
                "Missing problem text or `extra_info.teacher_prompt_text` for OPSD teacher prompt construction."
            )
        print(f"=== really using prompt ===  \n\n {teacher_prompt_text}")
        return self._tokenize_teacher_prompt_text(teacher_prompt_text)

    async def _teacher_prompt_specs_from_context(
        self,
        extra_info: dict | None,
        problem_id: str,
        problem: str,
        global_step: int,
        response_text: str = "",
        ground_truth: str = "",
    ) -> list[dict]:
        if not self.dynamic_rubric_enabled or not self.use_rubric_prompts or not problem:
            return [
                {
                    "prompt_ids": self._teacher_prompt_ids_from_context(extra_info, problem_id, problem, global_step),
                    "weight": 1.0,
                    "rubric_index": -1,
                }
            ]

        payload = await self._mine_dynamic_rubric_payload_from_response(
            problem=problem,
            ground_truth=ground_truth,
            response_text=response_text,
        )

        specs = select_dynamic_rubric_specs(
            payload=payload,
            max_rubrics=self.dynamic_max_rubrics_per_sample,
            selection_mode=self.dynamic_selection_mode,
            weight_mode=self.dynamic_weight_mode,
            weight_temperature=self.dynamic_weight_temperature,
            include_negative=self.dynamic_include_negative,
            seed_key=f"{problem_id}:{global_step}",
        )
        if not specs:
            return [
                {
                    "prompt_ids": self._teacher_prompt_ids_from_context(extra_info, problem_id, problem, global_step),
                    "weight": 1.0,
                    "rubric_index": -1,
                }
            ]

        prompt_specs = []
        for spec in specs:
            prompt_text = format_dynamic_teacher_scoring_prompt(problem=problem, payload=payload, spec=spec)
            print(f"=== dynamic rubric prompt ===  \n\n {prompt_text}")
            teacher_prompt_text = self._render_chat_template_prompt(prompt_text)
            prompt_specs.append(
                {
                    "prompt_ids": self._tokenize_teacher_prompt_text(teacher_prompt_text),
                    "weight": float(spec.weight),
                    "rubric_index": spec.rubric_index,
                }
            )
        return prompt_specs

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
        reward_model_batch = data.non_tensor_batch.get("reward_model")
        global_step = int(data.meta_info.get("global_steps", 0))
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]
        total_width = prompt_width + response_width

        sample_contexts = []
        prompt_specs_tasks = []
        response_lengths = []
        for i in range(len(data)):
            item = data[i : i + 1]
            response_ids = _valid_response_ids(item)
            extra_info = None if extra_info_batch is None else extra_info_batch[i]
            if extra_info is not None and not isinstance(extra_info, dict):
                extra_info = {"value": extra_info}
            problem_id = _coerce_text(None if problem_id_batch is None else problem_id_batch[i])
            if not problem_id:
                problem_id = _coerce_text((extra_info or {}).get("problem_id"))
            problem = _coerce_text(None if problem_batch is None else problem_batch[i])
            if not problem:
                problem = _coerce_text((extra_info or {}).get("problem"))
            response_text = (
                self.tokenizer.decode(response_ids, skip_special_tokens=True)
                if hasattr(self.tokenizer, "decode")
                else " ".join(map(str, response_ids))
            )
            reward_model_entry = None if reward_model_batch is None else reward_model_batch[i]
            if reward_model_entry is None:
                reward_model_entry = (extra_info or {}).get("reward_model")
            ground_truth = _ground_truth_from_context(reward_model_entry, extra_info)
            response_lengths.append(len(response_ids))
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            sample_contexts.append((response_ids, multi_modal_data))
            prompt_specs_tasks.append(
                asyncio.create_task(
                    self._teacher_prompt_specs_from_context(
                        extra_info,
                        problem_id,
                        problem,
                        global_step,
                        response_text=response_text,
                        ground_truth=ground_truth,
                    )
                )
            )

        sample_prompt_specs = await asyncio.gather(*prompt_specs_tasks)
        sample_tasks = []
        for (response_ids, multi_modal_data), prompt_specs in zip(sample_contexts, sample_prompt_specs, strict=True):
            prompt_tasks = []
            for prompt_spec in prompt_specs:
                sequence_ids = prompt_spec["prompt_ids"] + response_ids
                prompt_spec["response_start"] = len(prompt_spec["prompt_ids"])
                prompt_tasks.append(
                    asyncio.create_task(
                        self.compute_teacher_logprobs_single(
                            sequence_ids=sequence_ids,
                            multi_modal_data=multi_modal_data,
                        )
                    )
                )
            sample_tasks.append(prompt_tasks)

        outputs_by_sample = await asyncio.gather(*[asyncio.gather(*tasks) for tasks in sample_tasks])

        padded_teacher_ids = []
        padded_teacher_logprobs = []
        sample_multi_ids = []
        sample_multi_logprobs = []
        sample_weights = []
        sample_masks = []
        for outputs, prompt_specs, response_length in zip(
            outputs_by_sample, sample_prompt_specs, response_lengths, strict=True
        ):
            spec_ids = []
            spec_logprobs = []
            for (teacher_ids, teacher_logprobs), prompt_spec in zip(outputs, prompt_specs, strict=True):
                padded_ids, padded_logprobs = _pad_teacher_response_outputs(
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    prompt_width=prompt_width,
                    response_width=response_width,
                    response_start=prompt_spec["response_start"],
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
                spec_ids.append(padded_ids.squeeze(0))
                spec_logprobs.append(padded_logprobs.squeeze(0))

            padded_teacher_ids.append(spec_ids[0].unsqueeze(0))
            padded_teacher_logprobs.append(spec_logprobs[0].unsqueeze(0))

            if self.dynamic_rubric_enabled:
                multi_ids = torch.stack(spec_ids, dim=1)
                multi_logprobs = torch.stack(spec_logprobs, dim=1)
                weights = torch.zeros((total_width, len(prompt_specs)), dtype=torch.float32)
                mask = torch.zeros((total_width, len(prompt_specs)), dtype=torch.bool)
                if response_length > 0:
                    start = prompt_width
                    end = prompt_width + response_length
                    for rubric_idx, prompt_spec in enumerate(prompt_specs):
                        weights[start:end, rubric_idx] = float(prompt_spec["weight"])
                        mask[start:end, rubric_idx] = True
                sample_multi_ids.append(multi_ids)
                sample_multi_logprobs.append(multi_logprobs)
                sample_weights.append(weights)
                sample_masks.append(mask)

        batch_fields = {
            "teacher_ids": torch.cat(padded_teacher_ids),
            "teacher_logprobs": torch.cat(padded_teacher_logprobs),
        }
        if self.dynamic_rubric_enabled:
            max_rubrics = max(tensor.shape[1] for tensor in sample_multi_ids)
            padded_multi_ids = []
            padded_multi_logprobs = []
            padded_weights = []
            padded_masks = []
            for multi_ids, multi_logprobs, weights, mask in zip(
                sample_multi_ids, sample_multi_logprobs, sample_weights, sample_masks, strict=True
            ):
                pad_rubrics = max_rubrics - multi_ids.shape[1]
                if pad_rubrics > 0:
                    id_pad = torch.full(
                        (total_width, pad_rubrics, multi_ids.shape[-1]),
                        self.pad_token_id,
                        dtype=multi_ids.dtype,
                    )
                    logprob_pad = torch.zeros(
                        (total_width, pad_rubrics, multi_logprobs.shape[-1]),
                        dtype=multi_logprobs.dtype,
                    )
                    weight_pad = torch.zeros((total_width, pad_rubrics), dtype=weights.dtype)
                    mask_pad = torch.zeros((total_width, pad_rubrics), dtype=mask.dtype)
                    multi_ids = torch.cat([multi_ids, id_pad], dim=1)
                    multi_logprobs = torch.cat([multi_logprobs, logprob_pad], dim=1)
                    weights = torch.cat([weights, weight_pad], dim=1)
                    mask = torch.cat([mask, mask_pad], dim=1)
                padded_multi_ids.append(multi_ids.unsqueeze(0))
                padded_multi_logprobs.append(multi_logprobs.unsqueeze(0))
                padded_weights.append(weights.unsqueeze(0))
                padded_masks.append(mask.unsqueeze(0))

            batch_fields.update(
                {
                    "teacher_ids_multi": torch.cat(padded_multi_ids),
                    "teacher_logprobs_multi": torch.cat(padded_multi_logprobs),
                    "teacher_rubric_weights": torch.cat(padded_weights),
                    "teacher_rubric_mask": torch.cat(padded_masks),
                }
            )

        batch = TensorDict(batch_fields, batch_size=len(data))
        return DataProto(batch=batch)


class OPSDTeacherModelManager(TeacherModelManager):
    """Teacher model manager that scores student rollouts under privileged OPSD prompts."""

    def __init__(self, config: DictConfig, resource_pool=None):
        from verl_opsd.losses import ensure_opsd_distillation_losses_registered

        ensure_opsd_distillation_losses_registered()
        self.raw_config = config
        self._initialize_rubric_state(config)
        if hasattr(config, "get"):
            distillation_config = config.get("distillation")
        else:
            distillation_config = getattr(config, "distillation", None)
        if distillation_config is None:
            distillation_config = config
        super().__init__(config=distillation_config, resource_pool=resource_pool)
        self._start_rubric_worker()

    def _initialize_rubric_state(self, raw_config: DictConfig | dict | DistillationConfig) -> None:
        rubric_config = {}
        if hasattr(raw_config, "get"):
            rubric_config = raw_config.get("opsd_rubric", {}) or {}
        elif hasattr(raw_config, "opsd_rubric"):
            rubric_config = getattr(raw_config, "opsd_rubric") or {}

        self.opsd_rubric_enabled = bool(getattr(rubric_config, "get", lambda *_: None)("enabled", False))
        get_value = rubric_config.get if hasattr(rubric_config, "get") else lambda key, default=None: default
        dynamic_config = get_value("dynamic", {}) or {}
        dynamic_get_value = dynamic_config.get if hasattr(dynamic_config, "get") else lambda key, default=None: default
        self.dynamic_rubric_enabled = self.opsd_rubric_enabled and bool(dynamic_get_value("enabled", False))
        self.dynamic_rubric_max_rubrics_per_sample = int(dynamic_get_value("max_rubrics_per_sample", 1))
        self.dynamic_rubric_selection_mode = str(dynamic_get_value("selection_mode", "top_m"))
        self.dynamic_rubric_weight_mode = str(dynamic_get_value("weight_mode", "abs_points"))
        self.dynamic_rubric_weight_temperature = float(dynamic_get_value("weight_temperature", 1.0))
        self.dynamic_rubric_include_negative = bool(dynamic_get_value("include_negative", True))
        self.dynamic_rubric_mining_max_tokens = int(dynamic_get_value("mining_max_tokens", 1024))
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
        self.rubric_trace_examples_per_step = int(get_value("trace_examples_per_step", 3))
        self._pending_rubric_trace_seeds: dict[int, dict] = {}
        self._completed_rubric_trace_records: list[dict] = []
        self._rubric_trace_lock = threading.RLock()

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
            get_active_entry = getattr(self.rubric_memory, "get_active_entry", None)
            active_entry_before = get_active_entry(observation.problem_id) if callable(get_active_entry) else None
            request = self.rubric_memory.observe(observation)
            if request is not None:
                if hasattr(request, "correct_observation") and hasattr(request, "wrong_observation"):
                    self._store_pending_rubric_trace_seed(
                        request=request,
                        active_entry_before=active_entry_before,
                        global_step=global_step,
                    )
                    if not self.enqueue_rubric_request(request):
                        with self._rubric_trace_lock:
                            self._pending_rubric_trace_seeds.pop(id(request), None)
                else:
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
            dynamic_rubric_enabled=self.dynamic_rubric_enabled,
            dynamic_max_rubrics_per_sample=self.dynamic_rubric_max_rubrics_per_sample,
            dynamic_selection_mode=self.dynamic_rubric_selection_mode,
            dynamic_weight_mode=self.dynamic_rubric_weight_mode,
            dynamic_weight_temperature=self.dynamic_rubric_weight_temperature,
            dynamic_include_negative=self.dynamic_rubric_include_negative,
        )

    def enqueue_rubric_request(self, request: RubricMiningRequest) -> bool:
        if not self.opsd_rubric_enabled:
            return False
        try:
            self.pending_rubric_requests.put_nowait(request)
            return True
        except queue.Full:
            self.dropped_rubric_requests += 1
            return False

    def pending_rubric_request_count(self) -> int:
        return self.pending_rubric_requests.qsize()

    def drain_rubric_trace_records(self, max_records: int | None = None) -> list[dict]:
        self._ensure_rubric_trace_state()
        with self._rubric_trace_lock:
            if max_records is None or max_records < 0:
                records = list(self._completed_rubric_trace_records)
                self._completed_rubric_trace_records.clear()
                return records

            records = self._completed_rubric_trace_records[:max_records]
            del self._completed_rubric_trace_records[:max_records]
            return records

    def _ensure_rubric_trace_state(self) -> None:
        if not hasattr(self, "_pending_rubric_trace_seeds"):
            self._pending_rubric_trace_seeds = {}
        if not hasattr(self, "_completed_rubric_trace_records"):
            self._completed_rubric_trace_records = []
        if not hasattr(self, "_rubric_trace_lock"):
            self._rubric_trace_lock = threading.RLock()

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
        rubric_payload, trace_meta = self._mine_rubric_payload(request)
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
        self._record_completed_rubric_trace(
            request=request,
            entry=entry,
            trace_meta=trace_meta,
        )
        return entry

    def _mine_rubric_payload(self, request: RubricMiningRequest):
        if self._rubric_miner_fn is not None:
            payload = self._rubric_miner_fn(request)
            return payload, {"mining_mode": "custom_miner"}

        dynamic_rubric_enabled = getattr(self, "dynamic_rubric_enabled", False)
        if getattr(self, "server_manager", None) is None or not hasattr(self.tokenizer, "decode"):
            if dynamic_rubric_enabled:
                return build_dynamic_rubric_payload_from_request(request), {"mining_mode": "heuristic_dynamic"}
            return build_rubric_payload_from_request(request), {"mining_mode": "heuristic"}

        with self._teacher_service_lock:
            self.wake_up()
            try:
                return asyncio.run(self._mine_rubric_payload_with_teacher_service(request))
            finally:
                self.sleep()

    def _render_rubric_mining_prompt(self, request: RubricMiningRequest) -> str:
        correct_example_summary = summarize_response_text(request.correct_observation.response_text)
        wrong_example_summary = summarize_response_text(request.wrong_observation.response_text)
        if getattr(self, "dynamic_rubric_enabled", False):
            mining_prompt = format_dynamic_rubric_mining_prompt(
                problem=request.problem,
                ground_truth=request.ground_truth,
                response_summary=wrong_example_summary,
                correct_example_summary=correct_example_summary,
            )
        else:
            mining_prompt = format_rubric_mining_prompt(
                problem=request.problem,
                ground_truth=request.ground_truth,
                correct_example_summary=correct_example_summary,
                wrong_example_summary=wrong_example_summary,
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
        return normalize_token_ids(tokenized), prompt_text

    async def _mine_rubric_payload_with_teacher_service(self, request: RubricMiningRequest):
        prompt_ids, prompt_text = self._rubric_mining_prompt_ids(request)
        dynamic_rubric_enabled = getattr(self, "dynamic_rubric_enabled", False)
        sampling_params = {
            "max_tokens": getattr(self, "dynamic_rubric_mining_max_tokens", 1024) if dynamic_rubric_enabled else 256,
            "temperature": 0.0,
        }
        teacher_output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        response_token_ids = normalize_token_ids(teacher_output.token_ids)
        response_text = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        # mark
        if dynamic_rubric_enabled:
            parsed_response = parse_dynamic_rubric_response(response_text)
        else:
            parsed_response = parse_rubric_response(response_text)
        print(f"=== miner teacher ===: \n request: {request} \n \n rubric_response: \n {parsed_response}")
        return parsed_response, {
            "mining_mode": "teacher_service",
            "mining_prompt_text": prompt_text,
            "mining_response_text": response_text,
            "mining_sampling_params": sampling_params,
        }

    @staticmethod
    def _serialize_dataclass(value):
        if value is None:
            return None
        if is_dataclass(value):
            return {k: OPSDTeacherModelManager._serialize_dataclass(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): OPSDTeacherModelManager._serialize_dataclass(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OPSDTeacherModelManager._serialize_dataclass(v) for v in value]
        return value

    def _serialize_observation(self, observation):
        return self._serialize_dataclass(observation)

    def _serialize_rubric_entry(self, entry):
        return self._serialize_dataclass(entry)

    def _store_pending_rubric_trace_seed(
        self,
        request: RubricMiningRequest,
        active_entry_before: RubricEntry | None,
        global_step: int,
    ) -> None:
        self._ensure_rubric_trace_state()
        correct_observation = request.correct_observation
        wrong_observation = request.wrong_observation
        teacher_prompt_text_used = self._teacher_prompt_text_for_trace(
            observation=correct_observation,
            active_entry_before=active_entry_before,
            global_step=global_step,
        )
        seed = {
            "submitted_step": global_step,
            "teacher_prompt_text_used": teacher_prompt_text_used,
            "active_rubric_before": self._serialize_rubric_entry(active_entry_before),
        }
        with self._rubric_trace_lock:
            self._pending_rubric_trace_seeds[id(request)] = seed

    def _record_completed_rubric_trace(
        self,
        request: RubricMiningRequest,
        entry: RubricEntry,
        trace_meta: dict | None = None,
    ) -> None:
        trace_meta = trace_meta or {}
        self._ensure_rubric_trace_state()
        with self._rubric_trace_lock:
            seed = self._pending_rubric_trace_seeds.pop(id(request), {})
            record = {
                "trace_type": "rubric_trace",
                "problem_id": request.problem_id,
                "problem": request.problem,
                "ground_truth": request.ground_truth,
                "submitted_step": seed.get("submitted_step"),
                "updated_step": entry.updated_step,
                "teacher_prompt_text_used": seed.get("teacher_prompt_text_used", ""),
                "active_rubric_before": seed.get("active_rubric_before"),
                "correct_observation": self._serialize_observation(request.correct_observation),
                "wrong_observation": self._serialize_observation(request.wrong_observation),
                "stored_entry": self._serialize_rubric_entry(entry),
            }
            record.update(trace_meta)
            self._completed_rubric_trace_records.append(record)

    def _teacher_prompt_text_for_trace(
        self,
        observation,
        active_entry_before: RubricEntry | None,
        global_step: int,
    ) -> str:
        teacher_prompt_text = observation.teacher_prompt_text or ""
        if not self.opsd_rubric_enabled or not observation.problem:
            return teacher_prompt_text

        active_entry = None
        if (
            observation.problem_id
            and active_entry_before is not None
            and self.rubric_curriculum.should_use_self_mined(observation.problem_id, global_step)
        ):
            active_entry = active_entry_before

        rubric_prompt_text, _ = select_active_teacher_prompt(
            problem_id=observation.problem_id,
            problem=observation.problem,
            active_entry=active_entry,
            generic_rubric=self.generic_rubric,
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = build_teacher_scoring_messages(rubric_prompt_text)
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return rubric_prompt_text

    def shutdown_rubric_worker(self) -> None:
        if self._rubric_worker_thread is None:
            return
        self._rubric_worker_stop.set()
        self.pending_rubric_requests.put(None)
        self._rubric_worker_thread.join(timeout=1.0)
