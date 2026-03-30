from __future__ import annotations

import asyncio
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
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

    def _teacher_prompt_ids_from_extra_info(self, extra_info: dict | None) -> list[int]:
        extra_info = extra_info or {}

        teacher_prompt_ids = extra_info.get("teacher_prompt_ids")
        if teacher_prompt_ids is not None:
            return normalize_token_ids(teacher_prompt_ids)

        teacher_prompt_text = extra_info.get("teacher_prompt_text")
        if teacher_prompt_text is None:
            raise KeyError("Missing `extra_info.teacher_prompt_text` for OPSD teacher prompt construction.")

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
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        tasks = []
        response_lengths = []
        for i in range(len(data)):
            item = data[i : i + 1]
            response_ids = _valid_response_ids(item)
            extra_info = None if extra_info_batch is None else extra_info_batch[i]
            teacher_prompt_ids = self._teacher_prompt_ids_from_extra_info(extra_info)
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
        )

