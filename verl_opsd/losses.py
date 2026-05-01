"""Register OPSD-specific aliases for verl distillation losses."""

import torch
import torch.nn.functional as F

from verl.trainer.distillation.losses import (
    DISTILLATION_LOSS_REGISTRY,
    DISTILLATION_SETTINGS_REGISTRY,
    DistillationLossSettings,
    compute_distillation_loss_reverse_kl_estimator,
    compute_forward_kl_topk,
    register_distillation_loss,
)
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor


def _kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def _as_batch_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.values().unsqueeze(0) if tensor.is_nested else tensor


@register_distillation_loss(DistillationLossSettings(names=["opsd_forward_kl_topk"], use_topk=True))
def compute_opsd_forward_kl_topk(*args, **kwargs):
    return compute_forward_kl_topk(*args, **kwargs)


@register_distillation_loss(DistillationLossSettings(names=["opsd_multi_forward_kl_topk"], use_topk=True))
def compute_opsd_multi_forward_kl_topk(*args, **kwargs):
    return compute_forward_kl_topk(*args, **kwargs)


def compute_multi_forward_kl_topk_logits(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_rubric_weights: torch.Tensor,
    teacher_rubric_mask: torch.Tensor | None,
    config,
    data_format: str,
) -> dict[str, torch.Tensor]:
    del data_format
    teacher_topk_log_probs = _as_batch_tensor(teacher_topk_log_probs)
    teacher_topk_ids = _as_batch_tensor(teacher_topk_ids).long()
    teacher_rubric_weights = _as_batch_tensor(teacher_rubric_weights).float()
    teacher_rubric_mask = (
        torch.ones_like(teacher_rubric_weights, dtype=torch.bool)
        if teacher_rubric_mask is None
        else _as_batch_tensor(teacher_rubric_mask).bool()
    )

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        teacher_rubric_weights = slice_input_tensor(teacher_rubric_weights, dim=1)
        teacher_rubric_mask = slice_input_tensor(teacher_rubric_mask, dim=1)

    assert teacher_topk_log_probs.shape[:3] == teacher_topk_ids.shape[:3]
    assert teacher_topk_log_probs.shape[:2] == student_logits.shape[:2]
    assert teacher_rubric_weights.shape == teacher_rubric_mask.shape == teacher_topk_log_probs.shape[:3]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    expanded_student_log_probs = student_log_probs.unsqueeze(2).expand(
        -1,
        -1,
        teacher_topk_ids.shape[2],
        -1,
    )
    student_topk_log_probs = torch.gather(expanded_student_log_probs, dim=-1, index=teacher_topk_ids)

    loss_config = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    rubric_weights = teacher_rubric_weights * teacher_rubric_mask.float()
    rubric_weights = rubric_weights / rubric_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    per_rubric_losses = _kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)
    per_rubric_student_mass = student_topk_log_probs.exp().sum(dim=-1)
    per_rubric_teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    return {
        "distillation_losses": (per_rubric_losses * rubric_weights).sum(dim=-1),
        "student_mass": (per_rubric_student_mass * rubric_weights).sum(dim=-1),
        "teacher_mass": (per_rubric_teacher_mass * rubric_weights).sum(dim=-1),
    }


@register_distillation_loss(
    DistillationLossSettings(names=["opsd_kl", "opsd_k1", "opsd_k3"], use_estimator=True)
)
def compute_opsd_reverse_kl_estimator(*args, **kwargs):
    return compute_distillation_loss_reverse_kl_estimator(*args, **kwargs)


def ensure_opsd_distillation_losses_registered() -> None:
    """Register OPSD losses if the current process has an unpatched verl registry."""
    registrations = [
        (
            DistillationLossSettings(names=["opsd_forward_kl_topk"], use_topk=True),
            compute_opsd_forward_kl_topk,
        ),
        (
            DistillationLossSettings(names=["opsd_multi_forward_kl_topk"], use_topk=True),
            compute_opsd_multi_forward_kl_topk,
        ),
        (
            DistillationLossSettings(names=["opsd_kl", "opsd_k1", "opsd_k3"], use_estimator=True),
            compute_opsd_reverse_kl_estimator,
        ),
    ]
    for settings, loss_fn in registrations:
        for name in settings.names:
            if name not in DISTILLATION_LOSS_REGISTRY:
                DISTILLATION_LOSS_REGISTRY[name] = loss_fn
            if name not in DISTILLATION_SETTINGS_REGISTRY:
                DISTILLATION_SETTINGS_REGISTRY[name] = settings
