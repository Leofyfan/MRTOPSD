"""Register OPSD-specific aliases for verl distillation losses."""

from verl.trainer.distillation.losses import (
    DistillationLossSettings,
    compute_distillation_loss_reverse_kl_estimator,
    compute_forward_kl_topk,
    register_distillation_loss,
)


@register_distillation_loss(DistillationLossSettings(names=["opsd_forward_kl_topk"], use_topk=True))
def compute_opsd_forward_kl_topk(*args, **kwargs):
    return compute_forward_kl_topk(*args, **kwargs)


@register_distillation_loss(
    DistillationLossSettings(names=["opsd_kl", "opsd_k1", "opsd_k3"], use_estimator=True)
)
def compute_opsd_reverse_kl_estimator(*args, **kwargs):
    return compute_distillation_loss_reverse_kl_estimator(*args, **kwargs)

