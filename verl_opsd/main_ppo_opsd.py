"""Launch verl PPO with OPSD-specific rubric-conditioned teacher prompt handling."""

import os

from verl_opsd import losses as _losses  # noqa: F401
from verl.trainer import main_ppo as main_ppo_module


BaseTaskRunner = main_ppo_module.TaskRunner


def apply_opsd_teacher_patch():
    """Route verl teacher construction through OPSD in the current Python process."""
    from verl_opsd.losses import ensure_opsd_distillation_losses_registered
    from verl_opsd.teacher import OPSDTeacherModelManager

    import verl.experimental.agent_loop.agent_loop as agent_loop_module
    import verl.experimental.teacher_loop as teacher_loop_module
    import verl.experimental.teacher_loop.teacher_model as teacher_model_module

    ensure_opsd_distillation_losses_registered()
    teacher_loop_module.TeacherModelManager = OPSDTeacherModelManager
    teacher_model_module.TeacherModelManager = OPSDTeacherModelManager
    agent_loop_module.TeacherModelManager = OPSDTeacherModelManager
    os.environ["VERL_USE_OPSD_TEACHER"] = "1"
    return OPSDTeacherModelManager


class OPSDTaskRunner(BaseTaskRunner):
    """TaskRunner variant that applies OPSD routing inside the Ray actor process."""

    def run(self, config):
        apply_opsd_teacher_patch()
        return super().run(config)


def install_opsd_task_runner():
    """Make verl's Hydra entrypoint create OPSD-aware Ray TaskRunner actors."""
    main_ppo_module.TaskRunner = OPSDTaskRunner
    return OPSDTaskRunner


apply_opsd_teacher_patch()
install_opsd_task_runner()

main = main_ppo_module.main


if __name__ == "__main__":
    main()
