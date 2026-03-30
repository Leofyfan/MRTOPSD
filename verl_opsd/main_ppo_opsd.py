"""Launch verl PPO with OPSD-specific teacher prompt handling."""

import os

from verl_opsd import losses as _losses  # noqa: F401
from verl_opsd.teacher import OPSDTeacherModelManager

import verl.experimental.teacher_loop as teacher_loop_module
import verl.experimental.teacher_loop.teacher_model as teacher_model_module

teacher_loop_module.TeacherModelManager = OPSDTeacherModelManager
teacher_model_module.TeacherModelManager = OPSDTeacherModelManager
os.environ.setdefault("VERL_USE_OPSD_TEACHER", "1")

from verl.trainer.main_ppo import main


if __name__ == "__main__":
    main()
