from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class RubricCurriculum:
    warmup_steps: int
    mix_steps: int
    seed: int

    def course_stage(self, global_step: int) -> str:
        if global_step < self.warmup_steps:
            return "warmup"
        if global_step < self.warmup_steps + self.mix_steps:
            return "mix"
        return "mature"

    def should_use_self_mined(self, problem_id: str, global_step: int) -> bool:
        if global_step < self.warmup_steps:
            return False
        if global_step >= self.warmup_steps + self.mix_steps:
            return True
        if self.mix_steps <= 0:
            return True

        progress = (global_step - self.warmup_steps + 1) / self.mix_steps
        digest = sha256(f"{self.seed}:{problem_id}".encode("utf-8")).digest()
        sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return sample < progress
