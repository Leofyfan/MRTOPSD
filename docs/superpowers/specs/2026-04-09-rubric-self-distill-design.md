# Rubric-Based Self-Distillation Design

Date: 2026-04-09

## Goal

Extend the current OPSD-on-verl training path to support rubric-based self-distillation.

The new teacher privileged information should not be a fixed reference solution. Instead:

1. For each problem, compare one correct rollout answer and one wrong rollout answer.
2. Use the same teacher model to summarize a structured rubric explaining why the correct answer succeeds.
3. Cache that rubric per problem.
4. Use the cached rubric as privileged conditioning for later teacher scoring and distillation.
5. If no correct rollout answer is available, fall back to a generic rubric.

The design must preserve compatibility with verl async distillation patterns. Fresh rubrics generated from the current step must not block the current step's teacher logprob path.

## Current Baseline

The current implementation already has:

- dataset-side privileged teacher prompt construction in `verl_opsd/prepare_dataset.py`
- OPSD-specific teacher manager in `verl_opsd/teacher.py`
- verl distillation integration through `verl_opsd/main_ppo_opsd.py`
- `forward_kl_topk` distillation in the verl distillation stack

The current privileged signal is effectively a fixed teacher prompt. The new design replaces that fixed privileged condition with a curriculum-controlled rubric condition.

## High-Level Design

### Main idea

The system will maintain a per-problem rubric memory.

For any problem:

- if a self-mined rubric exists and the curriculum selects it, teacher scoring uses that rubric
- otherwise teacher scoring uses a generic rubric

Rubric generation is asynchronous and lagged:

- step `t` uses the current cached rubric state
- rollout results from step `t` may update the rubric cache
- the updated rubric is used only in later encounters with the same problem

This keeps rubric mining out of the current-step teacher critical path.

### Runtime layers

The runtime is split into four logical layers:

1. Rollout / reward main path
2. Rubric mining side path
3. Rubric memory
4. Teacher scoring main path

Only layer 4 directly affects the current step's distillation target.

## Rubric Semantics

Each rubric is a structured privileged summary for one problem.

Each active rubric contains:

- `core_correctness_rule`
- `core_key_steps_rule`
- `core_error_avoidance_rule`
- `free_rule`

The first three are required for self-mined rubrics. `free_rule` is optional.

The rubric is not intended to restate the full reference solution. It is intended to summarize the decision rules that make an answer correct and expose the specific failure mode seen in a wrong rollout answer.

## Rubric Sources

Two rubric sources are supported:

- `generic`
- `self_mined`

### Generic rubric

Generic rubric is a hand-authored structured rubric template for math reasoning tasks. It uses the same field schema as self-mined rubrics so that teacher prompt formatting stays stable.

### Self-mined rubric

Self-mined rubric is generated from:

- `problem`
- `ground_truth`
- one correct rollout answer
- one hard wrong rollout answer

The wrong answer should be selected from verifier-failed, non-trivial candidates. Empty, degenerate, or obviously broken outputs should be excluded.

## Curriculum

The curriculum controls rubric selection during distillation, not whether rubric mining runs.

### Curriculum stages

Three runtime stages are defined:

- `warmup`
- `mix`
- `mature`

### Stage behavior

In `warmup`:

- generic rubric is used almost always
- self-mined rubrics may still be generated and cached

In `mix`:

- generic and self-mined rubrics are mixed
- self-mined usage probability increases with `global_step`

In `mature`:

- if a valid self-mined rubric exists, use it
- otherwise fall back to generic rubric

### Important constraint

The curriculum changes the teacher conditioning distribution:

- baseline: fixed privileged teacher prompt
- new: generic rubric early, self-mined rubric later

This is a teacher-supervision curriculum. The loss function stays `forward_kl_topk`.

## Async Compatibility

The design explicitly avoids current-step synchronous rubric mining.

### Accepted async behavior

- current step reads cached rubric or generic fallback
- current step does not wait for newly generated rubric
- rubric mining is triggered after rollout/reward information is available
- rubric update is consumed only by later steps

### Reason

If current-step teacher scoring had to wait for current-step rubric mining, rubric generation would become a new blocking stage ahead of teacher logprob retrieval and would reduce the benefit of verl async distillation scheduling.

## Data Model

### Dataset-level identifiers

Each training sample must expose a stable problem identifier:

- `uid`
- `problem_id`

`problem_id` is the key used by rubric memory.

### Rubric memory entry

Each rubric cache entry should contain at least:

- `problem_id`
- `rubric_source`
- `rubric_version`
- `updated_step`
- `course_stage`
- `rubric_payload`
- `correct_example_summary`
- `wrong_example_summary`

Only `rubric_payload` is required by teacher scoring. The summaries are retained for debugging and auditability.

## Teacher Prompting

Two prompt families are required.

### Rubric miner prompt

Purpose:

- summarize structured rubric from correct vs wrong contrast

Input:

- problem
- ground truth
- correct answer
- wrong answer

Output:

- structured rubric fields

### Teacher scoring prompt

Purpose:

- condition teacher scoring on the active rubric

Input:

- problem
- active rubric
- student response context

Output:

- teacher prompt logprobs for distillation

The same underlying teacher model is reused for both miner and scorer roles. The prompts must stay separate.

## Code Boundaries

### New modules

Add the following OPSD-specific modules:

- `verl_opsd/rubric_memory.py`
- `verl_opsd/rubric_miner.py`
- `verl_opsd/rubric_curriculum.py`
- `verl_opsd/rubric_prompting.py`

### Existing files to update

- `verl_opsd/prepare_dataset.py`
  - add stable `problem_id`
  - preserve required fields for rubric mining

- `verl_opsd/teacher.py`
  - replace fixed privileged prompt construction with active rubric selection
  - keep teacher logprob serving responsibility

- `verl_opsd/main_ppo_opsd.py`
  - initialize and inject rubric-related managers and hooks

- `third_party/verl/verl/trainer/ppo/ray_trainer.py`
  - add the smallest possible post-step hook to submit rubric update tasks
  - do not make rubric mining part of the current-step blocking path

### Files to avoid changing heavily

- `third_party/verl/verl/trainer/distillation/losses.py`
- `third_party/verl/verl/experimental/teacher_loop/*`
- `third_party/verl/verl/experimental/agent_loop/*`

Reason:

The design changes teacher conditioning, not the distillation math itself. The implementation should remain recipe-local as much as possible.

## Distillation-Critical Insertion Points

There are only two insertion points that matter for the distillation path.

### A. Before teacher scoring prompt construction

Read:

- cached self-mined rubric if eligible
- else generic rubric

This directly changes the current teacher supervision signal.

### B. After rollout/reward information is available

Extract:

- candidate correct answer
- candidate hard wrong answer

Then submit async rubric mining and update rubric memory.

This only affects future teacher supervision signals.

## Quality Gates

Self-mined rubric must pass all of the following before becoming active.

### Sample gate

Require:

- one verifier-correct answer
- one non-trivial verifier-wrong answer

Reject:

- empty outputs
- extremely short outputs
- malformed outputs

### Format gate

Self-mined rubric must parse into the expected structured schema.

Required non-empty fields:

- `core_correctness_rule`
- `core_key_steps_rule`
- `core_error_avoidance_rule`

### Length gate

Each rule must stay within a bounded token range to prevent useless or overly long teacher prompts.

### Activation gate

Even a successfully cached self-mined rubric is not automatically active. It becomes eligible only when the curriculum stage allows it.

## Metrics and Success Criteria

The main evaluation target is validation score over training steps.

### Primary validation metrics

The implementation must log and track:

- `score/mean@1`
- `score/mean@4`
- `score/best@4`

These must be computed on the validation setup that mirrors the current test-like evaluation path.

To remove ambiguity:

- validation should generate 4 responses per problem
- `score/mean@4` is the average score over the 4 responses
- `score/best@4` is the best-of-4 score
- `score/mean@1` should be reported from a deterministic single-sample view derived from the same 4-response validation batch

The intended implementation is:

- use the first sampled response as the `@1` view
- use all 4 sampled responses for the `@4` view

This keeps metric logging unified inside one validation pass instead of requiring a separate `n=1` validation run.

### Secondary validation metric

Also track:

- `acc` metrics when available

But score-based metrics are the main comparison target.

### Diagnostic metrics

Keep existing distillation diagnostics and add rubric-specific diagnostics.

Important diagnostics:

- `distillation/loss`
- `distillation/student_mass`
- `distillation/teacher_mass`
- `rubric/cache_hit_rate`
- `rubric/self_mined_usage_rate`
- `rubric/generic_usage_rate`
- `rubric/update_success_rate`
- `rubric/active_prompt_tokens_mean`
- `perf/time_per_step`

These diagnostics are not the primary optimization target. They are used to interpret score movement and failure modes.

## Verification Plan

### Functional verification

Verify:

- cache miss falls back to generic rubric
- cache hit can switch teacher prompt to self-mined rubric
- async rubric update does not block current-step teacher scoring
- rubric memory updates are visible on later encounters

### Training stability verification

Compare:

- baseline OPSD-on-verl
- generic-rubric-only mode
- full rubric curriculum mode

Watch:

- validation score curves
- training throughput
- prompt length growth
- OOM behavior

### Effectiveness verification

Success means the rubric curriculum improves validation score curves over the baseline under comparable training budget, especially:

- `score/mean@1`
- `score/mean@4`
- `score/best@4`

## Implementation Strategy

All components should be implemented in one development pass, but controlled by config at runtime.

This means:

- full system support lands together
- curriculum stage and usage rates are config-driven
- generic-only and self-mined-enabled modes are both available without additional code changes

Recommended implementation order inside the single pass:

1. Add data identifiers and rubric config surfaces
2. Add generic rubric path into teacher scoring
3. Add rubric memory
4. Add rubric miner
5. Add async update hook
6. Add curriculum selector
7. Add metrics and validation reporting

## Non-Goals

This design does not include:

- loading multiple teacher models
- changing the distillation loss formula away from top-k forward KL
- making current-step rubric mining block teacher scoring
- redesigning verl async schedulers

## Expected Outcome

After implementation, OPSD-on-verl should support a new training mode where:

- teacher supervision starts from generic rubric conditioning
- later transitions toward per-problem self-mined rubric conditioning
- self-mined rubric is derived from correct vs wrong rollout comparisons
- asynchronous compatibility is preserved by lagged rubric updates
- evaluation focuses on score progression over steps, especially `mean@1`, `mean@4`, and `best@4`
