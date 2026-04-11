# Current `verl` Training Logic for OPSD

This document describes the training pipeline that is actually running in this repository today, with direct links to the code that implements each stage.

## Code Map

- Launch script and Hydra overrides: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L118)
- OPSD launch wrapper: [main_ppo_opsd.py](/root/MRTOPSD/verl_opsd/main_ppo_opsd.py#L1)
- verl main entry: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L36)
- Dataset conversion: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L52)
- Trainer construction: [TaskRunner.run](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L360)
- Trainer core loop: [RayPPOTrainer.fit](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1299)
- Validation loop: [_validate](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L524)
- OPSD teacher path: [teacher.py](/root/MRTOPSD/verl_opsd/teacher.py#L19)
- Distillation loss: [distillation/losses.py](/root/MRTOPSD/third_party/verl/verl/trainer/distillation/losses.py#L203)
- Actor update implementation: [dp_actor.py](/root/MRTOPSD/third_party/verl/verl/workers/actor/dp_actor.py#L557)
- FSDP actor worker wrapper: [fsdp_workers.py](/root/MRTOPSD/third_party/verl/verl/workers/fsdp_workers.py#L1036)
- Reward routing: [reward.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/reward.py#L154)
- `math_dapo` scorer: [math_dapo.py](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L242)
- Wandb logger: [tracking.py](/root/MRTOPSD/third_party/verl/verl/utils/tracking.py#L58)
- Checkpoint save path: [_save_checkpoint](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L897)

## 1. What This Training Really Is

This is not a plain reward-driven PPO run. The current recipe is a `verl`-orchestrated OPSD distillation pipeline with rollout, teacher scoring, validation, wandb logging, and checkpointing.

The current behavior is defined directly by the launch overrides in [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L122):

- `algorithm.adv_estimator=grpo`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L122)
- `distillation.enabled=True`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L174)
- teacher inference enabled: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L176)
- top-k forward KL distillation: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L188)
- task reward disabled as optimization target: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L190)
- policy gradient disabled in distillation branch: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L191)

The shortest correct summary is:

1. The student sees the plain problem and samples a response.
2. The teacher sees a privileged prompt that includes the reference solution.
3. The teacher scores the student response token by token.
4. The actor is updated mainly by top-k forward-KL distillation toward the teacher.
5. Reward and validation are still computed and logged, but they are not the main optimization driver in the current recipe.

## 2. Launch and Control Flow

The shell entry is [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L118). It launches:

- `python -m verl_opsd.main_ppo_opsd`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L119)

That wrapper does two OPSD-specific things before jumping into verl:

- register the OPSD teacher manager: [main_ppo_opsd.py](/root/MRTOPSD/verl_opsd/main_ppo_opsd.py#L11)
- import OPSD loss aliases: [main_ppo_opsd.py](/root/MRTOPSD/verl_opsd/main_ppo_opsd.py#L5)

Then it forwards into verl's normal main:

- `main()` in [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L36)

The runtime control flow is:

1. Hydra loads the trainer config: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L36)
2. Ray is initialized in `run_ppo()`: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L50)
3. A remote `TaskRunner` is created: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L80)
4. `TaskRunner.run()` builds datasets and the trainer: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L360)
5. `trainer.init_workers()` initializes actor, rollout, teacher, reward, and checkpoint managers: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L391)
6. `trainer.fit()` executes training: [main_ppo.py](/root/MRTOPSD/third_party/verl/verl/trainer/main_ppo.py#L395)

## 3. Dataset Format and Why OPSD Is Special

The verl-compatible parquet data is built by [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L52).

The important output fields are:

- student prompt in `prompt`: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L77)
- `data_source="math_dapo"`: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L86)
- validation ground truth in `reward_model.ground_truth`: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L87)
- privileged teacher prompt text in `extra_info.teacher_prompt_text`: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L88)
- stable sample id in `uid`: [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L93)

The student and teacher do not see the same prompt:

- student prompt builder: [build_student_messages](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L17)
- teacher prompt builder: [build_teacher_messages](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L27)

That teacher prompt includes the reference solution, which is the core OPSD idea.

## 4. Main Runtime Roles

Inside [RayPPOTrainer.__init__](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L260), the trainer decides which roles are active:

- actor / rollout hybrid engine required: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L282)
- reference policy enabled or not: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L292)
- teacher policy enabled or not: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L293)
- reward model path enabled or not: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L295)
- critic enabled or not: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L297)

For the current OPSD run:

- teacher is enabled
- critic is effectively off
- actor uses FSDP2
- rollout and teacher inference are separate runtime components

Worker-side initialization happens later in [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L823):

- actor / rollout worker group init: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L823)
- reward loop manager init: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L830)
- teacher manager init: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L846)
- async rollout manager init: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L880)
- checkpoint engine manager init: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L887)

## 5. OPSD Teacher Integration

The custom teacher path is implemented in [teacher.py](/root/MRTOPSD/verl_opsd/teacher.py#L19).

The core logic is:

1. read `teacher_prompt_text` from `extra_info`: [_teacher_prompt_ids_from_extra_info](/root/MRTOPSD/verl_opsd/teacher.py#L92)
2. extract the valid student response tokens: [_valid_response_ids](/root/MRTOPSD/verl_opsd/teacher.py#L34)
3. concatenate `teacher_prompt_ids + response_ids`: [teacher.py](/root/MRTOPSD/verl_opsd/teacher.py#L136)
4. request teacher `prompt_logprobs` with `temperature=1.0`: [_teacher_sampling_params](/root/MRTOPSD/verl_opsd/teacher.py#L19)
5. pad the teacher outputs back onto response positions: [_pad_teacher_response_outputs](/root/MRTOPSD/verl_opsd/teacher.py#L40)

The manager class that verl actually instantiates is [OPSDTeacherModelManager](/root/MRTOPSD/verl_opsd/teacher.py#L179).

The trainer switches to that manager here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L847)

## 6. One Training Step, End to End

The full training loop is [RayPPOTrainer.fit](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1299). The per-step body starts here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1346)

### 6.1 Load a batch

The trainer receives one batch from the parquet-backed dataloader and wraps it into `DataProto`:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1359)

### 6.2 Generate student responses

The rollout input is built, then generation is dispatched through the async rollout manager:

- build generation batch: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1367)
- generate responses: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1378)

This is the student sampling stage.

### 6.3 Compute teacher targets

If teacher distillation is active, the trainer computes teacher outputs here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1420)

That call routes into the OPSD teacher manager described in Section 5.

### 6.4 Build masks and token statistics

The trainer ensures `response_mask` exists, optionally balances token load, and stores global token counts:

- response mask: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1425)
- balance batch: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1431)
- token counts: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1434)

### 6.5 Compute reward signals

Reward extraction is triggered here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1443)

The generic extraction function is:

- [extract_reward](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/reward.py#L154)

Reward routing by `data_source` is here:

- [default_compute_score](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/__init__.py#L19)

Because OPSD data is labeled `math_dapo`, it uses:

- [math_dapo.compute_score](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L242)

That scorer returns a dict with:

- `score`: [math_dapo.py](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L268)
- `acc`: [math_dapo.py](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L270)
- `pred`: [math_dapo.py](/root/MRTOPSD/third_party/verl/verl/utils/reward_score/math_dapo.py#L271)

### 6.6 Recompute old log-probs

The trainer recomputes student log-probs on the sampled responses here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1467)

This stage produces:

- `old_log_probs`
- response entropy
- `perf/mfu/actor_infer`

### 6.7 Optional reference and critic branch

If enabled, the trainer would compute:

- reference log-probs: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1501)
- critic values: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1507)

In the current OPSD recipe, critic is effectively off, so the value-model update path is skipped.

### 6.8 Compute advantages and returns

The trainer writes:

- `token_level_scores`: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1516)
- `token_level_rewards`: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1528)

Then it calls:

- [compute_advantage(...)](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1550)

Even though `advantages` and `returns` still exist in the batch, the current update is not driven mainly by task reward because:

- `use_task_rewards=False`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L190)
- `use_policy_gradient=False`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L191)

### 6.9 Update the actor

The actor update is triggered here:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1570)

The trainer-side wrapper is:

- [_update_actor](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1227)

That call goes into the FSDP actor worker:

- [fsdp_workers.py](/root/MRTOPSD/third_party/verl/verl/workers/fsdp_workers.py#L1036)

and then into the actual actor optimization loop:

- [dp_actor.py](/root/MRTOPSD/third_party/verl/verl/workers/actor/dp_actor.py#L557)

The current distillation path is:

- combine PPO metrics with distillation loss: [distillation_ppo_loss](/root/MRTOPSD/third_party/verl/verl/trainer/distillation/losses.py#L203)
- top-k forward-KL metric extraction: [compute_forward_kl_topk](/root/MRTOPSD/third_party/verl/verl/trainer/distillation/losses.py#L284)
- OPSD-specific alias registration: [verl_opsd/losses.py](/root/MRTOPSD/verl_opsd/losses.py#L11)

So the practical update rule is:

1. student rollout is generated
2. teacher top-k log-probs are collected
3. distillation loss is computed on response tokens
4. actor weights are updated under FSDP

### 6.10 Save, sync, validate, and log

After the actor update, the trainer may:

- save a checkpoint: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1585)
- push fresh weights to rollout replicas: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1595)
- run validation: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1607)
- log metrics: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1666)

## 7. Validation Logic

Validation is implemented in [_validate](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L524).

The validation flow is:

1. repeat validation prompts if needed: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L544)
2. generate responses: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L567)
3. extract reward / score: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L600)
4. collect `reward`, `acc`, `pred`, and related infos: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L606)
5. aggregate into `val-core/*` and `val-aux/*`: [_val_metrics_update](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L649)

Validation metric aggregation rules live in:

- [metric_utils.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/metric_utils.py#L511)

That code generates:

- `mean@N`
- `std@N`
- `best@N/*`
- `worst@N/*`
- `maj@N/*`

and the trainer labels the most important validation series as `val-core/*`:

- [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L653)

## 8. Wandb and Metric Flow

The training script enables wandb here:

- `trainer.logger=["console","wandb"]`: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L40)
- wandb env vars: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L102)
- project / experiment / output dir wiring: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L197)

The actual logger is created here:

- [Tracking(...)](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1301)

The wandb backend is initialized here:

- [wandb.init(...)](/root/MRTOPSD/third_party/verl/verl/utils/tracking.py#L71)

Metrics are pushed here:

- per-step scalar logging: [tracking.py](/root/MRTOPSD/third_party/verl/verl/utils/tracking.py#L181)
- trainer-side `logger.log(...)`: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1667)

Validation sample tables are logged here:

- [val/generations table](/root/MRTOPSD/third_party/verl/verl/utils/tracking.py#L409)

## 9. Checkpoint Logic

Checkpoint save logic is implemented in [_save_checkpoint](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L897).

The save trigger is:

- last step, or
- `global_step % save_freq == 0`, or
- forced safety save

See:

- [save condition](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1585)

The default local output root is defined here:

- [OUTPUT_DIR](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L11)

The current default run path is therefore:

- `/root/autodl-tmp/opsd_verl_runs/${EXPERIMENT_NAME}`

Inside each `global_step_N` directory, the trainer saves:

- actor checkpoint folder: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L906)
- dataloader state `data.pt`: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L945)
- latest step marker: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L960)

The actor-side save entry for FSDP workers is:

- [fsdp_workers.py](/root/MRTOPSD/third_party/verl/verl/workers/fsdp_workers.py#L1210)

An actual run directory in this workspace is:

- [qwen3_4b_opsd_verl](/root/autodl-tmp/opsd_verl_runs/qwen3_4b_opsd_verl)

## 10. Metric Groups and Where They Come From

The trainer assembles one metric dict per step in [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1350), then augments it from several sources:

- training progress: [ray_trainer.py](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1635)
- batch statistics: [compute_data_metrics](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/metric_utils.py#L81)
- timing: [compute_timing_metrics](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/metric_utils.py#L228)
- throughput: [compute_throughout_metrics](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/metric_utils.py#L270)
- variance proxy: [compute_variance_proxy_metrics](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/metric_utils.py#L306)

Important metric families are:

- `training/*`: step and epoch counters
- `actor/*`: actor-side loss, entropy, grad norm, lr
- `distillation/*`: top-k KL distillation statistics
- `critic/*`: batch reward / advantage / return statistics, even when critic network is off
- `prompt_length/*`, `response_length/*`: sequence length statistics
- `timing_s/*`, `timing_per_token_ms/*`: runtime timing
- `perf/*`: throughput, MFU, memory
- `val-core/*`, `val-aux/*`: validation aggregates

## 11. Multi-GPU Mental Model

The current training is already organized the way verl expects for multi-GPU and future scale-out:

- actor training under FSDP: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L143)
- rollout inference config: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L154)
- teacher inference config: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L176)
- GPU and node counts passed into trainer: [run_opsd_4b_verl.sh](/root/MRTOPSD/scripts/run_opsd_4b_verl.sh#L201)

For Qwen3-4B on 2 GPUs, the current recommendation remains:

- keep `ROLLOUT_TP=1`
- keep `TEACHER_TP=1`

because that usually gives better throughput than collapsing everything into a single tensor-parallel group.

## 12. Shortest Accurate Mental Model

If you only want the shortest possible mental model:

1. Student prompt is built in [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L17).
2. Teacher prompt is built in [prepare_dataset.py](/root/MRTOPSD/verl_opsd/prepare_dataset.py#L27).
3. Student generates through [generate_sequences](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L1378).
4. Teacher scores the student response through [teacher.py](/root/MRTOPSD/verl_opsd/teacher.py#L128).
5. Distillation loss is computed in [distillation/losses.py](/root/MRTOPSD/third_party/verl/verl/trainer/distillation/losses.py#L203).
6. Actor update runs in [dp_actor.py](/root/MRTOPSD/third_party/verl/verl/workers/actor/dp_actor.py#L557).
7. Validation runs in [_validate](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L524).
8. Metrics go to wandb through [tracking.py](/root/MRTOPSD/third_party/verl/verl/utils/tracking.py#L181).
9. Checkpoints are saved through [_save_checkpoint](/root/MRTOPSD/third_party/verl/verl/trainer/ppo/ray_trainer.py#L897).

That is the current end-to-end `verl` training logic in this repository, mapped directly to code.
