import argparse
import json
import os
from pathlib import Path
from typing import Any

import wandb


DEFAULT_WANDB_ENTITY = "leofyfan-east-china-normal-university"
DEFAULT_WANDB_PROJECT = "opsd_verl"
DEFAULT_WANDB_MODE = "online"
DEFAULT_WANDB_API_KEY = "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"


def apply_default_wandb_env() -> None:
    os.environ.setdefault("WANDB_MODE", DEFAULT_WANDB_MODE)
    os.environ.setdefault("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    os.environ.setdefault("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    os.environ.setdefault("WANDB_API_KEY", DEFAULT_WANDB_API_KEY)


def _default_entity(entity: str | None) -> str:
    return entity or os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY


def _default_project(project: str | None) -> str:
    return project or os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Expected list of objects in {path}")
        return data
    raise ValueError(f"Unsupported payload type in {path}: {type(data)!r}")


def log_records(
    records: list[dict[str, Any]],
    *,
    wandb_module=wandb,
    entity: str | None,
    project: str | None,
    run_name: str | None,
    run_id: str | None,
    tags: list[str] | None,
    group: str | None,
    job_type: str,
    config: dict[str, Any] | None,
    step_key: str = "_step",
) -> None:
    resume = "allow" if run_id else None
    run = wandb_module.init(
        entity=_default_entity(entity),
        project=_default_project(project),
        name=run_name,
        id=run_id,
        resume=resume,
        tags=tags,
        group=group,
        job_type=job_type,
        config=config,
    )
    try:
        for record in records:
            payload = dict(record)
            step = payload.pop(step_key, None)
            wandb_module.log(payload, step=step)
    finally:
        run.finish()


def _load_json_arg(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--config-json must decode to an object")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually log records to the existing W&B project.")
    parser.add_argument("--data-file", required=True, help="JSON or JSONL file containing records to log.")
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to current project settings.")
    parser.add_argument("--project", default=None, help="W&B project. Defaults to current project settings.")
    parser.add_argument("--run-name", default=None, help="W&B run name for this manual upload.")
    parser.add_argument("--run-id", default=None, help="Optional W&B run id to resume/append to.")
    parser.add_argument("--group", default=None, help="Optional W&B group.")
    parser.add_argument("--job-type", default="manual_upload", help="W&B job_type for this upload run.")
    parser.add_argument("--tag", action="append", default=None, help="Repeatable W&B tag.")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON object stored in wandb config for this manual upload run.",
    )
    parser.add_argument(
        "--step-key",
        default="_step",
        help="Record key used as wandb step. The key is removed from the logged payload.",
    )
    return parser


def main() -> None:
    apply_default_wandb_env()
    parser = build_arg_parser()
    args = parser.parse_args()
    records = load_records(args.data_file)
    log_records(
        records,
        entity=args.entity,
        project=args.project,
        run_name=args.run_name,
        run_id=args.run_id,
        tags=args.tag,
        group=args.group,
        job_type=args.job_type,
        config=_load_json_arg(args.config_json),
        step_key=args.step_key,
    )


if __name__ == "__main__":
    main()
