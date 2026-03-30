import argparse
import importlib
from pathlib import Path


def import_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the local OPSD + verl uv environment.")
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/Qwen3-4B",
        help="Local model path used for Qwen3-4B reproduction.",
    )
    parser.add_argument(
        "--dataset",
        default="siyanzhao/Openthoughts_math_30k_opsd",
        help="Dataset to probe with a lightweight split.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip the Hugging Face dataset probe.",
    )
    args = parser.parse_args()

    modules = [
        "torch",
        "transformers",
        "trl",
        "datasets",
        "accelerate",
        "peft",
        "deepspeed",
        "bitsandbytes",
        "vllm",
        "verl",
        "flash_attn",
    ]

    print("Package versions")
    for name in modules:
        print(f"  {name}: {import_version(name)}")

    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU[0]: {torch.cuda.get_device_name(0)}")
        print(f"torch.version.cuda: {torch.version.cuda}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loaded config: {config.__class__.__name__}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    if not args.skip_dataset:
        sample = load_dataset(args.dataset, split="train[:1]")
        print(f"Loaded dataset sample: {len(sample)} row")
        print(f"Dataset columns: {sample.column_names}")


if __name__ == "__main__":
    main()
