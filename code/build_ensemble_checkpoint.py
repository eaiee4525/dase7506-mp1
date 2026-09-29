import argparse
import hashlib
import math
from pathlib import Path

import torch

from ensemble import ProbabilityEnsemble


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--weight-b",
        type=float,
        required=True,
        help="Prediction weight assigned to checkpoint B.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature applied after combining model probabilities.",
    )
    args = parser.parse_args()

    if not 0.0 < args.weight_b < 1.0:
        parser.error("--weight-b must be strictly between 0 and 1.")

    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        parser.error("--temperature must be finite and positive.")

    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")

    checkpoint_a = torch.load(
        args.checkpoint_a,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_b = torch.load(
        args.checkpoint_b,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint_a["protocol"] != checkpoint_b["protocol"]:
        raise ValueError("The checkpoints use different protocols.")

    if checkpoint_a["implementation"] != "student":
        raise ValueError("Checkpoint A must use the student implementation.")

    if checkpoint_b["implementation"] != "student":
        raise ValueError("Checkpoint B must use the student implementation.")

    config_a = checkpoint_a["config"]
    config_b = checkpoint_b["config"]

    if config_a["vocab"] != config_b["vocab"]:
        raise ValueError("The checkpoints use different vocabularies.")

    if config_a["context"] != config_b["context"]:
        raise ValueError("The checkpoints use different contexts.")

    weight_a = 1.0 - args.weight_b

    ensemble_config = {
        "vocab": config_a["vocab"],
        "context": config_a["context"],
        "temperature": args.temperature,
        "components": [
            config_a,
            config_b,
        ],
        "weights": [
            weight_a,
            args.weight_b,
        ],
    }

    model = ProbabilityEnsemble(ensemble_config)
    model.models[0].load_state_dict(
        checkpoint_a["model"],
        strict=True,
    )
    model.models[1].load_state_dict(
        checkpoint_b["model"],
        strict=True,
    )

    output = {
        "protocol": checkpoint_a["protocol"],
        "implementation": "ensemble",
        "config": ensemble_config,
        "model": model.state_dict(),
        "seed": [
            checkpoint_a["seed"],
            checkpoint_b["seed"],
        ],
        "train_tokens": (
            checkpoint_a["train_tokens"]
            + checkpoint_b["train_tokens"]
        ),
        "ensemble": {
            "weight_on_a": weight_a,
            "weight_on_b": args.weight_b,
            "temperature": args.temperature,
            "checkpoint_a": {
                "path": str(args.checkpoint_a),
                "sha256": sha256(args.checkpoint_a),
            },
            "checkpoint_b": {
                "path": str(args.checkpoint_b),
                "sha256": sha256(args.checkpoint_b),
            },
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=False,
    )
    torch.save(output, args.output)

    print(f"Saved: {args.output}")
    print(f"SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()