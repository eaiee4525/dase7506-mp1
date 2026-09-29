import argparse
import hashlib
from pathlib import Path

import torch


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
        help="Weight assigned to checkpoint B; checkpoint A receives 1-weight-b.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shared-ancestry",
        action="store_true",
        help=(
            "Use the maximum train-token count when checkpoints "
            "come from the same training run."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.weight_b <= 1.0:
        parser.error("--weight-b must be between 0 and 1.")

    if args.output.exists():
        parser.error(f"Refusing to overwrite existing file: {args.output}")

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

    if checkpoint_a["implementation"] != checkpoint_b["implementation"]:
        raise ValueError("The checkpoints use different implementations.")

    if checkpoint_a["model"].keys() != checkpoint_b["model"].keys():
        raise ValueError("The checkpoints have different parameter names.")

    averaged_state = {}

    for name, tensor_a in checkpoint_a["model"].items():
        tensor_b = checkpoint_b["model"][name]

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shape mismatch for {name}.")

        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError(f"Dtype mismatch for {name}.")

        if tensor_a.is_floating_point() or tensor_a.is_complex():
            averaged_state[name] = tensor_a.mul(
                1.0 - args.weight_b
            ).add(
                tensor_b,
                alpha=args.weight_b,
            )
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(f"Non-floating tensor differs: {name}")
            averaged_state[name] = tensor_b.clone()

    if args.shared_ancestry:
        train_tokens = max(
            checkpoint_a["train_tokens"],
            checkpoint_b["train_tokens"],
        )
    else:
        train_tokens = (
            checkpoint_a["train_tokens"]
            + checkpoint_b["train_tokens"]
        )
    output = {
        "protocol": checkpoint_b["protocol"],
        "implementation": checkpoint_b["implementation"],
        "config": checkpoint_b["config"],
        "model": averaged_state,
        "seed": checkpoint_b["seed"],
        "train_tokens": train_tokens,
        "averaging": {
            "weight_on_a": 1.0 - args.weight_b,
            "weight_on_b": args.weight_b,
            "checkpoint_a": {
                "path": str(args.checkpoint_a),
                "sha256": sha256(args.checkpoint_a),
            },
            "checkpoint_b": {
                "path": str(args.checkpoint_b),
                "sha256": sha256(args.checkpoint_b),
            },
            "shared_ancestry": args.shared_ancestry,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(output, args.output)

    print(f"Saved: {args.output}")
    print(f"SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()