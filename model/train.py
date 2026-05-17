"""Training entry point — supervised teacher-forced training of the graph generator
on (spec, reference_graph) pairs."""
from __future__ import annotations


def train(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    train(sys.argv[1] if len(sys.argv) > 1 else "experiments/000_baseline/config.yaml")
