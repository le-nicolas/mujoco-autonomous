from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window > len(values):
        window = len(values)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reward curve plot from pixel Reacher eval logs."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--eval-log", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--smooth-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_log = (
        Path(args.eval_log)
        if args.eval_log is not None
        else run_dir / "eval_logs" / "evaluations.npz"
    )
    if not eval_log.exists():
        raise FileNotFoundError(f"Evaluation log not found: {eval_log}")

    output_plot = (
        Path(args.output_plot)
        if args.output_plot is not None
        else run_dir / "reward_curve.png"
    )
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(eval_log)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    smoothed = moving_average(mean_rewards, args.smooth_window)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label="Mean Eval Reward", alpha=0.45)
    plt.plot(timesteps, smoothed, label="Smoothed", linewidth=2.0)
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="Std Dev",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Pixel Reacher SAC Evaluation Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()

    print(f"Saved reward curve: {output_plot}")
    print(
        "Final eval reward: "
        f"mean={mean_rewards[-1]:.3f}, std={std_rewards[-1]:.3f}, "
        f"timestep={int(timesteps[-1])}"
    )


if __name__ == "__main__":
    main()
