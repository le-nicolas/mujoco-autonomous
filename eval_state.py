from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from state_reacher import ReacherStateEnv


def resolve_model_path(explicit_model_path: str | None) -> Path:
    if explicit_model_path:
        model_path = Path(explicit_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    project_root = Path(__file__).resolve().parent
    search_roots = [project_root / "runs_state", project_root / "runs_state_smoke"]
    patterns = ["**/best_model/best_model.zip", "**/final_model.zip"]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "No trained model found. Run `python train_state.py` first, "
            "or pass --model-path."
        )

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    print(f"Auto-selected model: {latest}")
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate state-based Reacher SAC.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-id",
        type=str,
        default="Reacher-v5",
        choices=["Reacher-v4", "Reacher-v5"],
    )
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--ctrl-cost-weight", type=float, default=0.02)
    parser.add_argument("--vel-cost-weight", type=float, default=0.005)
    parser.add_argument("--success-threshold", type=float, default=0.03)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--hold-bonus", type=float, default=0.05)
    parser.add_argument("--terminate-on-success", action="store_true")
    parser.add_argument("--action-smoothing", type=float, default=0.6)
    parser.add_argument("--torque-scale", type=float, default=0.75)
    parser.add_argument("--xml-file", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--step-delay", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    env = ReacherStateEnv(
        max_episode_steps=args.max_episode_steps,
        env_id=args.env_id,
        xml_file=args.xml_file,
        render_mode="human" if args.render else None,
        ctrl_cost_weight=args.ctrl_cost_weight,
        vel_cost_weight=args.vel_cost_weight,
        success_threshold=args.success_threshold,
        success_hold_steps=args.success_hold_steps,
        hold_bonus=args.hold_bonus,
        terminate_on_success=args.terminate_on_success,
        action_smoothing=args.action_smoothing,
        torque_scale=args.torque_scale,
        human_step_delay=args.step_delay if args.render else 0.0,
    )
    model = SAC.load(str(model_path), env=env, device=args.device)

    rewards = []
    final_dists = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        final_dist = float(info.get("distance_to_target", 0.0))
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            final_dist = float(info.get("distance_to_target", final_dist))
        rewards.append(ep_reward)
        final_dists.append(final_dist)
        print(
            f"Episode {episode + 1}: reward={ep_reward:.3f}, "
            f"final_dist={final_dist:.4f}"
        )

    mean_reward = sum(rewards) / len(rewards)
    mean_final_dist = sum(final_dists) / len(final_dists)
    print(f"Mean reward over {args.episodes} episodes: {mean_reward:.3f}")
    print(f"Mean final distance over {args.episodes} episodes: {mean_final_dist:.4f}")
    env.close()


if __name__ == "__main__":
    main()
