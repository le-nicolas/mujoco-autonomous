from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from state_reacher import ReacherStateEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on state-based Reacher (custom 11D state)."
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
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
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--policy-hidden-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="reacher_state_sac")
    parser.add_argument("--log-dir", type=str, default="runs_state")
    parser.add_argument(
        "--live-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show MuJoCo viewer during training.",
    )
    parser.add_argument(
        "--final-play-episodes",
        type=int,
        default=3,
        help="Deterministic playback episodes after training (0 to disable).",
    )
    parser.add_argument(
        "--live-step-delay",
        type=float,
        default=0.02,
        help="Delay per env step in human view (seconds).",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def make_env(seed: int, args: argparse.Namespace, render_mode: str | None) -> Monitor:
    human_step_delay = args.live_step_delay if render_mode == "human" else 0.0
    env = ReacherStateEnv(
        max_episode_steps=args.max_episode_steps,
        env_id=args.env_id,
        xml_file=args.xml_file,
        render_mode=render_mode,
        ctrl_cost_weight=args.ctrl_cost_weight,
        vel_cost_weight=args.vel_cost_weight,
        success_threshold=args.success_threshold,
        success_hold_steps=args.success_hold_steps,
        hold_bonus=args.hold_bonus,
        terminate_on_success=args.terminate_on_success,
        action_smoothing=args.action_smoothing,
        torque_scale=args.torque_scale,
        human_step_delay=human_step_delay,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def run_final_playback(model_path: Path, args: argparse.Namespace) -> None:
    print(f"Starting final playback from model: {model_path}")
    env = ReacherStateEnv(
        max_episode_steps=args.max_episode_steps,
        env_id=args.env_id,
        xml_file=args.xml_file,
        render_mode="human",
        ctrl_cost_weight=args.ctrl_cost_weight,
        vel_cost_weight=args.vel_cost_weight,
        success_threshold=args.success_threshold,
        success_hold_steps=args.success_hold_steps,
        hold_bonus=args.hold_bonus,
        terminate_on_success=args.terminate_on_success,
        action_smoothing=args.action_smoothing,
        torque_scale=args.torque_scale,
        human_step_delay=args.live_step_delay,
    )
    model = SAC.load(str(model_path), env=env, device=args.device)
    try:
        for episode in range(args.final_play_episodes):
            obs, info = env.reset(seed=args.seed + 20_000 + episode)
            done = False
            ep_reward = 0.0
            final_dist = float(info.get("distance_to_target", 0.0))
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                final_dist = float(info.get("distance_to_target", final_dist))
            print(
                f"Playback episode {episode + 1}: "
                f"reward={ep_reward:.3f}, final_dist={final_dist:.4f}"
            )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_render_mode = "human" if args.live_view else None
    train_env = make_env(args.seed, args, render_mode=train_render_mode)
    eval_env = make_env(args.seed + 10_000, args, render_mode=None)
    tensorboard_log = (
        str(run_dir / "tb") if importlib.util.find_spec("tensorboard") is not None else None
    )
    if tensorboard_log is None:
        print("tensorboard not installed; disabling tensorboard logging.")

    net_arch = [args.policy_hidden_dim] * args.policy_hidden_layers
    policy_kwargs = {"net_arch": net_arch}

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=args.progress_bar,
    )
    model.save(str(run_dir / "final_model"))

    train_env.close()
    eval_env.close()

    best_model_path = run_dir / "best_model" / "best_model.zip"
    final_model_path = run_dir / "final_model.zip"
    playback_model = best_model_path if best_model_path.exists() else final_model_path
    if args.final_play_episodes > 0:
        run_final_playback(playback_model, args)


if __name__ == "__main__":
    main()
