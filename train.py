from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from vision_reacher import ReacherPixelEncoder, VisionReacherEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on pixel-only MuJoCo Reacher."
    )
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--xml-file", type=str, default=None)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--policy-hidden-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="vision_reacher_sac")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def make_env(
    seed: int,
    image_size: int,
    frame_stack: int,
    grayscale: bool,
    max_episode_steps: int,
    xml_file: str | None,
):
    def _init():
        env = VisionReacherEnv(
            image_size=image_size,
            frame_stack=frame_stack,
            grayscale=grayscale,
            max_episode_steps=max_episode_steps,
            xml_file=xml_file,
            render_mode="rgb_array",
        )
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    args = parse_args()
    grayscale = args.grayscale

    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = (
        str(run_dir / "tb") if importlib.util.find_spec("tensorboard") is not None else None
    )
    if tensorboard_log is None:
        print("tensorboard not installed; disabling tensorboard logging.")

    env_fns = [
        make_env(
            seed=args.seed + idx,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            grayscale=grayscale,
            max_episode_steps=args.max_episode_steps,
            xml_file=args.xml_file,
        )
        for idx in range(args.num_envs)
    ]
    train_env = VecMonitor(DummyVecEnv(env_fns))

    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_env(
                    seed=args.seed + 10_000,
                    image_size=args.image_size,
                    frame_stack=args.frame_stack,
                    grayscale=grayscale,
                    max_episode_steps=args.max_episode_steps,
                    xml_file=args.xml_file,
                )
            ]
        )
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // max(args.num_envs, 1), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    net_arch = [args.policy_hidden_dim] * args.policy_hidden_layers
    policy_kwargs = dict(
        features_extractor_class=ReacherPixelEncoder,
        features_extractor_kwargs={"features_dim": args.features_dim},
        net_arch=net_arch,
    )

    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=args.progress_bar,
    )
    model.save(str(run_dir / "final_model"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
