from __future__ import annotations

import argparse
import time

from stable_baselines3 import SAC

from vision_reacher import VisionReacherEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Reacher in native MuJoCo viewer.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--xml-file", type=str, default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = VisionReacherEnv(
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        grayscale=args.grayscale,
        max_episode_steps=args.max_episode_steps,
        xml_file=args.xml_file,
        render_mode="human",
    )

    model = SAC.load(args.model_path, env=env) if args.model_path else None
    obs, _ = env.reset(seed=args.seed)
    dt = 1.0 / args.fps if args.fps > 0 else 0.0
    end_t = time.time() + args.seconds
    steps = 0
    episodes = 0

    try:
        while time.time() < end_t:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset()
            if dt > 0:
                time.sleep(dt)
    finally:
        env.close()

    print(f"played_steps={steps}, episodes={episodes}")


if __name__ == "__main__":
    main()
