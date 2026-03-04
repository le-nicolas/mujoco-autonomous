from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import SAC

from vision_reacher import VisionReacherEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pixel Reacher SAC.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
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
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--gif-episode", type=int, default=1)
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument("--max-gif-steps", type=int, default=1_000)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grayscale = args.grayscale

    env = VisionReacherEnv(
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        grayscale=grayscale,
        max_episode_steps=args.max_episode_steps,
        xml_file=args.xml_file,
        render_mode="human" if args.render else "rgb_array",
    )
    model_path = Path(args.model_path)
    model = SAC.load(str(model_path), env=env, device=args.device)

    rewards = []
    gif_frames: list = []
    gif_episode_index = max(args.gif_episode - 1, 0)
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        should_capture = args.save_gif is not None and episode == gif_episode_index
        if should_capture and len(gif_frames) == 0:
            gif_frames.append(env.render())
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if should_capture and len(gif_frames) < args.max_gif_steps:
                gif_frames.append(env.render())
        rewards.append(ep_reward)
        print(f"Episode {episode + 1}: reward={ep_reward:.3f}")

    mean_reward = sum(rewards) / len(rewards)
    print(f"Mean reward over {args.episodes} episodes: {mean_reward:.3f}")
    if args.save_gif is not None:
        gif_path = Path(args.save_gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        if len(gif_frames) == 0:
            print("GIF was requested but no frames were captured.")
        else:
            imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps)
            print(f"Saved GIF: {gif_path}")
    env.close()


if __name__ == "__main__":
    main()
