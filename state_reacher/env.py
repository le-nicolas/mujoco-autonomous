from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ReacherStateEnv(gym.Env):
    """
    Reacher state wrapper with explicit 11D observation and reward:
      reward = -distance_to_target
               - ctrl_cost_weight * sum(action^2)
               - vel_cost_weight * sum(qvel^2)
               + hold_bonus (when within success_threshold)
    """

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 50}

    def __init__(
        self,
        max_episode_steps: int = 1_000,
        env_id: str = "Reacher-v5",
        xml_file: str | None = None,
        render_mode: str | None = None,
        ctrl_cost_weight: float = 0.02,
        vel_cost_weight: float = 0.005,
        success_threshold: float = 0.03,
        success_hold_steps: int = 15,
        hold_bonus: float = 0.05,
        terminate_on_success: bool = False,
        action_smoothing: float = 0.6,
        torque_scale: float = 0.75,
        human_step_delay: float = 0.0,
    ) -> None:
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        if ctrl_cost_weight < 0:
            raise ValueError("ctrl_cost_weight must be >= 0")
        if vel_cost_weight < 0:
            raise ValueError("vel_cost_weight must be >= 0")
        if success_threshold < 0:
            raise ValueError("success_threshold must be >= 0")
        if success_hold_steps <= 0:
            raise ValueError("success_hold_steps must be > 0")
        if hold_bonus < 0:
            raise ValueError("hold_bonus must be >= 0")
        if not (0.0 <= action_smoothing < 1.0):
            raise ValueError("action_smoothing must be in [0.0, 1.0)")
        if torque_scale <= 0:
            raise ValueError("torque_scale must be > 0")
        if human_step_delay < 0:
            raise ValueError("human_step_delay must be >= 0")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode}. "
                f"Expected one of {self.metadata['render_modes']}"
            )
        if env_id not in {"Reacher-v4", "Reacher-v5"}:
            raise ValueError("env_id must be one of {'Reacher-v4', 'Reacher-v5'}")

        default_xml = Path(__file__).resolve().parent.parent / "assets" / "reacher.xml"
        self.xml_file = Path(xml_file) if xml_file is not None else default_xml
        if not self.xml_file.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_file}")

        self.ctrl_cost_weight = ctrl_cost_weight
        self.vel_cost_weight = vel_cost_weight
        self.success_threshold = success_threshold
        self.success_hold_steps = success_hold_steps
        self.hold_bonus = hold_bonus
        self.terminate_on_success = terminate_on_success
        self.action_smoothing = action_smoothing
        self.torque_scale = torque_scale
        self.human_step_delay = human_step_delay
        self.env_id = env_id
        make_kwargs: dict[str, Any] = {
            "max_episode_steps": max_episode_steps,
            "render_mode": render_mode,
        }
        # Gymnasium Reacher-v4 does not accept xml_file, but v5 does.
        if env_id == "Reacher-v5":
            make_kwargs["xml_file"] = str(self.xml_file)
        self._env = gym.make(env_id, **make_kwargs)
        self.action_space = self._env.action_space

        low = np.array(
            [
                -1.0,  # cos(theta1)
                -1.0,  # sin(theta1)
                -1.0,  # cos(theta2)
                -1.0,  # sin(theta2)
                -np.inf,  # theta1_dot
                -np.inf,  # theta2_dot
                -np.inf,  # target_x
                -np.inf,  # target_y
                -np.inf,  # fingertip_x
                -np.inf,  # fingertip_y
                0.0,  # distance
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                1.0,  # cos(theta1)
                1.0,  # sin(theta1)
                1.0,  # cos(theta2)
                1.0,  # sin(theta2)
                np.inf,  # theta1_dot
                np.inf,  # theta2_dot
                np.inf,  # target_x
                np.inf,  # target_y
                np.inf,  # fingertip_x
                np.inf,  # fingertip_y
                np.inf,  # distance
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        unwrapped = self._env.unwrapped
        self._fingertip_body_id = unwrapped.model.body("fingertip").id
        self._target_body_id = unwrapped.model.body("target").id
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._hold_count = 0

    def _build_obs(self) -> tuple[np.ndarray, float]:
        data = self._env.unwrapped.data
        theta1 = float(data.qpos[0])
        theta2 = float(data.qpos[1])
        theta1_dot = float(data.qvel[0])
        theta2_dot = float(data.qvel[1])

        target_xy = data.xpos[self._target_body_id, :2].copy()
        fingertip_xy = data.xpos[self._fingertip_body_id, :2].copy()
        dist = float(np.linalg.norm(fingertip_xy - target_xy))

        obs = np.array(
            [
                np.cos(theta1),
                np.sin(theta1),
                np.cos(theta2),
                np.sin(theta2),
                theta1_dot,
                theta2_dot,
                float(target_xy[0]),
                float(target_xy[1]),
                float(fingertip_xy[0]),
                float(fingertip_xy[1]),
                dist,
            ],
            dtype=np.float32,
        )
        return obs, dist

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._env.reset(seed=seed, options=options)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._hold_count = 0
        obs, dist = self._build_obs()
        info = {"distance_to_target": dist, "hold_count": self._hold_count}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        clipped_action = np.asarray(clipped_action, dtype=np.float32)
        smoothed_action = (
            self.action_smoothing * self._prev_action
            + (1.0 - self.action_smoothing) * clipped_action
        )
        applied_action = np.clip(
            self.torque_scale * smoothed_action,
            self.action_space.low,
            self.action_space.high,
        )
        applied_action = np.asarray(applied_action, dtype=np.float32)
        self._prev_action = smoothed_action

        _, _, terminated, truncated, info = self._env.step(applied_action)
        obs, dist = self._build_obs()
        qvel = self._env.unwrapped.data.qvel[:2]
        reward_ctrl = -self.ctrl_cost_weight * float(np.sum(np.square(applied_action)))
        reward_vel = -self.vel_cost_weight * float(np.sum(np.square(qvel)))
        reward_dist = -dist

        in_target = dist <= self.success_threshold
        if in_target:
            self._hold_count += 1
        else:
            self._hold_count = 0
        reward_hold = self.hold_bonus if in_target else 0.0

        success = self._hold_count >= self.success_hold_steps
        if success and self.terminate_on_success:
            terminated = True

        reward = reward_dist + reward_ctrl + reward_vel + reward_hold
        info = dict(info)
        info["distance_to_target"] = dist
        info["reward_dist"] = reward_dist
        info["reward_ctrl"] = reward_ctrl
        info["reward_vel"] = reward_vel
        info["reward_hold"] = reward_hold
        info["hold_count"] = self._hold_count
        info["success"] = success
        info["applied_action"] = applied_action.copy()
        if self.render_mode == "human" and self.human_step_delay > 0.0:
            time.sleep(self.human_step_delay)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
