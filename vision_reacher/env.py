from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class VisionReacherEnv(gym.Env):
    """
    Pixel-only wrapper around MuJoCo Reacher.

    The policy receives stacked preprocessed frames instead of the simulator state.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        image_size: int = 64,
        frame_stack: int = 1,
        grayscale: bool = False,
        max_episode_steps: int = 1_000,
        xml_file: str | None = None,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode}. "
                f"Expected one of {self.metadata['render_modes']}"
            )
        if image_size <= 0:
            raise ValueError("image_size must be > 0")
        if frame_stack <= 0:
            raise ValueError("frame_stack must be > 0")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")

        self.image_size = image_size
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        self.render_mode = render_mode
        default_xml = Path(__file__).resolve().parent.parent / "assets" / "reacher.xml"
        self.xml_file = Path(xml_file) if xml_file is not None else default_xml
        if not self.xml_file.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_file}")

        # Use the requested render mode so "human" uses MuJoCo's native viewer.
        self._env = gym.make(
            "Reacher-v5",
            xml_file=str(self.xml_file),
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        self.action_space = self._env.action_space
        self._model = self._env.unwrapped.model
        self._data = self._env.unwrapped.data
        self._pixel_renderer = mujoco.Renderer(
            self._model, width=self.image_size, height=self.image_size
        )

        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels * frame_stack, image_size, image_size),
            dtype=np.uint8,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    def _get_frame(self) -> np.ndarray:
        # Offscreen renderer is independent of viewer mode, so this works for both
        # rgb_array and native human viewer rendering.
        self._pixel_renderer.update_scene(self._data)
        frame = self._pixel_renderer.render()
        if frame is None:
            raise RuntimeError("MuJoCo offscreen renderer returned None.")
        return frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame,
            dsize=(self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        if self.grayscale:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            return gray[np.newaxis, ...].astype(np.uint8)
        return np.transpose(resized, (2, 0, 1)).astype(np.uint8)

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def _maybe_show_human(self) -> None:
        if self.render_mode != "human":
            return
        # This uses MuJoCo's native viewer window.
        self._env.render()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self._env.reset(seed=seed, options=options)
        frame = self._get_frame()
        processed = self._preprocess_frame(frame)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(processed)
        self._maybe_show_human()
        return self._stacked_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self._env.step(action)
        frame = self._get_frame()
        self._frames.append(self._preprocess_frame(frame))
        self._maybe_show_human()
        return self._stacked_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        self._maybe_show_human()
        return self._get_frame()

    def close(self) -> None:
        self._pixel_renderer.close()
        self._env.close()
