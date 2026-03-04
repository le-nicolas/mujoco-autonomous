# Vision-Based RL in MuJoCo: Reacher from Pixels

This project trains a 2-joint MuJoCo Reacher agent using **only image input** (no simulator state).
The policy receives rendered frames and outputs the 2 continuous joint torques.

This repo also includes a separate **state-based Reacher SAC** baseline.

## What is included

- `vision_reacher/env.py`: Pixel-only wrapper for `Reacher-v5`
- `vision_reacher/models.py`: Small CNN image encoder
- `train.py`: SAC training with explicit encoder + MLP policy/critic heads
- `eval.py`: Run a trained checkpoint and optionally render it
- `assets/reacher.xml`: Local MuJoCo model file (editable)

## 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Train

```bash
python train.py --total-timesteps 1000000 --run-name reacher_pixels
```

By default this now does everything in one run:

- trains SAC from pixels
- shows live MuJoCo arm movement during training
- runs final deterministic playback episodes after training

Useful options:

- `--image-size 64` (default)
- `--frame-stack 3` (default)
- `--grayscale` (default) / `--no-grayscale` for RGB
- `--max-episode-steps 1000`
- `--xml-file assets/reacher.xml` (default local model file)
- `--no-live-view` to train without opening the viewer
- `--final-play-episodes 0` to skip post-training playback
- defaults target real pixel learning runs (`--total-timesteps 1000000`, `--learning-starts 10000`, `--eval-freq 10000`)
- `--progress-bar`

Artifacts are saved in `runs/<run-name>/`:

- `final_model.zip`
- `best_model/best_model.zip`
- TensorBoard logs in `tb/`

## 3) Evaluate

```bash
python eval.py --model-path runs/reacher_pixels/best_model/best_model.zip --episodes 10 --render --deterministic
```

Create a GIF:

```bash
python eval.py --model-path runs/reacher_pixels/best_model/best_model.zip --episodes 1 --no-render --save-gif runs/reacher_pixels/policy.gif
```

Create reward curve plot:

```bash
python report_pixel.py --run-dir runs/reacher_pixels
```

## State Baseline (Custom 11D State)

State observation (11D):

- `cos(theta1), sin(theta1), cos(theta2), sin(theta2)`
- `theta1_dot, theta2_dot`
- `target_x, target_y`
- `fingertip_x, fingertip_y`
- `distance_to_target`

Action:

- 2 continuous torques, clipped to `[-1, 1]`

Reward:

- `reward = -distance_to_target`
- `- 0.02 * (torque1^2 + torque2^2)` (control penalty)
- `- 0.005 * (theta1_dot^2 + theta2_dot^2)` (velocity penalty)
- `+ 0.05` while inside the target radius (hold bonus)

Train:

```bash
python train_state.py --total-timesteps 500000 --run-name reacher_state
```

By default this now does everything in one run:

- trains SAC
- shows live MuJoCo arm movement during training
- runs final deterministic playback episodes after training

Useful flags:

- `--no-live-view` to train without opening the viewer
- `--final-play-episodes 0` to skip post-training playback
- `--live-step-delay 0.02` to slow viewer updates so motion is visible
- `--torque-scale 0.75` and `--action-smoothing 0.6` for less jerky control
- defaults are tuned for live watching (`--learning-starts 2000`, `--eval-freq 2000`)

Evaluate:

```bash
python eval_state.py --model-path runs_state/reacher_state/best_model/best_model.zip --episodes 10 --render --deterministic
```

Version control:

- default is `--env-id Reacher-v5` (supports local `--xml-file assets/reacher.xml`)
- optional `--env-id Reacher-v4` (Gymnasium build here does not accept `xml_file`)

## Notes

- This uses `Reacher-v5` from Gymnasium MuJoCo.
- Environment is loaded from local XML at `assets/reacher.xml` by default.
- Observation is preprocessed from render frames to `(C x H x W)` `uint8`.
- Default setup is `64x64x3` pixels, matching a simple "from pixels" setup.
- Model path is: image -> CNN encoder -> latent vector -> MLP actor/critic -> 2 torques.
- SAC is used because it is a strong baseline for continuous pixel control.
- State baseline exposes the 11D state layout explicitly, independent of MuJoCo default obs.
