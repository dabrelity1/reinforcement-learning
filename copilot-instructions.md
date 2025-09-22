# Copilot Instructions

This file helps Copilot understand the repository, constraints, and expectations so it can produce focused, high‑quality PRs.

## Repository overview
- Purpose: Train a vision-based DQN agent to control a fishing minigame (sim or real capture) on Windows.
- Tech: Python 3.10, PyTorch 2.x (CUDA), Gymnasium, OpenCV, Pygame, TensorBoard.
- Key entry points:
  - `train.py` — main training loop, flags, logging, replay.
  - `agents/dqn.py` — DQN variants, loss/targets, optimizer, AMP, compile.
  - `agents/replay_buffer.py` — PER/memmap replay, n-step, (sequence replay TBD).
  - `env/chet_sim_env.py` — fast simulator (OpenCV); `env/sim_env.py` Pygame; `env/fishing_env.py` real screen.
  - `tests/` — imports + tiny train smoke tests.

## Build, run, and tests
- Local setup (Windows):
  1. `python -m venv .venv`
  2. `.\.venv\Scripts\Activate.ps1`
  3. `pip install --upgrade pip`
  4. `pip install -r requirements.txt`
- Tests: `python -m pytest -q`
- Training examples:
  - `python train.py --env chet-sim --config presets/chet_sim_rainbow_fast.json --run-name chet-sim`
  - Use `--help` to explore flags (AMP, prioritized, memmap, compile, etc.)

## Constraints and non-goals
- Preserve Windows compatibility (multiprocessing spawn, no fork-only solutions).
- No large artifacts in git history; ignore `replay_memmap*/` and `tensorboard relatorios/`.
- Do not move augmentations into envs; keep DrQ and preprocessing in the learner.
- Keep existing flags stable; add new flags rather than breaking changes. When changing defaults, do it in presets.
- Keep PRs small and focused; 1 logical change per PR.

## Performance & quality expectations
- Prefer GPU-bound pipelines: channels_last (NHWC), AMP, pinned memory + non_blocking transfers.
- Use torchvision v2 or Kornia for GPU ops (e.g., random crop/shift) inside the training step.
- For compile: expose `--compile-mode` (reduce-overhead|max-autotune|off). Consider optional CUDA Graphs when shapes are static.
- Add micro-benchmarks when changing the training step; report median step time deltas.
- Always ensure `pytest -q` passes locally and in CI.

## Roadmap context (umbrella issue)
Implement the 3-tier plan tracked in the umbrella issue (stability/speed, smarter Q-learners, throughput). Create small PRs that reference that issue and tick specified acceptance criteria. Highlights:
- Tier 1: Double DQN correctness; Polyak; GPU DrQ; on-GPU preprocessing; compile/capture.
- Tier 2: QR-DQN (default) & IQN; Munchausen; R2D2 with sequence replay and PER-by-sequence.
- Tier 3: Ape-X actor/learner; distributed R2D2; optional multi-GPU.

## File-by-file hints
- `agents/dqn.py`
  - Target computation: ensure Double DQN selection via online net, evaluation via target net.
  - Add QR/IQN heads behind flags; keep C51 available.
  - Support Polyak via helper function and flag.
  - Keep channels_last; ensure forward() and inputs use NHWC where expected.
- `train.py`
  - Wire flags: `--target-tau`, `--compile-mode`, `--cuda-graphs` (optional), distributional head selection, Munchausen alpha.
  - Place DrQ and normalization on CUDA tensors inside the training step.
  - Maintain prioritized + memmap options; uint8 storage in replay.
- `agents/replay_buffer.py`
  - Confirm pinned memory usage for host tensors; provide sequence sampling for R2D2 with burn-in masking.

## PR checklist for Copilot
- [ ] Keep the diff small and scoped.
- [ ] Add or update unit tests (targets correctness; shapes/logits; sequence slicing).
- [ ] Update help/README or presets when adding flags.
- [ ] Run `pytest -q`; ensure CI is green.
- [ ] Include a brief before/after step-time metric when touching the training step.

## Security & safety
- Don’t commit secrets or tokens.
- Don’t change real screen-control defaults that could cause unwanted mouse movement without an explicit flag.

## How to ask for help
- If something is ambiguous, post a comment in the PR with the specific question and proposed assumption.
