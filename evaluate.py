from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro

from agents.dqn import DQNAgent, DQNConfig


def make_env(env: str, capture_rect: Optional[list[int]] = None):
    if env == "sim":
        from env.sim_env import FishingSimEnv, SimConfig
        return FishingSimEnv(SimConfig())
    elif env == "chet-sim":
        from env.chet_sim_env import ChetMiniGameEnv, ChetSimConfig
        return ChetMiniGameEnv(ChetSimConfig())
    elif env == "real":
        from env.fishing_env import FishingEnv, RealEnvConfig
        from utils.screen import CaptureRect
        assert capture_rect is not None and len(capture_rect) == 4, "Provide --capture-rect x y w h"
        rect = CaptureRect(capture_rect[0], capture_rect[1], capture_rect[2], capture_rect[3])
        return FishingEnv(RealEnvConfig(capture_rect=rect))
    else:
        raise ValueError("env must be 'sim' or 'real'")


@dataclass
class Args:
    env: str = "sim"
    capture_rect: Optional[list[int]] = None
    model_path: str = "models/dqn_latest.pt"
    episodes: int = 5
    render: bool = True


def main(args: Args):
    env = make_env(args.env, args.capture_rect)
    obs, _ = env.reset()
    obs_shape = obs.shape
    # Auto-detect architecture from checkpoint if available
    if os.path.exists(args.model_path):
        try:
            map_loc = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt = torch.load(args.model_path, map_location=map_loc)
            cfg_dict = ckpt.get('config', {})
            # Ensure essential fields match current env
            cfg_dict['in_channels'] = int(obs_shape[0])
            try:
                cfg_dict['num_actions'] = int(getattr(env.action_space, 'n', cfg_dict.get('num_actions', 3)))
            except Exception:
                cfg_dict['num_actions'] = 3
            # Prefer current device availability
            cfg_dict['device'] = map_loc
            agent = DQNAgent(DQNConfig(**cfg_dict))
            agent.load(args.model_path, map_location=map_loc)
        except Exception as e:
            print(f"Warning: failed to load model/config from {args.model_path}: {e}. Falling back to default agent.")
            agent = DQNAgent(DQNConfig(in_channels=obs_shape[0], num_actions=3))
    else:
        print(f"Warning: model not found: {args.model_path}. Running with random weights.")
        agent = DQNAgent(DQNConfig(in_channels=obs_shape[0], num_actions=3))

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += r
            steps += 1
            if args.render and hasattr(env, 'render'):
                env.render()
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, steps={steps}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
