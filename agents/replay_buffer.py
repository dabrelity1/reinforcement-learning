from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]):
        self.capacity = int(capacity)
        C, H, W = obs_shape
        # Use uint8 to store normalized frames (0..255) to cut RAM ~4x
        self.obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        # Expect obs in float32 [0,1]; store as uint8 [0,255]
        self.obs[self.idx] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)
        self.dones[self.idx] = 1.0 if done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def can_sample(self, batch_size: int) -> bool:
        size = self.capacity if self.full else self.idx
        return size >= batch_size

    def sample(self, batch_size: int):
        size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, size, size=batch_size)
        # Decode uint8 back to float32 [0,1]
        obs = self.obs[idxs].astype(np.float32) / 255.0
        next_obs = self.next_obs[idxs].astype(np.float32) / 255.0
        return (
            obs,
            self.actions[idxs],
            self.rewards[idxs],
            next_obs,
            self.dones[idxs],
        )


class SumTree:
    """Binary sum tree for efficient sampling by cumulative priorities.
    Capacity must be a power of two for indexing simplicity.
    """
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1) == 0), "capacity must be power of two"
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.size = 0
        self.write = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float):
        idx = self.write + self.capacity
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        # propagate up
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def get(self, s: float) -> Tuple[int, float]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al. 2016).

    - alpha controls how much prioritization is used (0 = uniform)
    - beta controls importance sampling correction towards 1 over time
    - priorities initialized to max priority to ensure new samples are drawn
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        C, H, W = obs_shape
        self.obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        # priorities held in a sum tree
        # round capacity up to next power of two for the tree
        cap_pow2 = 1
        while cap_pow2 < capacity:
            cap_pow2 <<= 1
        self.tree = SumTree(cap_pow2)
        self.max_priority = 1.0
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.idx = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.idx] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)
        self.dones[self.idx] = 1.0 if done else 0.0
        # set priority for this index in tree (by write position modulo capacity)
        # Note: tree holds up to cap_pow2 items; we map only first `capacity` valid ones.
        p = (self.max_priority + self.eps) ** self.alpha
        # When wrapping, we overwrite the priority at that data idx; ensure tree has a slot
        # We fill tree sequentially for first `capacity` adds, then overwrite circularly.
        if self.tree.size < self.capacity:
            self.tree.add(p)
        else:
            # update existing leaf for this position
            leaf_idx = (self.idx % self.tree.capacity) + self.tree.capacity
            self.tree.update(leaf_idx, p)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def can_sample(self, batch_size: int) -> bool:
        size = self.capacity if self.full else self.idx
        return size >= batch_size

    def sample(self, batch_size: int, beta: float):
        size = self.capacity if self.full else self.idx
        assert size > 0
        # Draw uniformly spaced segments on [0, total_p)
        total_p = self.tree.total()
        seg = total_p / batch_size
        samples = []
        idxs = []
        ps = []
        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            s = np.random.uniform(a, b)
            idx, p = self.tree.get(s)
            # map idx into valid data range
            data_idx = idx % self.capacity
            idxs.append(data_idx)
            ps.append(p)
            samples.append(data_idx)
        idxs = np.array(idxs, dtype=np.int64)
        ps = np.array(ps, dtype=np.float32)
        # probabilities
        probs = ps / (total_p + 1e-12)
        # importance sampling weights
        beta = float(beta)
        # Use size for min prob normalization to avoid bias when buffer not full
        # min_prob approximated by min over sampled probs to keep it stable
        weights = (probs * size + 1e-12) ** (-beta)
        weights /= (weights.max() + 1e-12)
        # Gather batch
        obs = self.obs[idxs].astype(np.float32) / 255.0
        next_obs = self.next_obs[idxs].astype(np.float32) / 255.0
        batch = (
            obs,
            self.actions[idxs],
            self.rewards[idxs],
            next_obs,
            self.dones[idxs],
        )
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        # Priority p_i = (|delta| + eps)^alpha
        td = np.abs(td_errors).astype(np.float32) + self.eps
        p = np.power(td, self.alpha)
        self.max_priority = max(self.max_priority, float(td.max()))
        for data_idx, pi in zip(idxs, p):
            leaf = (int(data_idx) % self.tree.capacity) + self.tree.capacity
            self.tree.update(leaf, float(pi))


class MemmapReplayBuffer(ReplayBuffer):
    """Replay buffer storing obs/next_obs on disk via numpy.memmap to reduce RAM.

    Notes:
    - Actions, rewards, dones remain in RAM (small arrays) for speed.
    - Memmaps are created under a directory path; files are obs.dat and next_obs.dat.
    - Data layout matches ReplayBuffer: (N, C, H, W) uint8 frames.
    - Safe for single-process use; not multiprocess-safe.
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], dirpath: str):
        capacity = int(capacity)
        C, H, W = obs_shape
        import os
        os.makedirs(dirpath, exist_ok=True)
        self.capacity = capacity
        self._dir = dirpath
        # Disk-backed frames
        # Expected disk footprint ~ 2 * capacity * C * H * W bytes
        self.obs = np.memmap(os.path.join(dirpath, 'obs.dat'), mode='w+', dtype=np.uint8, shape=(capacity, C, H, W))
        self.next_obs = np.memmap(os.path.join(dirpath, 'next_obs.dat'), mode='w+', dtype=np.uint8, shape=(capacity, C, H, W))
        # In-memory small arrays
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False
        # track writes for periodic flush
        self._since_flush = 0
        self._flush_every = max(1024, capacity // 16)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.idx] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)
        self.dones[self.idx] = 1.0 if done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
        # Periodically flush to ensure data hits disk
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            try:
                self.obs.flush()
                self.next_obs.flush()
            except Exception:
                pass
            self._since_flush = 0

    def sample(self, batch_size: int):
        size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, size, size=batch_size)
        obs = np.asarray(self.obs[idxs], dtype=np.float32) / 255.0
        next_obs = np.asarray(self.next_obs[idxs], dtype=np.float32) / 255.0
        return (
            obs,
            self.actions[idxs],
            self.rewards[idxs],
            next_obs,
            self.dones[idxs],
        )


class PrioritizedMemmapReplayBuffer:
    """Prioritized experience replay storing frames on disk via numpy.memmap.

    Combines PER (SumTree + importance sampling) with memmap-backed obs/next_obs
    to minimize RAM usage for large buffers. Actions/rewards/dones and the sum tree
    remain in RAM for speed.

    Disk footprint ~ 2 * capacity * C * H * W bytes.
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], dirpath: str, alpha: float = 0.6, eps: float = 1e-6):
        capacity = int(capacity)
        C, H, W = obs_shape
        import os
        os.makedirs(dirpath, exist_ok=True)
        self.capacity = capacity
        self._dir = dirpath
        # Disk-backed frames
        self.obs = np.memmap(os.path.join(dirpath, 'obs.dat'), mode='w+', dtype=np.uint8, shape=(capacity, C, H, W))
        self.next_obs = np.memmap(os.path.join(dirpath, 'next_obs.dat'), mode='w+', dtype=np.uint8, shape=(capacity, C, H, W))
        # In-memory small arrays
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        # PER structures
        cap_pow2 = 1
        while cap_pow2 < capacity:
            cap_pow2 <<= 1
        self.tree = SumTree(cap_pow2)
        self.max_priority = 1.0
        self.alpha = float(alpha)
        self.eps = float(eps)
        # write pointers
        self.idx = 0
        self.full = False
        # periodic flush control
        self._since_flush = 0
        self._flush_every = max(1024, capacity // 16)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        # store frames as uint8
        self.obs[self.idx] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        self.next_obs[self.idx] = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = 1.0 if done else 0.0
        # PER priority write
        p = (self.max_priority + self.eps) ** self.alpha
        if self.tree.size < self.capacity:
            self.tree.add(p)
        else:
            leaf_idx = (self.idx % self.tree.capacity) + self.tree.capacity
            self.tree.update(leaf_idx, p)
        # advance pointer
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
        # occasional flush so data hits disk
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            try:
                self.obs.flush()
                self.next_obs.flush()
            except Exception:
                pass
            self._since_flush = 0

    def can_sample(self, batch_size: int) -> bool:
        size = self.capacity if self.full else self.idx
        return size >= batch_size

    def sample(self, batch_size: int, beta: float):
        size = self.capacity if self.full else self.idx
        assert size > 0
        total_p = self.tree.total()
        seg = total_p / batch_size
        idxs = []
        ps = []
        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            s = np.random.uniform(a, b)
            idx, p = self.tree.get(s)
            data_idx = idx % self.capacity
            idxs.append(data_idx)
            ps.append(p)
        idxs = np.array(idxs, dtype=np.int64)
        ps = np.array(ps, dtype=np.float32)
        probs = ps / (total_p + 1e-12)
        beta = float(beta)
        weights = (probs * size + 1e-12) ** (-beta)
        weights /= (weights.max() + 1e-12)
        # gather batch from memmaps
        obs = np.asarray(self.obs[idxs], dtype=np.float32) / 255.0
        next_obs = np.asarray(self.next_obs[idxs], dtype=np.float32) / 255.0
        batch = (
            obs,
            self.actions[idxs],
            self.rewards[idxs],
            next_obs,
            self.dones[idxs],
        )
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        td = np.abs(td_errors).astype(np.float32) + self.eps
        p = np.power(td, self.alpha)
        self.max_priority = max(self.max_priority, float(td.max()))
        for data_idx, pi in zip(idxs, p):
            leaf = (int(data_idx) % self.tree.capacity) + self.tree.capacity
            self.tree.update(leaf, float(pi))
