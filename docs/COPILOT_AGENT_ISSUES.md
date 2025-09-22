# Copilot Agent Backlog

This backlog is structured for small, testable PRs. Each item includes rationale and acceptance criteria.

## Tier 1 — Stability & Speed

1. Double DQN audit and fix
   - Why: Reduce overestimation bias.
   - Done when:
     - Action selection uses online net; evaluation uses target net.
     - Unit test validates argmax is from online net.
     - `pytest -q` passes.

2. Soft target updates (Polyak)
   - Why: Smoother, more stable training.
   - Done when:
     - Add `--target-tau` float flag (default 0.005) and `--target-update-interval` fallback.
     - Implement `polyak_update(target, online, tau)` gated by flag.
     - Tests updated; training smoke passes.

3. GPU-native DrQ augmentations
   - Why: Keep pipeline GPU-bound; reduce CPU overhead.
   - Done when:
     - DrQ crops/shifts run on CUDA tensors (Kornia/torchvision).
     - Batch augmentation benchmark shows lower step time.

4. Replay pipeline hygiene
   - Why: Remove hidden host stalls.
   - Done when:
     - Ensure pinned memory + non_blocking=True on all CPU→GPU copies.
     - Keep frames uint8 in replay; normalize on-GPU.

## Tier 2 — Smarter Q-Learners

5. QR-DQN head (default)
   - Why: Better distributional approximation than C51 in many cases.
   - Done when:
     - Config switch between C51 and QR-DQN; default to QR in best preset.
     - Unit tests for shape/logits; training smoke passes.

6. IQN head option
   - Why: SOTA distributional performance at similar cost.
   - Done when:
     - Add IQN head; wire config flags; add simple test.

7. Munchausen DQN knob
   - Why: Improved exploration/stability.
   - Done when:
     - Add `--munchausen-alpha` and clamp; integrate into target computation.
     - Guarded by flag; tests updated.

## Tier 3 — Scale Throughput

8. Ape-X style actors + learner
   - Why: 2–10x wall-clock environment throughput.
   - Done when:
     - Separate actor process feeding PER server; single GPU learner consumes.
     - Windows-friendly IPC (multiprocessing, ZeroMQ, or Redis) with simple launcher script.

9. R2D2 sequences
   - Why: Handle partial observability.
   - Done when:
     - Add GRU/LSTM head; sequence replay with burn-in.
     - PER by sequence; tests for sequence slicing.

---

Notes:
- Keep PRs small and focused (1 item per PR where possible).
- Respect existing flags; avoid breaking changes.
- Ensure `pytest -q` passes; CI enforces it.
