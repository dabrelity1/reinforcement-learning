# Reinforcement Learning DQN Agent

A PyTorch-based DQN (Deep Q-Network) implementation for training agents on fishing minigame environments, with support for simulation, OpenCV visualization, and real screen capture on Windows.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
**Windows 10/11 Required** - This project requires Windows with screen capture and mouse control permissions for real environment training.

- Bootstrap dependencies:
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1` (Windows PowerShell)
  - `source .venv/bin/activate` (Linux/macOS for simulation only)
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt` -- **NEVER CANCEL**: Takes 15-45 minutes due to PyTorch CUDA packages. Set timeout to 60+ minutes.

### Build and Test Commands
- **NEVER CANCEL**: All PyTorch operations and training may take extended time
- Test imports and basic functionality: `python -m pytest -q` -- takes 2-5 minutes. Set timeout to 10+ minutes.
- **CRITICAL**: If pytest fails due to missing dependencies, ensure virtual environment is activated and all requirements are installed

### Training Commands (Validated)
- Basic simulation training: `python train.py --env sim --total-steps 200000 --run-name sim-baseline` -- **NEVER CANCEL**: Takes 30-90 minutes depending on hardware. Set timeout to 120+ minutes.
- Fast OpenCV simulator: `python train.py --env chet-sim --total-steps 200000 --render-every 1000 --run-name chet-sim` -- **NEVER CANCEL**: Takes 20-60 minutes. Set timeout to 90+ minutes.
- Rainbow DQN preset: `python train.py --config presets/chet_sim_rainbow_fast.json --run-name chet-sim-rainbow` -- **NEVER CANCEL**: Takes 45-120 minutes. Set timeout to 150+ minutes.
- Real screen capture (Windows only): `python train.py --env real --capture-rect 800 400 300 300 --total-steps 300000 --frame-skip 4 --run-name real-dqn --safe-mode` -- **NEVER CANCEL**: Takes 60-180 minutes. Set timeout to 240+ minutes.

### Evaluation and Analysis
- Evaluate trained model: `python evaluate.py --model-path models/dqn_latest.pt --episodes 5 --render` -- takes 1-5 minutes
- Hyperparameter sweep: `python sweep.py --grid '{"lr": [0.0001, 0.0003], "batch_size": [64, 128]}' --config presets/base.json` -- **NEVER CANCEL**: Can take several hours. Set timeout to 480+ minutes.

## Validation Scenarios

**CRITICAL**: After making code changes, always run these validation steps:

1. **Import Test**: `python -m pytest tests/test_imports.py -v` -- ensures all modules can be imported
2. **Training Smoke Test**: `python -m pytest tests/test_tiny_train.py -v` -- **NEVER CANCEL**: Takes 5-15 minutes for mini training loop. Set timeout to 30+ minutes.
3. **Manual Training Validation**: Run a short training session:
   ```bash
   python train.py --env chet-sim --total-steps 1000 --batch-size 32 --eval-every 0 --save-every 0 --render-every 0
   ```
   -- **NEVER CANCEL**: Takes 2-10 minutes. Set timeout to 20+ minutes.
4. **Verify TensorBoard Logging**: Check that `tensorboard relatorios/` directory contains logs after training
5. **GPU Functionality**: If CUDA available, verify GPU usage with: `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')""`

## Repository Structure

### Key Entry Points
- `train.py` - Main training script (941 lines), supports extensive configuration via CLI flags and JSON presets
- `evaluate.py` - Model evaluation and testing
- `agents/dqn.py` - DQN agent implementation with Rainbow features (Dueling, Noisy, C51, QR-DQN, etc.)
- `agents/replay_buffer.py` - Experience replay with Prioritized Experience Replay (PER) and memory mapping support
- `utils/` - Preprocessing, screen capture, logging, and scheduling utilities

### Environment Types  
- `sim` - Pygame-based local simulation (env/sim_env.py)
- `chet-sim` - OpenCV-based fast simulator for visualization (env/chet_sim_env.py) 
- `real` - Windows screen capture environment (env/fishing_env.py)

### Configuration Presets
- `presets/chet_sim_rainbow_fast.json` - Rainbow DQN with all advanced features
- `presets/chet_sim_qr_fast.json` - QR-DQN variant
- All presets include optimized hyperparameters for fast convergence

## Build Timing Expectations
- **Virtual environment setup**: 2-5 minutes
- **Dependencies installation**: 15-45 minutes (PyTorch CUDA packages are large)
- **Import tests**: 30 seconds - 2 minutes  
- **Training smoke test**: 5-15 minutes
- **Short training (1K steps)**: 2-10 minutes
- **Medium training (200K steps)**: 30-90 minutes
- **Long training (300K+ steps)**: 60-180 minutes
- **Full hyperparameter sweep**: 4-12 hours

## Advanced Features and Flags

### Performance Optimization
- `--amp` - Automatic Mixed Precision for faster training on modern GPUs
- `--compile` - PyTorch 2.0 compile mode for additional speedup
- `--async-learner` - Asynchronous learning thread for improved throughput
- `--num-envs 4` - Parallel environments using AsyncVectorEnv

### DQN Variants
- `--dueling` - Dueling DQN architecture
- `--noisy` - Noisy networks for exploration
- `--c51` - Categorical DQN (C51) for distributional RL
- `--qr-dqn` - Quantile Regression DQN (alternative to C51)
- `--prioritized` - Prioritized Experience Replay

### Training Configuration
- `--n-step 3` - Multi-step returns
- `--target-update-every 2000` - Target network update frequency
- `--lr-schedule cosine` - Learning rate scheduling
- `--curriculum-v2` - Adaptive curriculum learning

## Safety and Constraints

### Windows Compatibility
- Preserve multiprocessing spawn compatibility (no fork-only solutions)
- Use `--safe-mode` for real environment to limit mouse movement during development

### Version Control
- Never commit large artifacts: `replay_memmap*/` and `tensorboard relatorios/` are gitignored
- Keep PRs small and focused on single logical changes
- Always ensure `pytest -q` passes before committing

### GPU Requirements
- CUDA 12.4+ recommended for full PyTorch functionality
- CPU-only mode available but significantly slower
- Use Docker for containerized CUDA training: `docker build -t rl-trainer . && docker run --gpus all -it rl-trainer`

## Troubleshooting

### Common Issues
- **Import errors**: Ensure virtual environment is activated and all dependencies installed
- **PyTorch CUDA issues**: Verify CUDA version compatibility with PyTorch installation
- **Screen capture failures**: On Windows, ensure accessibility permissions are granted
- **Memory errors**: Reduce `--batch-size` or `--buffer-size` for limited memory systems
- **Training hangs**: Check for infinite loops in environment step functions

### Performance Issues
- Use `--amp` and `--compile` for modern GPUs
- Enable `--async-learner` for CPU-bound scenarios
- Monitor GPU utilization with `nvidia-smi` during training
- Consider memory mapping with `--memmap` for large replay buffers

## CI/CD Integration

The repository includes GitHub Actions CI that:
- Tests on Windows (required platform)
- Runs `python -m pytest -q`  
- Validates all imports work correctly
- **NEVER CANCEL CI builds** - They may take 15-30 minutes to complete

Always run local tests before pushing:
1. `python -m pytest -q`
2. Test at least one training command manually
3. Verify no breaking changes to existing flag behavior

## Docker Support (Optional)

For containerized training with CUDA:
```bash
docker build -t rl-trainer .
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  rl-trainer python train.py --config presets/chet_sim_rainbow.json
```

**NEVER CANCEL** Docker builds - CUDA base images are large and take 10-30 minutes to download and build.

## Additional Tools

### ChetBot Automation
- `chet-bot/` contains a separate visual automation system
- Run with: `python -m fishbot.main --dry-run --debug`
- Includes calibration tools and minigame simulator

### Experiment Management
- TensorBoard logs automatically saved to `tensorboard relatorios/`
- Use `sweep.py` for hyperparameter grid searches
- Models automatically saved to `models/` directory

Always follow the imperative instructions above and validate each command works before relying on it for development workflows.