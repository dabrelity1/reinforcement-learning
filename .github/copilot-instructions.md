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

## Common Development Workflows

### Quick Start Development Cycle
1. **Setup (first time only)**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   pip install --upgrade pip
   pip install -r requirements.txt  # NEVER CANCEL: 15-45 minutes
   ```

2. **Code Change Validation** (run after every change):
   ```bash
   python -m pytest -q                    # NEVER CANCEL: 2-10 minutes
   python train.py --env chet-sim --total-steps 1000 --batch-size 32 --eval-every 0 --save-every 0 --render-every 0
   ```

3. **Development Testing**:
   ```bash
   # Quick functionality test (2-5 minutes)
   python train.py --env chet-sim --total-steps 5000 --run-name dev-test
   
   # Medium validation (15-30 minutes)  
   python train.py --config presets/chet_sim_rainbow_fast.json --total-steps 50000 --run-name validation
   ```

4. **Production Training**:
   ```bash
   # Full Rainbow DQN training (60-120 minutes)
   python train.py --config presets/chet_sim_rainbow_fast.json --run-name production-run
   
   # Hyperparameter sweep (4-12 hours)
   python sweep.py --grid '{"lr": [0.0001, 0.0003], "batch_size": [64, 128]}' --config presets/base.json
   ```

### Model Evaluation and Analysis
```bash
# Evaluate latest checkpoint
python evaluate.py --model-path models/dqn_latest.pt --episodes 10 --render

# Evaluate specific checkpoint
python evaluate.py --model-path models/dqn_step_200000.pt --episodes 5

# Evaluate with different environment
python evaluate.py --env real --capture-rect 800 400 300 300 --model-path models/dqn_best.pt
```

### ChetBot Visual Automation (Separate Tool)
```bash
# Test automation (dry run)
python -m fishbot.main --dry-run --debug

# Calibration workflow  
python -m fishbot.main --calibrate

# Live automation (use carefully)
python -m fishbot.main --config path/to/config.json
```

## Validation Scenarios  

**CRITICAL**: After making code changes, always run these validation steps in order:

### 1. Import Validation
```bash
python -m pytest tests/test_imports.py -v
# Ensures all modules can be imported correctly
# Expected time: 30 seconds - 2 minutes
# If this fails, check virtual environment activation and dependencies
```

### 2. Training Smoke Test  
```bash
python -m pytest tests/test_tiny_train.py -v
# Runs mini training loop on CPU to catch runtime errors
# NEVER CANCEL: Takes 5-15 minutes. Set timeout to 30+ minutes.
# Tests: Basic DQN training with chet-sim environment for 300 steps
```

### 3. Manual Training Validation
```bash
python train.py --env chet-sim --total-steps 1000 --batch-size 32 --eval-every 0 --save-every 0 --render-every 0
# Quick functional test of training pipeline  
# NEVER CANCEL: Takes 2-10 minutes. Set timeout to 20+ minutes.
# Verifies: Environment creation, agent initialization, training loop
```

### 4. GPU and Performance Validation
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"

# Test AMP and compile features (if GPU available)
python train.py --env chet-sim --total-steps 500 --amp --compile --batch-size 64 --eval-every 0 --save-every 0
# NEVER CANCEL: Takes 3-12 minutes. Set timeout to 25+ minutes.
```

### 5. TensorBoard Logging Verification
```bash
# After any training run, verify logs are created:
ls -la "tensorboard relatorios"/
ls -la runs/
# Should contain timestamped directories with TensorBoard event files
```

### 6. Preset Configuration Validation
```bash  
# Test preset loading and training
python train.py --config presets/chet_sim_rainbow_fast.json --total-steps 2000 --run-name preset-test
# NEVER CANCEL: Takes 5-20 minutes. Set timeout to 35+ minutes.
# Verifies: JSON config parsing, Rainbow features, advanced options
```

### 7. Model Evaluation Validation
```bash
# After training produces a checkpoint:
python evaluate.py --model-path models/dqn_latest.pt --episodes 3 --render
# Takes 1-5 minutes depending on episode length
# Verifies: Model loading, evaluation loop, rendering (if applicable)
```

### 8. Memory and Resource Validation  
```bash
# Test memory-intensive configurations
python train.py --env chet-sim --total-steps 1000 --batch-size 128 --buffer-size 50000 --num-envs 2
# NEVER CANCEL: Takes 3-15 minutes. Set timeout to 30+ minutes.
# Verifies: Large batch handling, parallel environments, memory management
```

### TensorBoard Monitoring
```bash
# View training logs
tensorboard --logdir "tensorboard relatorios" --port 6006

# Monitor specific experiment  
tensorboard --logdir runs/my-experiment --port 6006
```

## Repository Structure and Key Files

### Core Training Components
```
train.py                 # Main training script (941 lines) - Primary entry point
├── Args dataclass       # Complete configuration with 70+ parameters
├── make_env()          # Environment factory (sim/chet-sim/real)
├── async learning      # Optional parallel learner thread
└── evaluation loops    # Periodic model validation

evaluate.py             # Model evaluation and testing
├── Model loading       # Automatic architecture detection
├── Episode rollouts    # Performance measurement  
└── Rendering support   # Visual validation

agents/
├── dqn.py             # DQN agent with Rainbow features
│   ├── DQNAgent       # Main agent class with act/learn methods
│   ├── DQNConfig      # Network architecture configuration
│   ├── Rainbow        # C51, QR-DQN, Dueling, Noisy, PER integration
│   └── Optimizations  # AMP, compile, memory formats
└── replay_buffer.py   # Experience replay with PER and memory mapping
    ├── ReplayBuffer   # Standard uniform sampling
    ├── PrioritizedReplayBuffer # PER implementation
    └── MemoryMapping  # Disk-backed large buffers
```

### Environment Implementations  
```
env/                    # Note: May be dynamically imported (not present in all checkouts)
├── sim_env.py         # Pygame-based local simulation
├── chet_sim_env.py    # OpenCV-based fast simulator  
└── fishing_env.py     # Real Windows screen capture environment

# Key environment features:
# - Gymnasium-compatible interface  
# - Frame stacking and preprocessing
# - Action space: [0=hold, 1=move_left, 2=move_right]
# - Observation: 84x84 grayscale stacked frames
```

### Utilities and Preprocessing
```
utils/
├── preprocessing.py   # Frame preprocessing (resize, normalize, stack)
├── screen.py         # Windows screen capture (mss library)
├── vision.py         # Computer vision utilities (bar detection)
├── logger.py         # TensorBoard integration
└── schedules.py      # Learning rate and exploration scheduling
```

### Configuration and Presets
```
presets/
├── chet_sim_rainbow_fast.json    # Recommended preset: Rainbow DQN optimized
├── chet_sim_qr_fast.json         # QR-DQN variant
├── chet_sim_rainbow.json         # Full Rainbow (longer training)
└── chet_sim_4070_beefy.json      # High-performance GPU preset

# All presets include:
# - Optimized hyperparameters
# - Feature combinations (C51, PER, Noisy, etc.)
# - Performance flags (AMP, async-learner)
```

### ChetBot Visual Automation
```
chet-bot/
├── fishbot/
│   ├── main.py           # CLI entry point
│   ├── minigame.py       # Local test harness (visual simulator)
│   ├── calibration.py    # ROI and template selection
│   ├── capture.py        # Screen capture pipeline (dxcam/mss)
│   ├── detection.py      # Fish and rectangle detection  
│   ├── controller.py     # Predictive control FSM
│   └── config.py         # Configuration management
└── README.md            # ChetBot documentation
```

### Build and Test Infrastructure
```
tests/
├── conftest.py          # Test configuration and path setup
├── test_imports.py      # Import validation for all modules
└── test_tiny_train.py   # Smoke test with mini training loop

.github/
├── workflows/ci.yml     # Windows-based CI with pytest
└── copilot-instructions.md # This file

scripts/
└── seed_github.ps1      # Repository initialization script
```

### Generated Artifacts (Gitignored)
```
models/                  # Model checkpoints
├── dqn_latest.pt       # Most recent checkpoint  
├── dqn_best.pt         # Best performing model
├── dqn_final.pt        # Final model on completion
├── dqn_step_*.pt       # Periodic checkpoints
└── trainer_state.json  # Training state persistence

runs/                    # TensorBoard logs by experiment
└── [run-name]/         # Individual experiment directories

tensorboard relatorios/  # Portuguese: "TensorBoard reports"
└── [experiment logs]   # Auto-generated training logs

replay_memmap*/          # Memory-mapped replay buffers (when enabled)
└── [binary data]       # Large buffer files
```

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

### Essential Training Arguments (Validated)
```bash
# Basic configuration
--env sim|chet-sim|real          # Environment type
--total-steps 200000             # Training duration
--batch-size 64                  # Training batch size
--buffer-size 100000             # Replay buffer capacity
--start-learning 10000           # Steps before learning starts
--run-name my-experiment         # Experiment identifier
--seed 42                        # Random seed for reproducibility

# Learning parameters
--lr 0.0001                      # Learning rate
--gamma 0.99                     # Discount factor
--epsilon-start 1.0              # Initial exploration
--epsilon-end 0.05               # Final exploration
--epsilon-decay 150000           # Exploration decay steps
--target-update-every 1000       # Target network update frequency
--tau 0.0                        # Polyak averaging (0=hard updates)
```

### Performance Optimization
```bash
--amp                            # Automatic Mixed Precision (GPU speedup)
--compile                        # PyTorch 2.0 compile mode
--async-learner                  # Asynchronous learning thread
--num-envs 4                     # Parallel environments (chet-sim only)
--updates-per-step 2             # Multiple gradient updates per step
--pin-memory                     # CUDA memory pinning
--channels-last                  # NHWC memory format (GPU optimization)
--allow-tf32                     # TensorFloat-32 acceleration
```

### DQN Algorithm Variants
```bash
# Architecture improvements
--dueling                        # Dueling DQN (separate value/advantage)
--noisy                          # Noisy networks for exploration
--noisy-sigma-init 0.5           # Initial noise parameter
--noisy-sigma-final 0.5          # Final noise parameter

# Distributional RL
--c51                            # Categorical DQN (C51)
--num-atoms 51                   # Number of distributional atoms
--vmin -10.0 --vmax 10.0        # Value distribution range
--qr-dqn                         # Quantile Regression DQN (alternative to C51)
--num-quantiles 51               # Number of quantiles for QR-DQN
--huber-kappa 1.0               # Huber loss parameter

# Experience Replay
--prioritized                    # Prioritized Experience Replay (PER)
--per-alpha 0.6                  # PER prioritization exponent
--per-beta-start 0.4             # PER importance sampling start
--per-beta-end 1.0               # PER importance sampling end
--per-beta-decay 200000          # PER beta annealing steps
--n-step 3                       # Multi-step returns
--replay-memmap-dir replay_memmap # Memory-mapped replay buffer
```

### Advanced Learning Techniques
```bash
# Learning rate scheduling
--lr-schedule cosine             # Cosine annealing schedule
--lr-warmup-steps 10000         # Learning rate warmup

# Munchausen DQN
--munchausen                     # Enable Munchausen DQN
--munchausen-alpha 0.9           # Munchausen scaling factor
--munchausen-tau 0.03            # Munchausen temperature
--munchausen-clip -1.0           # Munchausen clipping value

# Curriculum learning
--curriculum-v2                  # Adaptive curriculum (chet-sim)
--curv2-reward-threshold 0.7     # Curriculum advancement threshold
--curv2-scale 1.10               # Curriculum difficulty scaling
```

### Environment-Specific Options
```bash
# Real environment (Windows only)
--capture-rect 800 400 300 300   # Screen capture region (x y w h)
--safe-mode                      # Limit mouse movement speed/range

# Simulation environments  
--frame-skip 4                   # Action repeat
--stack-frames 4                 # Frame stacking
--sim-speed-scale 1.0            # Simulation speed multiplier
--render-every 2000              # Visualization frequency (sim/chet-sim)

# Data augmentation
--drq                            # DrQ augmentation
--drq-pad 4                      # DrQ padding size
```

### Evaluation and Logging
```bash
--eval-every 25000               # Evaluation frequency  
--eval-episodes 5                # Episodes per evaluation
--save-every 50000               # Model checkpoint frequency
--log-video-every 10000          # Video logging frequency
--video-length 600               # Max steps per video
--resume-from models/checkpoint.pt # Resume training
```

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