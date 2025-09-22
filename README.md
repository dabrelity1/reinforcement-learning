# Fishing RL (DQN)

Treino de uma IA com DQN para jogar um minijogo de pesca mantendo a barra branca sobreposta à barra cinzenta.

## Estrutura

- `env/`
  - `sim_env.py`: ambiente de simulação (Pygame) para treino rápido.
  - `fishing_env.py`: ambiente real via captura de ecrã e controlo do rato.
  - `chet_sim_env.py`: simulador compatível com o minijogo do chet-bot (OpenCV), ideal para ver a aprendizagem.
- `agents/`
  - `dqn.py`: implementação do DQN (rede, treino, target network).
  - `replay_buffer.py`: buffer de replay com experiências e frame stacking.
- `utils/`
  - `preprocessing.py`: recorte, grayscale, resize para 84×84 e normalização.
  - `vision.py`: deteção de barras no ecrã real e cálculo de overlap.
  - `screen.py`: captura de ecrã (mss) e controlo do rato (pyautogui).
  - `logger.py`: TensorBoard logging e checkpoint helpers.
  - `schedules.py`: agendas (epsilon, lr) e curriculum.
- `train.py`: script de treino (simulação ou real), com frame skip e logging.
- `evaluate.py`: avaliação/teste de um modelo treinado.

## Requisitos

Windows 10/11 com permissões para captura de ecrã e controlo do rato.

Python 3.10+ recomendado.

Instalar dependências:

```pwsh
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Se o `pyautogui` pedir dependências extra, confirme permissões de Acessibilidade/Security no Windows.

## Como correr (simulação)

Treino num ambiente Pygame local (mais rápido):

```pwsh
python train.py --env sim --total-steps 200000 --run-name sim-baseline
```

Renderizar ocasionalmente:

```pwsh
python train.py --env sim --render-every 2000
```

Simulador inspirado no chet-bot (linha cinzenta e barra branca com física semelhante):

```pwsh
python train.py --env chet-sim --total-steps 200000 --render-every 1000 --run-name chet-sim
```

## Como correr (jogo real)

- Abra o minijogo e deixe a janela visível.
- Ajuste a região de captura com `--capture-rect x y w h`.
- Tenha o rato livre; o script vai movê-lo verticalmente.

Exemplo:

```pwsh
python train.py --env real --capture-rect 800 400 300 300 --total-steps 300000 --frame-skip 4 --run-name real-dqn
```

Notas:
- O jogo começa com ~1.5s de barras centradas; o ambiente respeita isso no `reset()`.
- Use `--safe-mode` para limitar velocidade e amplitude do rato enquanto afina a região.

## Checkpoints e TensorBoard

Checkpoints são gravados em `models/` e logs do TensorBoard em `runs/`.

Abrir TensorBoard:

```pwsh
tensorboard --logdir runs
```

Avaliar último checkpoint rapidamente:

```pwsh
python evaluate.py --env chet-sim --model_path models/dqn_latest.pt --episodes 5 --render True
```

### Presets e modos avançados

Para ativar Rainbow (C51 + PER + n-step + Noisy + Dueling + AMP) num só comando:

```pwsh
python train.py --config presets/chet_sim_rainbow_fast.json --run-name chet-sim-rainbow
```

Ou experimentar QR-DQN:

```pwsh
python train.py --config presets/chet_sim_qr_fast.json --run-name chet-sim-qr
```

Opções úteis (todas disponíveis via `--help`):
- `--num-envs 4` paraleliza o chet-sim (AsyncVectorEnv)
- `--async-learner` ativa um thread de aprendizagem em paralelo
- `--c51` ou `--qr-dqn` (exclusivos) para distribuição de valores
- `--n-step 3` para returns de 3 passos
- `--prioritized` (PER), com `--per-alpha` e `--per-beta-*`
- `--amp` para mixed precision (mais rápido na GPU)
- `--replay-memmap-dir replay_memmap` grava frames no disco e reduz RAM
- `--log-video-every 25000` e `--video-length 600` para vídeos curtos

Recomeçar um treino a partir do último checkpoint:

```pwsh
python train.py --resume-from models/dqn_latest.pt --config presets/chet_sim_rainbow_fast.json
```

### Docker (opcional)

O projeto inclui um `Dockerfile` (CUDA) para treino containerizado. Ajuste volumes e acesso à GPU conforme o seu host.

### Experimentos (sweeps)

Use `sweep.py` para varrer grelhas de hiperparâmetros em série. Edite o ficheiro para definir combinações.

## Segurança

- Durante treino real, não use o rato para outras tarefas.
- Considere usar ecrã dedicado ou VM.

## Licença

Uso educacional. Sem garantias.
