# A2G-MADRL

This project is an air-to-ground multi-agent deep reinforcement learning (A2G-MADRL) for mobile crowdsensing (MCS) source code repository. It includes reinforcement learning algorithms, environment configurations, experiment tools, and visualization modules. It is suitable for research and experiments in multi-agent path planning, task allocation, and related areas.

## Directory Structure

```
.
├── algorithms/           # RL algorithms and models
│   ├── algo/             # Algorithm implementations (e.g., IPPO, IA2C)
│   ├── config/           # Algorithm and experiment configs
│   ├── mat/              # Multi-agent Transformer related
│   ├── model/            # Network models
│   ├── utils.py          # Algorithm utilities
│   └── models.py         # Common model definitions
├── env_configs/          # Environment configs and wrappers
│   ├── roadmap_env/      # Roadmap environments and tools
│   ├── wrappers/         # Environment wrappers
├── LaunchMCS/            # Main MCS environment and tools
│   ├── env_setting/      # Environment parameter settings
│   ├── util/             # Utility functions
│   ├── launch_mcs.py     # Main environment entry
│   └── compression.py    # Data compression
├── tools/                # Data processing and visualization tools
│   ├── macro/            # Macro definitions
│   ├── post/             # Post-processing and visualization
│   ├── pre/              # Pre-processing scripts
├── get_args.py           # Argument parser
├── main_DPPO.py          # Main training entry (example: DPPO)
├── sweep.yaml            # Hyperparameter search config
└── README.md             # Project description
```

## Quick Start

### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Install dependencies with:

```sh
pip install -r requirements.txt
```

### 2. Run Training

For example, to run the DPPO algorithm:

```sh
python main_DPPO.py --dataset KAIST --algo CPPO --random_permutation --n_iter 30000 --gpu 2 --device cuda:0
```

See [`get_args.py`](get_args.py) or [`sweep.yaml`](sweep.yaml) for parameter details.

### 3. Environment and Algorithm Configurations

- Environment configs: [`env_configs`](env_configs), [`LaunchMCS/env_setting`](LaunchMCS/env_setting)
- Algorithm/experiment configs: [`algorithms/config`](algorithms/config)

### 4. Visualization

After training, use the tools for visualization:

```sh
python tools/post/vis.py --output_dir <your_experiment_output>
```

Or summarize experiment results:

```sh
python tools/post/walk_summary.py --group_dir <your_experiment_group>
```

## Main Modules

- [`algorithms`](algorithms): Multiple RL algorithms (IPPO, IA2C, DPPO, etc.) and models.
- [`env_configs`](env_configs): Environment configs and wrappers for different maps/tasks.
- [`LaunchMCS`](LaunchMCS): Main MCS environment, data processing, etc.
- [`tools`](tools): Data pre/post-processing and visualization.


## Acknowledgements

This project integrates various RL and multi-agent system methods, suitable for academic research and engineering experiments.

---

For questions, please contact the project maintainer.
````
