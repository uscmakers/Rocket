## Quick Navigation
- [Template for Isaac Lab Projects](#template-for-isaac-lab-projects)
- [Isaac Sim + Isaac Lab Quick Start (HPC)](#isaac-sim--isaac-lab-quick-start-hpc)
- [Running Checkpoints on Roboland Isaac Sim](#running-checkpoints-on-roboland-isaac-sim)


# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/rocket

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/rocket/rocket/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/rocket"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```



# Isaac Sim + Isaac Lab Quick Start (HPC)

## Prerequisites (Satisfied by USC Carc)
- Nvidia GPU required (RTX GPUs preferred for faster training)
- Apptainer/Singularity available
- No sudo needed

## One-Time Setup

SSH into the cluster and ensure you are in your home directory `home1/<usc_username>`:
```bash
ssh discovery
pwd
```

### 1. Check and load Apptainer
```bash
module load apptainer
apptainer --version
```

### 2. Pull Isaac Sim container
```bash
apptainer pull isaac-sim_5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0
```

### 3. Create cache directories
```bash
mkdir -p ~/isaac-sim-cache/{data,cache,logs}
```

### 4. Clone the Rocket repository
```bash
git clone https://github.com/uscmakers/Rocket.git
cd Rocket
```

### 5. Create setup file for W&B credentials
Create `setup.sh` in your home directory (`/home1/<username>/setup.sh`) if it does not already exist:
```bash
#!/bin/bash
export WANDB_API_KEY="your_wandb_api_key"
```

Make it executable:
```bash
chmod +x ~/setup.sh
```

**Important:** Replace `your_wandb_api_key` with your Weights & Biases API key (get from https://wandb.ai/authorize).

### 6. Configure SLURM account in train.sh (Optional)
If you want to use a specific SLURM account, edit `train.sh` and uncomment/add this line:
```bash
#SBATCH --account=your_account_name
```

If commented out, your default SLURM account will be used. Check your accounts with: `sacctmgr show user $USER`

**Note:** All pip packages (Isaac Lab, rl-games, etc.) are automatically installed by the sbatch script on first run and cached for future runs.

## Running Training

All sbatch scripts are located in the `Rocket` directory.

### Submit Training Job
```bash
cd ~/Rocket
sbatch train.sh
```

### Submit Evaluation/Play Job
```bash
cd ~/Rocket
sbatch play.sh
```

### Customize Training/Evaluation
Edit the shell scripts (`train.sh`, `play.sh`) to modify running scripts and flags.

### Monitor Jobs
```bash
# View all your jobs
squeue -u $USER

# Quick view (alias myqueue to this if desired)
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

# Check estimated start time for queued job
squeue --start -j <jobid>

# View job output (while running or after completion)
tail -f jobs/rocket_<jobid>.out
```

### View Results
Job outputs are saved to `jobs/rocket_<jobid>.out` and errors to `jobs/rocket_<jobid>.err`.

## Key Rules
- Always use `--nv` for GPU passthrough
- Always bind the 3 cache directories
- Use `/isaac-sim/python.sh` not `python`
- **Always use `--headless` flag** on HPC - prevents GUI rendering (no display available, saves memory/CPU)
- GPU required for running simulations
- GPU NOT required for pip installs
- sbatch scripts automatically install/update packages on each run, the sif container image is read-only

# Running checkpoints on Roboland Isaac Sim

Once you have connected to the Roboland computer, Navigate to the Rocket directory and run the checkpoint script:
```bash
cd Rocket
./run.sh
```

This will play the currently loaded checkpoint in real-time on Isaac Sim.

** IMPORTANT: If you have too many isaac sim simulations running simultaneously, you may run out of memory and get a ```CUDA: OUT OF MEMORY``` error. In this case, you should exit out of all other isaac-sim simulations. Run `nvidia-smi` in terminal to debug your GPU usage if needed. 

### Running Different Checkpoints

To run a new checkpoint:

1. **Verify the checkpoint exists**:
```
   Rocket/logs/rl_games/rocket_direct/<checkpoint_name>/nn/rocket_direct.pth
```

2. **If the checkpoint is missing:**
   - Push the checkpoint from the training computer to the Git repo
   - Run `git pull` on Roboland to get the latest checkpoint

3. **Update the checkpoint path** in `run.sh`:
```bash
   --checkpoint logs/rl_games/rocket_direct/<your_checkpoint_name>/nn/rocket_direct.pth
```

4. **Re-run:**
```bash
   ./run.sh
```
