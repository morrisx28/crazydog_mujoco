
## Dependencies
- *Python* - 3.10  
- [*PyTorch* - 2.4.0 with CUDA 12.1](https://pytorch.org/get-started/previous-versions/)

## Installation
1. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

## Quick Start
1. Replay the RL policy on Biped wheel robot:
    ```bash
    conda activate rlmujoco
    cd ~/quick_bipedal_urdf
    python quick_gs2mj.py


