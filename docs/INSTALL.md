# Installation

This codebase is tested on python 3.10.13. Follow the below steps to create environment and install dependencies.

* Configure the environment for LLaVA 1.5
```bash
# Install the LLaVA dependencies, tested with CUDA 12.1.1
git clone https://github.com/wlin-at/CaD-VI.git
cd CaD-VI
pip install --upgrade pip
pip install -e .

# Install additional packages for training cases
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

* Configure the environment for the LLM - Mixtral 8x7b
```bash
# Tested with CUDA 12.1.1
conda create -n mixtral python=3.12 -y
conda activate mixtral
pip install --upgrade pip
pip install transformers==4.38.1
pip3 install torch torchvision torchaudio
pip install bitsandbytes tqdm
```
Note that the Mixtral environment requires a higher version of `transformers` than the LLaVA environment.
