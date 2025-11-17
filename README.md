# Installation
- Environment Setup
    ```bash
    # Anaconda env setup
    conda create -n lehome python=3.11
    conda activate lehome

    # Install PyTorch
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

    # Install lehome
    
    git clone http://git.lightwheel.ai/zeyi.li/lehome.git
    cd lehome
    python -m pip install -e source/lehome

    # Install lerobot
    pip install lerobot
    pip install 'lerobot[all]'          # All available features
    pip install 'lerobot[aloha,pusht]'  # Specific features (Aloha & Pusht)
    pip install 'lerobot[feetech]'      # Feetech motor support
    
    # Install IsaacSim
    cd ..
    pip install --upgrade pip
    pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com

    # Install IsaacLab
    git clone git@github.com:isaac-sim/IsaacLab.git
    sudo apt install cmake build-essential
    cd IsaacLab
    ./isaaclab.sh --install
    ``` 
