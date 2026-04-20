eval "$(conda shell.bash hook)"
# ######################## Phantom Env ###############################
conda create -n phantom python=3.10 -y
conda activate phantom
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# Install SAM2
cd submodules/sam2
pip install -v -e ".[notebooks]"
cd ../..

# Install Hamer
cd submodules/phantom-hamer
pip install -e .\[all\]
pip install -v -e third-party/ViTPose
wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
cd ../..

# Install mmcv
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0
pip install mmcv==1.3.9
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install numpy==1.26.4

# Install phantom-robosuite
cd submodules/phantom-robosuite
pip install -e .
cd ../..

# Install phantom-robomimic
cd submodules/phantom-robomimic
pip install -e .
cd ../..

# Install additional packages
pip install joblib mediapy open3d pandas
pip install transformers==4.42.4
pip install PyOpenGL==3.1.4
pip install Rtree
pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git
pip install protobuf==3.20.0
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0

# Download E2FGVI weights
cd submodules/phantom-E2FGVI/E2FGVI/release_model/
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
cd ../..

# Install phantom-E2FGVI
pip install -e .
cd ../..

# Install phantom 
pip install -e .

# Download sample data
cd data/raw
wget https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
unzip pick_and_place.zip
rm pick_and_place.zip
wget https://download.cs.stanford.edu/juno/phantom/epic.zip
unzip epic.zip
rm epic.zip
cd ../..
