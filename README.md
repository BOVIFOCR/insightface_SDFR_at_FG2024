## CONFIG ENVIRONMENT (BOVIFOCR)

#### 1. Requirements:
- CUDA=11.6

#### 2. Clone this repo:
```
git clone https://github.com/BOVIFOCR/insightface_SDFR_at_FG2024.git
cd insightface_SDFR_at_FG2024
``` 

#### 3. Create conda env and install python libs:
```
export CONDA_ENV=insightface_sdfr2024_py39
conda create -y -n $CONDA_ENV python=3.9
conda activate $CONDA_ENV
conda env config vars set CUDA_HOME="/usr/local/cuda-11.6"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib64"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set PATH="$CUDA_HOME:$CUDA_HOME/bin:$LD_LIBRARY_PATH:$PATH"; conda deactivate; conda activate $CONDA_ENV

conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install -y pytorch3d -c pytorch3d
pip3 install -r requirements.txt
```

#### 4. Train model:
```
export CUDA_VISIBLE_DEVICES=0; python train_v2_sdfr2024.py configs/idiffface-uniform_sdfr2024_r50_yaw-augment=60.py
```
<br> <br> <br>

