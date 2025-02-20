# Synthetic Data for Face Recognition (SDFR) Competition
### The 18th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2024)
### [`https://www.idiap.ch/challenge/sdfr/leaderboard.html`](https://www.idiap.ch/challenge/sdfr/leaderboard.html)

#### Team details:
- Team name: BOVIFOCR-UFPR
- Team members: Bernardo Biesseck¹, Pedro Vidal¹, Roger Leitzke Granada², Luiz Coelho², David Menotti¹
- Affiliation: ¹Federal University of Paraná (UFPR), ²unico-idTech

<br>

# DATA AUGMENTATION

#### 1. Requirements:
- CUDA=10.2
- Python=3.6

#### 2. Clone the repo [`face_pose_augmentation`](https://github.com/BOVIFOCR/face_pose_augmentation):
```
git clone https://github.com/BOVIFOCR/face_pose_augmentation.git
cd face_pose_augmentation
```

#### 3. Augment data:
```
export CUDA_VISIBLE_DEVICES=0; python python face_pose_augmentation_main_BOVIFOCR.py --input-folder /path/to/dataset_input --output-folder /path/to/dataset_output --shuffle-subfolders --random-sample --samples-per-folder 1 --yaw 60
```

#### 4. Merge original data and augmented data:
- Merge `/path/to/dataset_input` and `/path/to/dataset_output` to obtain the new dataset.

<br>

# FACE RECOGNITION MODEL

#### 1. Requirements:
- CUDA=11.6
- Python=3.9

#### 2. Clone this repo [`insightface_SDFR_at_FG2024`](https://github.com/BOVIFOCR/insightface_SDFR_at_FG2024):
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

#### 4. Train model (Resnet50):
```
export CUDA_VISIBLE_DEVICES=0; python train_v2_sdfr2024.py configs/idiffface-uniform_sdfr2024_r50_yaw-augment=60.py
```

<br>

# CONVERT MODEL TO ONNX FORMAT

#### 1. Requirements:
- CUDA=11.8
- Python=3.10

#### 2. Create conda env:
```
conda env create -f submission_kit/dev_kit_v1_1/sdfr_env.yaml
conda activate sdfr
```

#### 3. Convert model:
```
python export_onnx_sdfr_py310.py --config configs/idiffface-uniform_sdfr2024_r50_yaw-augment=60.py --weights work_dirs/idiffface-uniform_sdfr2024_r50_yaw-augment=60/2024-03-10_23-57-49/model.pt
```

Output file `model_r50.onnx` will be saved next to file `model.pt`. 

#### 4. Generate sanitizer scores:
```
python export_sanitize_model_sdfr_py310.py --model_path work_dirs/idiffface-uniform_sdfr2024_r50_yaw-augment=60/2024-03-10_23-57-49/model_r50.onnx --task task2
```

Output file `task2_sanitizer_scores.txt` will be saved next to file `model_r50.onnx`.
