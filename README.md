# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).



<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).


### Step2 (optional): Create Virtual environment 

```
# creation environmen with name "my_venv"
virtualenv my_venv
or (if it does not work)
python3 -m venv my_venv

# activate
source venv/bin/activate

# install requirements
pip install -r requirements.txt

# install table separetly (if you put it in requirements.txt it does not work)
pip istall table
and (if it does not work)
pip install tables

# in case of issue with  conv1D (RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but ...) - https://github.com/YuvalNirkin/fsgan/issues/162
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
and (if the error persists)
pip install opencv-python


```

### Step3: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```


