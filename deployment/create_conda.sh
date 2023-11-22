conda create -n rl-11 python=3.11 -y
conda activate rl-11
#conda install -c conda-forge -c powerai gym -y
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
#conda install matplotlib openblas -y
#conda install -c conda-forge opencv ffmpeg -y


python -m pip install --upgrade pip wheel setuptools
python -m pip install gymnasium torch torchvision torchaudio matplotlib ruamel.yaml absl-py tensorboardX scikit-learn seaborn psutil dill tensorboard wandb pygame tqdm

