# **RainHistoNet**: Single-Image Day and Night Raindrop Removal via Histogram-Guided Restoration

## Description

As shown in the figure, our model is based on [**ESDNet**](https://github.com/CVMI-Lab/UHDM). The backbone primarily consists of an encoder-decoder network. At each encoder and decoder level, a Semantic-Aligned Scale-Aware Module (SAM) is incorporated to address scale variations. Additionally, we introduce a Histogram Transformer Block from [**Histoformer**](https://github.com/CVMI-Lab/UHDM), which employs histogram self-attention with dynamic range spatial attention. This block is placed between the encoder and decoder to achieve global and efficient degradation removal.

![net](https://github.com/minyan8/RainHistoNet/blob/main/figs/net.png)

## Installation

You can follow the step of MIRNet-v2 in [here](https://github.com/swz30/MIRNetv2/blob/main/INSTALL.md#installation) or based on follows:

1. Clone our repository

```
git clone https://github.com/minyan8/RainHistoNet.git
cd RainHistoNet
```

2. Make the conda environment

```
conda create -n rain python=3.9
conda activate rain
```

3. Install dependencies

- We trained and tested our model using CUDA Toolkit version 11.8.
```
# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
pip install numpy==1.26.4
```

- We have included a requirements.txt file in our repository. Please refer to it if you encounter any issues related to package versions.

4. Install basicsr

```
python setup.py develop --no_cuda_ext
```

### Train

Set the dataset root in the configuration file located at `./Enhancement/Options/rainHistoNet.yml`, and then run the script below.

```
python basicsr/train.py
```

### Test

To test on your own data, you can run:

```
cd Enhancement
python test.py --weights [your pretrained model weights] --input_dir [your input data path] --result_dir [your result saved path] --dataset [your dataset name]
```

We have placed our pre-trained model for this challenge in `Enhancement/pretrained_models/net_g_150000.pth`. **If you just want to run the challenge official input data, you can run** 

```
cd Enhancement
python test.py --input_dir [your input data path]
```

the results will be saved in `Enhancement/results/NtireLL`, also, the final result can be downloaded [here](https://drive.google.com/file/d/1RE_DMzGqqX9bOusmIpKGEj7h_4UF3kWE/view?usp=sharing)

## Acknowledgments

This repository draws inspiration and builds upon the following outstanding projects:

- [**UHDM**](https://github.com/CVMI-Lab/UHDM) 
- [**SYSU-FVL-T2**](https://github.com/wangchx67/SYSU-FVL-T2) 
- [**Histoformer**](https://github.com/CVMI-Lab/UHDM)

We sincerely thank the authors of these works for their valuable contributions to the community.
