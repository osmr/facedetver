# Face Detection Verifier
This repo contains scripts for solving the binary classification problem, where the positive class is undistorted images
of a person’s face, and the negative class is everything else, including images of parts of a person’s face, face
drawings, etc.  

The repository contains scripts for building and analyzing the corresponding dataset, for training and testing models.
Two deep learning frameworks are supported: MXNet/Gluon and PyTorch. All scripts are completely duplicated.
In addition, the releases contain two training datasets and four models that solve the problem.

## Deployment Instructions 
Recommended repository deployment protocol on the machine with CUDA 10.0 and cuDNN 7:
1. Install prerequesities for MXNet:
    ```
    apt update
    apt upgrade
    apt install -y htop mc wget unzip python3-pip ipython3
    apt install -y build-essential git ninja-build ccache
    apt install -y apt-transport-https build-essential ca-certificates curl git libatlas-base-dev libcurl4-openssl-dev libjemalloc-dev libhdf5-dev liblapack-dev libopenblas-dev libopencv-dev libturbojpeg libzmq3-dev ninja-build software-properties-common sudo vim-nox
    apt install -y libopenblas-dev libopencv-dev
    apt install -y libsm6 libxext6 libxrender-dev
    ```
2. Install pip-packages from `requirements.txt`: 
    ```
    pip3 install --upgrade numpy opencv-python imgaug tqdm
    pip3 install --upgrade mxnet-cu100 gluoncv2
    pip3 install --upgrade torch torchvision pytorchcv
    ```
3. Clone the repo:
    ```
    mkdir projects
    cd projects
    git clone https://github.com/osmr/facedetver.git
    ```
4. Create directory for dataset and models:
    ```
    mkdir facedetver_data
    cd facedetver_data
    ```
4.1. Download and extract dataset FDV1:
    ```
    mkdir fdv1
    cd fdv1
    wget https://github.com/osmr/facedetver/releases/download/v0.0.1/fdv1_test.zip
    wget https://github.com/osmr/facedetver/releases/download/v0.0.1/fdv1_train.zip
    wget https://github.com/osmr/facedetver/releases/download/v0.0.1/fdv1_val.zip
    unzip fdv1_test.zip
    unzip fdv1_train.zip
    unzip fdv1_val.zip
    ```
4.2. Or download and extract dataset FDV2:
    ```
    mkdir fdv2
    cd fdv2
    wget https://github.com/osmr/facedetver/releases/download/v0.0.2/fdv2_test.zip
    wget https://github.com/osmr/facedetver/releases/download/v0.0.2/fdv2_train.zip
    wget https://github.com/osmr/facedetver/releases/download/v0.0.2/fdv2_val.zip
    unzip fdv2_test.zip
    unzip fdv2_train.zip
    unzip fdv2_val.zip
    ```
5. Download and extract a model:
    ```
    cd ..
    mkdir resnet18_fdv1-0014
    cd resnet18_fdv1-0014
    wget https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0014-a03f116e.params.zip
    unzip resnet18_fdv1-0014-a03f116e.params.zip
    ```
6. Run a testing script:
    ```
    cd ../../facedetver
    python3 eval_gl.py --num-gpus=1 --model=resnet18 --save-dir=../facedetver_data/resnet18_fdv1-0014/ --batch-size=100 -j=4 --resume=../facedetver_data/resnet18_fdv1-0014/resnet18_fdv1-0014-a03f116e.params --calc-flops --show-bad-samples --data-subset=test
    ```

## Pretrainded Models  

| Model | Dataset | Framework | Acc | F1 | MCC | Params | FLOPs/2 | Remarks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 | FDV1 | Gluon | 0.9976 | 0.9976 | 0.9952 | 11,177,538 | 1819.90M | Training ([log](https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0014-a03f116e.params.log)) |
| ResNet-18 | FDV1 | PyTorch | 0.9976 | 0.9976 | 0.9952 | 11,177,538 | 1819.90M | Training ([log](https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0011-85475034.pth.log)) |
| ResNet-18 | FDV2 | Gluon | 0.9971 | 0.9971 | 0.9942 | 11,177,538 | 1819.90M | Training ([log](https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv2-0011-391f0c7e.params.log)) |
| ResNet-18 | FDV2 | PyTorch | 0.9971 | 0.9971 | 0.9942 | 11,177,538 | 1819.90M | Training ([log](https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv2-0009-e1a3e6f2.pth.log)) |

