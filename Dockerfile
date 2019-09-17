FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="osemery@gmail.com"

RUN apt update
RUN apt install -y htop mc wget unzip python3-pip ipython3
RUN apt install -y build-essential git ninja-build ccache
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && export DEBIAN_FRONTEND=noninteractive && apt install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata
RUN apt install -y apt-transport-https build-essential ca-certificates curl git libatlas-base-dev libcurl4-openssl-dev libjemalloc-dev libhdf5-dev liblapack-dev libopenblas-dev libopencv-dev libturbojpeg libzmq3-dev ninja-build software-properties-common sudo vim-nox
RUN apt install -y libopenblas-dev libopencv-dev
RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade numpy opencv-python imgaug tqdm
RUN pip3 install --upgrade mxnet-cu100 gluoncv2
RUN pip3 install --upgrade torch torchvision pytorchcv

RUN git clone https://github.com/osmr/facedetver.git

WORKDIR /facedetver_data/fdv1
RUN wget https://github.com/osmr/facedetver/releases/download/v0.0.1/fdv1_test.zip
RUN unzip fdv1_test.zip

WORKDIR /facedetver_data/resnet18_fdv1-0014
RUN wget https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0014-a03f116e.params.zip
RUN unzip resnet18_fdv1-0014-a03f116e.params.zip

WORKDIR /facedetver
RUN python3 eval_gl.py --num-gpus=1 --model=resnet18 --save-dir=../facedetver_data/resnet18_fdv1-0014/ --batch-size=100 -j=4 --resume=../facedetver_data/resnet18_fdv1-0014/resnet18_fdv1-0014-a03f116e.params --calc-flops --show-bad-samples --data-subset=test
ENTRYPOINT CMD tail -f /dev/null
