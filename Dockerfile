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

ADD bootstrap.sh /root/
RUN chmod ugo+x /root/bootstrap.sh
CMD /root/bootstrap.sh