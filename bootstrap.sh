#!/bin/sh

git clone https://github.com/osmr/facedetver.git

mkdir facedetver_data
cd facedetver_data
mkdir fdv1
cd fdv1
wget https://github.com/osmr/facedetver/releases/download/v0.0.1/fdv1_test.zip
unzip fdv1_test.zip

cd ..
mkdir resnet18_fdv1-0014
cd resnet18_fdv1-0014
wget https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0014-a03f116e.params.zip
unzip resnet18_fdv1-0014-a03f116e.params.zip

cd ..
mkdir resnet18_fdv1-0011
cd resnet18_fdv1-0011
wget https://github.com/osmr/facedetver/releases/download/v0.0.3/resnet18_fdv1-0011-85475034.pth.zip
unzip resnet18_fdv1-0011-85475034.pth.zip

cd ../../facedetver
#python3 eval_gl.py --num-gpus=1 --model=resnet18 --save-dir=../facedetver_data/resnet18_fdv1-0014/ --batch-size=100 -j=4 --resume=../facedetver_data/resnet18_fdv1-0014/resnet18_fdv1-0014-a03f116e.params --calc-flops --show-bad-samples --data-subset=test
python3 eval_pt.py --num-gpus=1 --model=resnet18 --save-dir=../facedetver_data/resnet18_fdv1-0011/ --batch-size=100 -j=4 --resume=../facedetver_data/resnet18_fdv1-0011/resnet18_fdv1-0011-85475034.pth --calc-flops --show-bad-samples --data-subset=test