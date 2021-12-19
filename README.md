# Sky-Segmentation
This repository  contains the nine hundred sky segmentation datasets and the sky segmentation model provided by us. <br>
<br>
* Sky segmentation model structure diagram<br>
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/%E5%9B%BE%E7%89%871.png)  


## Environment
* Python 3.6 <br>
* PyTorch 1.8.0 <br>
* CUDA 11.1 <br>
* Ubuntu 20.04 <br>

## Datasets
Baidu network disk：https://pan.baidu.com/s/1p2hlvfoi4FXi74Ar2qPfhA 
key：CcSs

## Training
The downloaded training data is placed in the following file  
>data
>>images  
>>labels
# Titan X (Pascal)
# chainer==2.0.2
# pytorch==0.2.0.post2
# pytorch-fcn==1.7.0

% cd examples/voc

% ./speedtest.py --gpu 2
==> Benchmark: gpu=2, times=1000, dynamic_input=False
==> Testing FCN32s with Chainer
Elapsed time: 45.95 [s / 1000 evals]
Hz: 21.76 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 42.63 [s / 1000 evals]
Hz: 23.46 [hz]

% ./speedtest.py --gpu 3 --dynamic-input
==> Benchmark: gpu=3, times=1000, dynamic_input=True
==> Testing FCN32s with Chainer
Elapsed time: 47.68 [s / 1000 evals]
Hz: 20.97 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 54.49 [s / 1000 evals]
Hz: 18.35 [hz]


## Testing
The testing data is placed in the following file  
>data
>>test
>>>images

## Acknowledgments
Code is inspired by Retinex and CycleGAN.
