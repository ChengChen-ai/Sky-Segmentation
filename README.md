# Sky-Segmentation
This repository  contains the nine hundred sky segmentation datasets and the sky segmentation model network(SSMNet) provided by us. <br>
* Sky segmentation model structure diagram<br>
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/%E5%9B%BE%E7%89%871.png)  


## Environment
* Python 3.6 <br>
* PyTorch 1.8.0 <br>
* CUDA 11.1 <br>
* Ubuntu 20.04 <br>

## Datasets
Baidu Cloud：https://pan.baidu.com/s/1p2hlvfoi4FXi74Ar2qPfhA 
Extraction code：CcSs  
Note: the data contains paired training sets and pre-trained models

## Training
The downloaded training data is placed in the following file  
>data
>>images  
>>labels

    python ./train.py


## Testing
The pre-trained models are placed in the following file
>results  

The testing data is placed in the following file  
>data
>>test
>>>images  

    python ./test.py  

* Test Results  
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/test_1.jpg)  
![image](https://github.com/ChengChen-ai/Sky-Segmentation/blob/main/data/MAG/test_2.jpg)
## Acknowledgments
Code is inspired by [Retinex](https://github.com/weichen582/RetinexNet) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
