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
<br>
```
python ./train.py
```


## Testing
The testing data is placed in the following file  
>data
>>test
>>>images

## Acknowledgments
Code is inspired by Retinex and CycleGAN.
