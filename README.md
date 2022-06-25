# AAT
Attention-Aware Three-Branch Networks for Salient Object Detection in Remote Sensing Images

#### [Xin Wang](https://github.com/WangXin81) , [Zhilu Zhang], Huiyu Zhou




## Usage
1. Data preparation:

```
dataset|——edges
	   |——0001
	   |——0002
	   |——....
       |——images
	   |——0001
	   |——0002
	   |——....
       |——labels
     |——0001
	   |——0002
	   |——....
       |——test.txt
       |——train.txt   	   
```

2. The training and testing pipeline is organized in run.ipynb (Jupyter Notebook).
3. Enter the "./Envaluation" folder and run the "main.py" file to obtain 1) MAE; 2) F-measure; 3) S-measure.


## Figs

![image-20210601165926181](https://github.com/WangXin81/ACRNet/blob/main/2021-06-01_171017.png)

## Datasets:

EORSSD: 

[http://weegee.vision.ucmerced.edu/datasets/landuse.html](https://github.com/rmcong/EORSSD-dataset)

ORSSD: 

[https://captain-whu.github.io/AID/](https://pan.baidu.com/s/1k44UlTLCW17AS0VhPyP7JA)

ECSSD: 

[http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)

PASCAL-S:

https://academictorrents.com/details/6c49defd6f0e417c039637475cde638d1363037e

DUTSTE:

http://saliencydetection.net/duts/

DUT-OMRON:

http://saliencydetection.net/dut-omron/#outline-container-org0e04792

HKU-IS:

https://i.cs.hku.hk/~gbli/deep_saliency.html


## Environments

1. Ubuntu 18.04
2. cuda 10.2
3. pytorch 1.9.0
4. python 3.7
