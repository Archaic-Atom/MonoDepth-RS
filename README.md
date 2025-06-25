# MonoDepth-RS

[Trained model](https://pan.baidu.com/s/1H41V78ddq6KIT3BD60jLeg) 提取码: sn5a

<div align="center">




## Installation
```
conda create -n iebins python=3.8
conda activate iebins
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib, tqdm, tensorboardX, timm, mmcv, open3d
```

## Training
Training the whu_omvs model:
```
python iebins/whu_train.py uav_configs/arguments_train_whu.txt
```

Training the whu_mvs model:
```
python iebins/whu_mvs_train.py uav_configs/arguments_train_whu.txt
```


## Evaluation
Evaluate the whu_omvs model:
```
python iebins/whu_eval.py uav_configs/arguments_eval_whu.txt
```

Evaluate the whu_mvs model on the SUN RGB-D dataset:
```
python iebins/whu_mvs_eval.py uav_configs/arguments_eval_whu_mvs.txt
```


To generate whu_omvs picture:
```
python iebins_kittiofficial/test.py uav_configs/arguments_test_whu.txt
```



<img width="1111" alt="comingsoon" src="https://user-images.githubusercontent.com/6929121/87441911-486bf600-c611-11ea-9d45-94c215733cf7.png">


