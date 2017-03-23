Exotic structured image classifier
=====================================

This implements training of residual networks from [Large-scale evolution of image classifiers](https://arxiv.org/abs/1703.01041) by Esteban Real, et. al.

## Requirements
- Install [pytorch](http://pytorch.org/) (I recommend Anaconda environment.)
- Install [scikit-learn](http://scikit-learn.org/stable/)

## Steps
1. Copy 2 files to {TORCHVISON_PATH}/models.

`cp {__init__.py,simple.py} {TORCHVISON_PATH}/models`

2. Run

`python main.py -a evolution {CIFAR10_DATA_DIR}`

cf. How to know {TORCHVISON_PATH}?
```
import torchvision
print(torchvision.__file__)
```

## Note
Numbers of channels are not in the paper and it is set by me.
You need to adjust these for better performance.

## Implemented network
![alt](fig_network.png)
