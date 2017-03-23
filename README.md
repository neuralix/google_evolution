Exotic structured image classifier
=====================================

This implements one of result networks from [Large-scale evolution of image classifiers](https://arxiv.org/abs/1703.01041) by Esteban Real, et. al.

## Requirements
- Install [pytorch](http://pytorch.org/) (I recommend anaconda environment.)
- Install [scikit-learn](http://scikit-learn.org/stable/)

## Steps
1. Copy two files to {torchvision_path}/models.

```cp {__init__.py,simple.py} {torchvision_path}/models```

2. Run

```python main.py -a evolution {cifar10_data_dir}```

cf. How to know {torchvision_path}?
```
import torchvision
print(torchvision.__file__)
```

## Note
Numbers of channels are not in the paper and it is set by me similar with vgg.
You need to adjust these for better performance.

## Implemented network
![alt](fig_network.png)
