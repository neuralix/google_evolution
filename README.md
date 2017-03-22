Paper: https://arxiv.org/abs/1703.01041

1. Overwrite {torchvision}/__init__.py with new one.
2. Copy evolution.py to {torchvision}.
3. `python main.py -a evolution {CIFAR10_DATA}

cf. How to know {torchvision}?
```
import torchvision
print(torchvision.__file__)
```

Channel number is not in the paper. It is set by me.
