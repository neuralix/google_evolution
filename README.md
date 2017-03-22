### Dependences
pytorch, scikit-learn

### Steps
1. Overwrite `{TORCHVISON_PATH}/models/__init__.py` with new one.
2. Copy `evolution.py` to `{TORCHVISON_PATH}/models`.
3. Run.
`python main.py -a evolution {CIFAR10_DATA_DIR}`

cf. How to know {TORCHVISON_PATH}?
```
import torchvision
print(torchvision.__file__)
```

### Note
Numbers of channels are not in the paper. It is set by me.

### Original Paper
https://arxiv.org/abs/1703.01041 (Google Brain)

![alt](fig_network.png)
