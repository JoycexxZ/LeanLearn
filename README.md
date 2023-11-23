# Prunned models trained on CIFAR-10 dataset
- I modified [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) official implementation of popular CNN models, and trained those on CIFAR-10 dataset. Also used Neural Network Intelligence(https://nni.readthedocs.io/en/stable/compression/overview.html) for pruning.
- The **weights** of these models are also shared so you can just load the pre trained models.
- Start by installing the requirements file
`pip install -r requirements.txt`

**Automatically download and extract the weights from Box (933 MB)**
```python
python train.py --download_weights 1
```


## How to prune and fine tune pre trained models 
Check the `train.py` to see all available hyper-parameter choices.

`python train.py --classifier resnet18`
