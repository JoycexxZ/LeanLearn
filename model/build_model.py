from model.densenet import densenet121, densenet161, densenet169
from model.googlenet import googlenet
from model.inception import inception_v3
from model.mobilenetv2 import mobilenet_v2
from model.resnet import resnet18, resnet34, resnet50
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Pruning
from nni.compression.pruning import *
from nni.compression.speedup import ModelSpeedup


all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

pruner_config = {
    "vgg11_bn": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["classifier.6"],
            "sparse_ratio": 0.6,
        }
    ],
    "vgg13_bn": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["classifier.6"],
            "sparse_ratio": 0.6,
        }
    ],
    "vgg19_bn": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["classifier.6"],
            "sparse_ratio": 0.6,
        }
    ],
    "vgg16_bn": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["classifier.6"],
            "sparse_ratio": 0.6,
        }
    ],
    "resnet18": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["fc"],
            "sparse_ratio": 0.6,
        }
    ],
    "resnet34": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["fc"],
            "sparse_ratio": 0.6,
        }
    ],
    "resnet50": [
        {
            "op_types": ["Conv2d", "Linear"],
            "exclude_op_names": ["fc"],
            "sparse_ratio": 0.6,
        }
    ],
    # "densenet121": {},
    # "densenet161": {},
    # "densenet169": {},
    "mobilenet_v2": [
        {
            "op_types": ["Conv2d", "Linear", "BatchNorm2d"],
            "exclude_op_names": ["classifier.1"],
            "sparse_ratio": 0.6,
        }
    ],
    # "googlenet": {},
    # "inception_v3": {},
}

pruning_methods = {
    "level": LevelPruner,
    "l1norm": L1NormPruner,
    "l2norm": L2NormPruner,
    "fpgm": FPGMPruner,
    "slim": SlimPruner,
    "taylor": TaylorPruner,
    "linear": LinearPruner,
    "agp": AGPPruner,
    "movement": MovementPruner,
}