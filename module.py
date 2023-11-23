import pytorch_lightning as pl
import torch

# from pytorch_lightning.metrics import Accuracy
from torchmetrics import Accuracy
from model.densenet import densenet121, densenet161, densenet169
from model.googlenet import googlenet
from model.inception import inception_v3
from model.mobilenetv2 import mobilenet_v2
from model.resnet import resnet18, resnet34, resnet50
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

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


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hyparams):
        super().__init__()
        self.hyparams = hyparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

        self.model = all_classifiers[hyparams.classifier]

        # code to get rid of params issue
        self.classifier = hyparams.classifier
        self.learning_rate = hyparams.learning_rate
        self.max_epochs = hyparams.max_epochs
        self.weight_decay = hyparams.weight_decay

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def prune(self):
        config_list = pruner_config[self.hyparams.classifier]
        config_list[0]["sparse_ratio"] = self.hyparams.sparse_ratio
        pruner = pruning_methods[self.hyparams.pruning_method](self.model, config_list)
        _, masks = pruner.compress()
        pruner.unwrap_model()
        dummy_ip = torch.rand(64, 3, 32, 32)
        ModelSpeedup(
            model=self.model, dummy_input=dummy_ip, masks_or_file=masks
        ).speedup_model()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.max_epochs * 50000
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
