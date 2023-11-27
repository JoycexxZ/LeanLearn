import os
import torch
import numpy as np
import streamlit as st
import time
from torch.optim import SGD
from torchmetrics import Accuracy
import torch.nn as nn

from utils.scheduler import WarmupCosineLR
from utils.loggings import Log
from utils.data import CIFAR10Data
from model.build_model import *


class Engine:
    def __init__(self, args, summary_writer):
        self.args = args
        
        self.seed_everything(args.seed)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        
        output_dir = "outputs/{}".format(args.model_name + "_" + args.dataset)
        output_tb_dir = "{}/tensorboard".format(output_dir)
        output_log = "{}/log_{}.txt".format(output_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_tb_dir, exist_ok=True)
        
        self.logger = Log(output_log)
        self.logger.info("Logger initialized.")
        
        if args.dataset == 'CIFAR10':
            data = CIFAR10Data(args)
            self.num_classes = 10
        else:
            raise NotImplementedError("Dataset not implemented.")
        self.train_loader = data.train_dataloader()
        self.test_loader = data.test_dataloader()
        
        self.__init_model()
        self.__init_optimizer()
        self.__init_accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.summary_writer = summary_writer
        
    @staticmethod
    def seed_everything(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def __init_model(self):
        self.model = all_classifiers[self.args.model_name]
        if self.args.pretrained:
            self.model.load_state_dict(torch.load(self.args.weight))
        self.model = self.model.cuda()
        
    def __init_optimizer(self):
        if self.args.optimizer == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise NotImplementedError
        
    def __init_accuracy(self):
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes).cuda()
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes).cuda()
        self.train_class_acc = Accuracy(task='multiclass', num_classes=self.num_classes, average=None).cuda()
        self.test_class_acc = Accuracy(task='multiclass', num_classes=self.num_classes, average=None).cuda()
    
    def train(self, mode='train', progress_bar_sec=None, st_log_window=None):
        progress_bar_text = "{}ing the model...".format(mode.capitalize())
        assert progress_bar_sec is not None
        progress_bar = progress_bar_sec.progress(0, text=progress_bar_text)
        if st_log_window is not None:
            log_id = self.logger.start_capture(st_log_window)
        
        if mode == 'train':
            epoch = self.args.train_epoch 
        elif mode == 'tune':
            epoch = self.args.tune_epoch
        scheduler = WarmupCosineLR(
                self.optimizer, 
                warmup_epochs=epoch * 0.3, 
                max_epochs=epoch
            )
        for e in range(epoch):
            self.model.train()
            self.train_acc.reset()
            self.test_class_acc.reset()
            for images, labels in self.train_loader:
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                acc = self.train_acc.update(output, labels)
                per_class_acc = self.train_class_acc.update(output, labels)
                loss.backward()
                self.optimizer.step()
            scheduler.step()
            
            acc = self.train_acc.compute()
            per_class_acc = self.train_class_acc.compute()
            self.logger.info("{} Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}".format(mode.capitalize(), e, loss, acc*100))
            self.summary_writer.add_scalar('Loss/{}'.format(mode), loss, e)
            self.summary_writer.add_scalar('Accuracy/{}'.format(mode), acc, e)
            if e == epoch - 1:
                self.summary_writer.add_class_acc('ClassAcc/{}'.format(mode), per_class_acc)
            
            if progress_bar is not None:
                progress_bar.progress((e+1) / epoch, text=progress_bar_text)
        self.logger.stop_capture(log_id)
        
    def test(self, progress_bar_sec=None, st_log_window=None):
        progress_bar_text = "Testing the model..."
        assert progress_bar_sec is not None
        progress_bar = progress_bar_sec.progress(0, text=progress_bar_text)
        if st_log_window is not None:
            log_id = self.logger.start_capture(st_log_window)
        
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.cuda(), labels.cuda()
                output = self.model(images)
                loss = self.criterion(output, labels)
                accuracy = self.test_acc.update(output, labels)
                class_accuracy = self.test_class_acc.update(output, labels)
                if progress_bar is not None:
                    progress_bar.progress((i+1) / len(self.test_loader), text=progress_bar_text)
        accuracy = self.test_acc.compute()
        class_accuracy = self.test_class_acc.compute()
        self.logger.info("Test Loss: {:.4f}, Accuracy: {:.2f}".format(loss, accuracy*100))
        self.logger.info("Classwise Accuracy: {}".format(class_accuracy.cpu().tolist()))
        self.summary_writer.add_class_acc('ClassAcc/test', class_accuracy)
        self.logger.stop_capture(log_id)
    
    def prune(self, st_log_window_1=None, st_log_window_2=None):
        if st_log_window_1 is not None:
            log_id = self.logger.start_capture(st_log_window_1, fmt='plain')
        self.logger.info("Model before pruning:")
        self.logger.info(self.model)
        self.logger.stop_capture(log_id)
            
        config_list = pruner_config[self.args.model_name]
        config_list[0]["sparse_ratio"] = self.args.sparse_ratio
        pruner = pruning_methods[self.args.pruner](self.model, config_list)
        _, masks = pruner.compress()
        pruner.unwrap_model()
        dummy_ip = torch.rand(64, 3, 32, 32).cuda()
        ModelSpeedup(
            model=self.model, dummy_input=dummy_ip, masks_or_file=masks
        ).speedup_model()
        self.__init_optimizer()
        
        if st_log_window_2 is not None:
            log_id = self.logger.start_capture(st_log_window_2, fmt='plain')
        self.logger.info("Model after pruning:")
        self.logger.info(self.model)
        self.logger.stop_capture(log_id)
        