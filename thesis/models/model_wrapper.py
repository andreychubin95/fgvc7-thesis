import torch
import lightning as pl
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC


class ModelWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler: LambdaLR = None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_accuracy = MulticlassAccuracy(num_classes=4)
        self.val_accuracy = MulticlassAccuracy(num_classes=4)
        self.train_roc = MulticlassAUROC(num_classes=4)
        self.val_roc = MulticlassAUROC(num_classes=4)

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = self.train_accuracy(y_hat, y)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        roc = self.train_roc(y_hat, y)
        self.log('train_roc_auc', roc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = self.val_accuracy(y_hat, y)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        roc = self.val_roc(y_hat, y)
        self.log('val_roc_auc', roc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return {
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'monitor': 'val_loss'
            }
