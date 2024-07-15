import torch
from typing import Union
import numpy as np
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import wandb
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy,F1Score,ConfusionMatrix, MeanSquaredError
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import f1_score, accuracy_score
from models.model_att import Attention
from models.model_cmha import ModelCMHA
from models.model_cls import ModelCLS
from models.model_cross1d import ModelCross1d
from models.model_cross1dpre import ModelCross1dpre
from models.model_cmha1d import ModelCMHA1d
from models.model_cm1d import ModelCM1d
from models.model_multi import MultimodalModel
from models.model_mult import Mult
from models.pretrain import Pretrain


class MER(LightningModule):
    """Initialize a cross-modality attention Model using Pytorch Lightning. 
    
        Args: 
            lr = int (learning rate during training)
            loss_fn = loss function to use during training
            model = the model to train
            scheduler_patience = int (patience for learning rate scheduler)
            scheduler_factor = int (factor for learning rate scheduler)"""

    def __init__(self, model, loss_fn,loss_cont, learning_rate,scheduler, scheduler_patience, scheduler_factor,multi):
        super().__init__()
        self.model = model
        self.loss = loss_fn
        self.loss_cont = loss_cont
        self.lr = learning_rate
        self.sched = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.multi = multi
    
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """ The training step, relevant for training using Pytorch Lightning."""
        a,t,v,a1,t1,v1,y,y1 = batch
        loss = self(a,t,v,a1,t1,v1)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """ The validation step, relevant for training using Pytorch Lightning."""
        a,t,v,a1,t1,v1,y,y1 = batch
        loss = self(a,t,v,a1,t1,v1)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def configure_optimizers(self) -> dict:
        """ Obtain Adam optimizer and learning rate scheduler. """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=0.02)#,betas=(0.9, 0.98),eps=1e-06)
        #if self.sched=="ann":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        #elif self.sched=="exp":
         #   lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,total_iters=5)
        #elif self.sched=="reduce":
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience)
        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": lr_scheduler,"monitor": "val_loss","interval": "epoch"}}

class single_task_MER(MER):
    def __init__(self, model, loss_fn,loss_cont, learning_rate,scheduler,scheduler_patience, scheduler_factor,multi):
        super().__init__(model, loss_fn,loss_cont, learning_rate,scheduler,scheduler_patience, scheduler_factor,multi)
        self.acc = Accuracy(task="multiclass",num_classes=self.num_classes)
        self.mse = MeanSquaredError()
        #self.f1 = F1Score(task='multiclass', num_classes=self.num_classes)
        self.f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.conf_matrix = ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.multi = multi
    
    def test_step(self, batch: Tensor, batch_idx: int,dataloader_idx=0) -> Tensor:
        """ The test step, relevant for training using Pytorch Lightning."""
        a,t,v,a1,t1,v1,y,y1 = batch
        if self.multi:
            x_hat,x_hat1 = self(a,t,v,a1,t1,v1)
            x_hat1 = x_hat1.float() 
            mse = self.mse(x_hat1.squeeze(),y1)
        else:
            x_hat = self(a,t,v,a1,t1,v1)
        self.conf_matrix.update(x_hat, y)
        x_hat = torch.argmax(x_hat, dim=1)
        #accuracy = self.acc(x_hat,y)
        f1 = self.f1(x_hat,y)

        self.conf_matrix.update(x_hat, y)
        #self.log("test_accuracy", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_f1", f1, prog_bar=True, on_epoch=True, on_step=False)
        if self.multi:
            self.log("test_mse", mse, prog_bar=True, on_epoch=True, on_step=False)
            return {"test_f1": f1, "test_mse":mse}
        return {"test_f1": f1}
    
    def on_test_epoch_end(self):
        conf_matrix = self.conf_matrix.compute()
        self.logger.experiment.log({"Confusion Matrix": [wandb.Image(self.plot_confusion_matrix(conf_matrix), caption="CM")]})

    def plot_confusion_matrix(self, conf_matrix):
        # Function to plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        return fig

class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss

class classifier(single_task_MER):
    def __init__(self, model_name,multi,learning_rate,h_dim,num_classes,dropout_prob,scheduler,scheduler_patience,scheduler_factor,a,t,v,type):
        loss_fn =  CELoss()
        loss_cont = MSELoss()
        self.a = a
        self.t = t
        self.v = v
        self.multi = multi
        self.num_classes = num_classes
        input_shapes=[1024,4096,768]
        attention_model = Pretrain(input_shapes, h_dim,1, num_classes,dropout_prob,multi,type,self.a,self.t,self.v)
        super().__init__(classifier, loss_fn,loss_cont, learning_rate,scheduler,scheduler_patience,scheduler_factor,multi)
        self.fusion = attention_model
        
    def forward(self, a,t,v,a1,t1,v1):
        x=self.fusion(a,a1,t,t1,v,v1)
        return x