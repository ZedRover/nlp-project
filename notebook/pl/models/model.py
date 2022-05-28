import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
from torch import nn

class BaseLineModel(pl.LightningModule):

    def __init__(self,layer1 =300*30,num_class=1,lr=0.01):
        super().__init__()
        self.accuracy = tm.Accuracy()
        self.save_hyperparameters()
        self.lr = lr
        self.calc_loss = torch.nn.BCEWithLogitsLoss()
        self.l1 = torch.nn.Linear(layer1, 1024)
        self.l2 = torch.nn.Linear(1024,num_class)
        self.dropout=torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self(x).squeeze(1)
        loss = self.calc_loss(x, y.float())
        self.log('train_loss',loss)
        self.log('train_acc',self.accuracy(x,y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x).squeeze(1)
        loss = self.calc_loss(x, y.float())
        self.log('val_loss', loss)
        self.log('val_acc',self.accuracy(x,y))
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self(x).squeeze(1)
        loss = self.calc_loss(x, y.float())
        self.log('test_loss', loss)
        self.log('test_acc',self.accuracy(x,y))
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    
