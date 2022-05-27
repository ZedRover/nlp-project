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
        self.l1 = torch.nn.Linear(layer1, 512)
        self.l2 = torch.nn.Linear(512,num_class)
        self.dropout=torch.nn.Dropout(0.2)
        self.sm = torch.nn.Sigmoid()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.sm(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.sm(x)
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
    
    
    
    
    
class CNNModel(pl.LightningModule):

    def __init__(self,n_feat =300,num_class=1,lr=0.01):
        super().__init__()
        self.accuracy = tm.Accuracy()
        self.save_hyperparameters()
        self.lr = lr
        
        self.calc_loss = torch.nn.BCEWithLogitsLoss()
        
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=8, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(n_feat-3+1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(n_feat-3+1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3)
        self.maxpool3 = nn.MaxPool1d(n_feat-3+1)
        self.out = nn.Linear(1000,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x11 = self.maxpool1(x1)
        x2 = self.conv2(x)
        x21 = self.maxpool2(x2)
        x3 = self.conv3(x)
        x31 = self.maxpool3(x3)
        b, c, d, h, w = x11.size()
        x12 = x11.view(-1, c * d * h * w)
        x22 = x21.view(-1, c * d * h * w)
        x32 = x31.view(-1, c * d * h * w)
        # x4 = self.concat([x12,x22,x32])
        x5 = self.out(12)
        x = self.sig(x5)
        
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
    
    
