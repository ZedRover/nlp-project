import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import  ModelCheckpoint, EarlyStopping
import transformers
from nlp import load_dataset
import torchmetrics as tm
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.model import *
from datamodule.mydata import *
import warnings
warnings.filterwarnings('ignore')
import config
RANDOM_SEED = 999
pl.seed_everything(RANDOM_SEED)
# model
class TextCNN(LightningModule):

    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5,lr=1e-3):
        super(TextCNN, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.input = nn.Linear(300,512)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                            # nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.Conv1d(in_channels=embedding_dim, out_channels=3, kernel_size=3),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((30 - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.lr = lr
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x (128,300)
        # x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1).unsqueeze(2)     # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)   # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)        # 构建dropout层
        logits = self.fc(out)          # 结果输出[128, 2]
        return logits
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = F.cross_entropy(output, target.long())
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = F.cross_entropy(output, target.long())
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target.long())
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        



if __name__=='__main__':
    NUM_GPU=2
    MAX_EPOCH=500
    BATCH_SIZE=128
    SEQ_LEN = 30
    WEMD_LEN = 300
    wandb.login()
    wandb_logger = WandbLogger(project='cnn')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='sample-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(monitor="val_loss",
     min_delta=0.00,
     patience=10,
     verbose=False,
     mode="min")
    
    trainer = pl.Trainer(accelerator='auto',
                         max_epochs=MAX_EPOCH,
                        callbacks=[checkpoint_callback,
                                   early_stop_callback,
                                   ],
                        logger=wandb_logger,
                        strategy="fsdp",
                        check_val_every_n_epoch=1,
                        devices=NUM_GPU,
                        precision=16,
                        )

    cnn = TextCNN(vocab_size=SEQ_LEN, embedding_dim=WEMD_LEN, output_size=2,lr=1e-5)
    data_module = CNNDataModule(batch_size=BATCH_SIZE,num_workers=10)
    trainer.fit(cnn,data_module)
    trainer.test(cnn, datamodule=data_module)
    wandb.finish()

