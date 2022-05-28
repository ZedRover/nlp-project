import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from datamodule.mydata import *
from models.model import *
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning.plugins import DDPPlugin
import warnings
warnings.filterwarnings('ignore')
RANDOM_SEED = 999
pl.seed_everything(RANDOM_SEED)
a = 1
# model


class TextCNN(LightningModule):
    # num_labels为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, seq_length, embedding_dim, num_labels, filter_num=64, kernel_list=(3,4,5,30), dropout=0.5, lr=1e-3):
        super(TextCNN, self).__init__()
        # self.embedding = nn.Embedding(seq_length, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                nn.LeakyReLU(),
                nn.MaxPool2d((seq_length - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.loss_fct = nn.CrossEntropyLoss()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filter_num * len(kernel_list), 256)
        self.fc2 = nn.Linear(256,num_labels)

    def forward(self, x):
        # x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)   # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.fc(out)          # 结果输出[128, 2]
        out = self.dropout(out)        # 构建dropout层
        logits = self.fc2(out)
        return logits

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = self.loss_fct(output, target.long())

        self.log_dict({
            'train_loss': loss
        })
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = self.loss_fct(output, target.long())
        a, y_hat = torch.max(output, dim=1)
        self.log_dict({
            'val_acc': accuracy_score(target.cpu().detach().numpy(), y_hat.cpu().detach().numpy().astype(int)),
            'val_loss': loss
        })
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log_dict({
            'val_loss': avg_loss,
        })
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_fct(output, target.long())
        a,y_hat = torch.max(output,dim=1)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy_score(
             target.cpu().detach().numpy(), y_hat.cpu().detach().numpy().astype(int)))
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer


if __name__ == '__main__':
    NUM_GPU = 2
    MAX_EPOCH = 500
    BATCH_SIZE = 1024
    SEQ_LEN = 30
    WEMD_LEN = 300
    wandb.login()
    wandb_logger = WandbLogger(
        project='cnn-clf', save_dir='./wnb_logs/', offline=False)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./logs/checkpoints",
        filename='sample-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=True,
                                        mode="min")

    trainer = pl.Trainer(accelerator='gpu',
                         gpus=NUM_GPU,
                         default_root_dir='./logs',
                         max_epochs=MAX_EPOCH,
                         callbacks=[checkpoint_callback,
                                    early_stop_callback,
                                    ],
                         logger=wandb_logger,
                         plugins=DDPPlugin(find_unused_parameters=False),
                         # strategy="fsdp",
                         check_val_every_n_epoch=1,
                         # devices=NUM_GPU,
                         precision=16,
                         )

    cnn = TextCNN(seq_length=SEQ_LEN, embedding_dim=WEMD_LEN,
                  num_labels=2, lr=1e-4,dropout=0.2)
    data_module = CNNDataModule(batch_size=BATCH_SIZE, num_workers=8)
    trainer.fit(cnn, data_module)
    trainer.test(cnn, data_module)
    wandb.finish()
