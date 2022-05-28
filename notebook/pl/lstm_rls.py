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
import math
import warnings
warnings.filterwarnings('ignore')
RANDOM_SEED = 999
pl.seed_everything(RANDOM_SEED)
a = 1
# model


class BiLSTM(LightningModule):
    '''
    https://www.cnblogs.com/cxq1126/p/13504437.html
    '''
    # num_labels为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, seq_length, embedding_dim, num_labels,output_size =1 , hidden_dim=64,n_layers=1,dropout=0.5, lr=1e-3):
        super(BiLSTM, self).__init__()
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.calc_acc=tm.Accuracy()
        # self.embedding = nn.Embedding(seq_length, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    
    def _attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    
    def forward(self, x):
        # x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.permute(1,0,2)
        output, (final_hidden_state, final_cell_state) = self.rnn(x)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, attention = self._attention_net(output, query)
        logits = self.fc(attn_output)
        return logits

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = self.loss_fct(output, target.float())

        self.log_dict({
            'train_loss': loss
        })
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data).squeeze(1)
        loss = self.loss_fct(output, target.float())
        self.log_dict({
            'val_acc': self.calc_acc(output, target),
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
        loss = self.loss_fct(output.squeeze(), target.float())
        self.log('test_loss', loss)
        self.log('test_acc', self.calc_acc(output.squeeze(), target))
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
    LEARNING_RATE=0.01
    wandb.login()
    wandb_logger = WandbLogger(
        project='lstm-clf', save_dir='./wnb_logs/', offline=False, name='lstm-release')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath="./logs/checkpoints",
        filename='sample-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.003,
                                        patience=10,
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

    lstm = BiLSTM(seq_length=SEQ_LEN, embedding_dim=WEMD_LEN,
                  num_labels=2, lr=LEARNING_RATE,dropout=0.2,n_layers=1)
    data_module = CNNDataModule(batch_size=BATCH_SIZE, num_workers=8)
    trainer.fit(lstm, data_module)
    trainer.test(lstm, data_module)
    wandb.finish()



