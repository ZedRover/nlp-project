import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import numpy as np
import pandas as pd
import config
import torch
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pl_bolts.datamodules import SklearnDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics as tm
from pytorch_lightning.plugins import DDPPlugin
import warnings
warnings.filterwarnings('ignore')
from models.model import *
from datamodule.mydata import *

RANDOM_SEED=999
pl.seed_everything(RANDOM_SEED)
NUM_GPU=2
MAX_EPOCH=500
LEARNING_RATE = 1e-3
BATCH_SIZE=1024

if __name__=='__main__':
    wandb.login()
    wandb_logger = WandbLogger(
        project='mlp-clf', save_dir='./wnb_logs/', offline=False, name='mlp-release')
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
     verbose=True,
     mode="min")
    
    trainer = pl.Trainer(accelerator='auto',
                         max_epochs=MAX_EPOCH,
                        callbacks=[checkpoint_callback,
                                   early_stop_callback,
                                   ],
                        logger=wandb_logger,
                        # strategy="fsdp",
                        check_val_every_n_epoch=1,
                        plugins=DDPPlugin(find_unused_parameters=False),
                        gpus=NUM_GPU,
                        precision=16,
                        )
    model = BaseLineModel(lr=LEARNING_RATE)
    data_module = MlpDataModule(batch_size=BATCH_SIZE,num_workers=6)
    trainer.fit(model,data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

#  https://wandb.ai/sfcap/mlp-clf/runs/15pbbvv4