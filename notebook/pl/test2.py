import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import  ModelCheckpoint, EarlyStopping
import transformers
from nlp import load_dataset
import torchmetrics as tm
from pytorch_lightning.loggers import WandbLogger
import wandb
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
import config
RANDOM_SEED = 999
pl.seed_everything(RANDOM_SEED)


df_pos = pd.read_excel(config.DATA_DIR+'/女权无标点.xlsx', index_col=0)
df_neg = pd.read_excel(config.DATA_DIR+'/非女权无标点.xlsx', index_col=0)

df_pos['target'] = 1
df_neg['target'] = 0
df = pd.concat([df_pos, df_neg], axis=0)
df.rename({0: 'reviews'}, inplace=True, axis=1)

df_insmp, df_test = train_test_split(df, test_size=0.2, shuffle=True)
df_train, df_valid = train_test_split(df_insmp, test_size=0.1, shuffle=False)


class ImdbDataset(Dataset):
    def __init__(self, notes, targets, tokenizer, max_len):
        self.notes = notes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
         
    def __len__(self):
        return (len(self.notes))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        note = str(self.notes.values.flatten()[idx])
        target = self.targets.values.flatten()[idx]
        
        encoding = self.tokenizer.encode_plus(
          note,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          truncation=True,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )    
        return {
            #'text': note,
            'label': torch.tensor(target, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }
        
        
BERT_MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
train_dataset = ImdbDataset(notes=df_train['reviews'],
    targets=df_train['target'],
    tokenizer=tokenizer,
    max_len=512)
val_dataset = ImdbDataset(notes=df_valid['reviews'],
    targets=df_valid['target'],
    tokenizer=tokenizer,
    max_len=512)
test_dataset = ImdbDataset(notes=df_test['reviews'],
    targets=df_test['target'],
    tokenizer=tokenizer,
    max_len=512)

class MyDataModule(pl.LightningDataModule):

    def __init__(self, train,val,test, batch_size: int = 64, num_workers: int = 20,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2
        self.train = train
        self.val = val
        self.test = test
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        pass
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    
datamodule = MyDataModule(train_dataset,val_dataset,test_dataset)
class ImdbModel(pl.LightningModule):

    def __init__(self,
                 learning_rate: float = 0.0001 * 8,
                 batch_size: int = 64,
                 num_workers: int = 20,
                 **kwargs):
        super().__init__()
        self.lr = learning_rate
        self.save_hyperparameters()
        self.sched = None
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(
            BERT_MODEL_NAME, return_dict=True)
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.acc = tm.Accuracy()
        # self.dropout = torch.nn.Dropout(self.bert.config.seq_classif_dropout)
        self.dropout=torch.nn.Dropout(0.2)
        # relu activation function
        self.sm = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, input_ids, attention_mask, labels):
      
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        logits = self.sm(logits)
        return logits
    
    def get_outputs(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        return pooled_output
        

    def training_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        #loss = F.cross_entropy(y_hat, label)
        self.log("train_loss", loss)
        self.log('train_acc',self.acc(y_hat.view(-1, self.num_labels), label.view(-1)))

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids'] 
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        #loss = F.cross_entropy(y_hat, label)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = self.acc(y_hat, label)
        self.log('val_loss',loss)
        self.log('val_acc', val_acc)
        
        

    # def validation_epoch_end(self, outputs):
    #     return None
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
    #     # logs
    #     self.log('avg_val_loss',avg_loss,'avg_val_acc',avg_val_acc)
     
    
    def on_batch_end(self):
        #for group in self.optim.param_groups:
        #    print('learning rate', group['lr'])
        # This is needed to use the One Cycle learning rate that needs the learning rate to change after every batch
        # Without this, the learning rate will only change after every epoch
        if self.sched is not None:
            self.sched.step()
    
    def on_epoch_end(self):
        if self.sched is not None:
            self.sched.step()

    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = self.acc(y_hat, label)
        self.log('test_acc',test_acc)

    # def test_epoch_end(self, outputs):
    #     return None
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

    #     tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_test_acc}
    #     self.log('avg_test_loss',avg_loss,'avg_test_acc',avg_test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
        
def main():

    """ Main training routine specific for this project. """

    wandb.login()
    wandb_logger = WandbLogger(project='bert-test')
    model = ImdbModel(learning_rate=1e-3)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
    trainer = pl.Trainer(
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=100,
    )
    trainer.fit(model,datamodule)
    trainer.test(model,datamodule)
    wandb.finish()
if __name__=='__main__':
    main()